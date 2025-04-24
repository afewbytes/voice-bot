package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"os/signal"
	"syscall"
	"time"

	pb "mic-client/proto"

	"github.com/gordonklaus/portaudio"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const (
	serverAddr       = "localhost:8090"
	sampleRate       = 16000
	framesPerBuffer  = 1024
	channels         = 1
	bytesPerSample   = 2
	chunkDuration    = 1 * time.Second
	silenceRMS       = 500.0 // Base RMS threshold for VAD
	maxSilenceFrames = 10    // Number of consecutive silent frames to mark end of speech
	vadHoldTime      = 500   // Hold time in milliseconds after speech detected
	energySmoothing  = 0.7   // Smoothing factor for energy levels (0-1)
	minSpeechDur     = 300   // Minimum speech duration in milliseconds to consider valid
	preRollFrames    = 3     // Number of frames to include before speech is detected
	dynamicThreshold = true  // Whether to use dynamic thresholding
	thresholdFactor  = 1.5   // Factor above background noise to trigger VAD
	adaptRate        = 0.02  // Rate at which background noise level adapts
	initialAdapt     = 5     // Number of initial frames for noise profile
)

// MicSource implements audio source for microphone input
type MicSource struct {
	stream *portaudio.Stream
	buffer []int16
}

// NewMicSource creates a new microphone audio source
func NewMicSource() (*MicSource, error) {
	buffer := make([]int16, framesPerBuffer*channels)
	stream, err := portaudio.OpenDefaultStream(
		channels, 0, float64(sampleRate), framesPerBuffer, buffer,
	)
	if err != nil {
		return nil, fmt.Errorf("open mic: %v", err)
	}

	if err := stream.Start(); err != nil {
		stream.Close()
		return nil, fmt.Errorf("start mic: %v", err)
	}

	return &MicSource{
		stream: stream,
		buffer: buffer,
	}, nil
}

// ReadSamples reads audio samples from the microphone
func (m *MicSource) ReadSamples(samples []int16) (int, error) {
	if err := m.stream.Read(); err != nil {
		return 0, err
	}
	
	copy(samples, m.buffer)
	return len(m.buffer), nil
}

// Close closes the microphone stream
func (m *MicSource) Close() error {
	if err := m.stream.Stop(); err != nil {
		m.stream.Close()
		return err
	}
	return m.stream.Close()
}

// VAD state management
type VADState struct {
	active          bool // Whether speech is currently active
	silentFrames    int  // Count of consecutive silent frames
	speechFrames    int  // Count of consecutive speech frames
	lastSpeechTime  time.Time
	lastEnergy      float64
	noiseFloor      float64
	threshold       float64
	prerollBuffer   [][]byte // Buffer to store frames before speech starts
	adaptationCount int      // Count frames for initial adaptation
}

func newVADState() *VADState {
	return &VADState{
		active:          false,
		silentFrames:    0,
		speechFrames:    0,
		lastSpeechTime:  time.Time{},
		lastEnergy:      0,
		noiseFloor:      silenceRMS, // Start with default threshold
		threshold:       silenceRMS * thresholdFactor,
		prerollBuffer:   make([][]byte, preRollFrames),
		adaptationCount: 0,
	}
}

func (vad *VADState) reset() {
	vad.active = false
	vad.silentFrames = 0
	vad.speechFrames = 0
	vad.lastSpeechTime = time.Time{}
}

func main() {
	// Parse command line arguments (keeping flag for compatibility)
	flag.Parse()

	// Establish gRPC connection
	fmt.Printf("Dialing %s using gRPC…\n", serverAddr)
	conn, err := grpc.Dial(serverAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("did not connect: %v", err)
	}
	defer conn.Close()
	client := pb.NewWhisperServiceClient(conn)
	fmt.Println("Connected.")

	// Create a context and gRPC stream
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stream, err := client.StreamAudio(ctx)
	if err != nil {
		log.Fatalf("could not create stream: %v", err)
	}
	
	// Start transcript reader
	go receiveTranscriptions(stream)

	// Handle Ctrl+C for graceful shutdown
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)

	// Initialize PortAudio
	if err := portaudio.Initialize(); err != nil {
		log.Fatalf("portaudio init: %v", err)
	}
	defer portaudio.Terminate()

	// Create microphone source
	micSource, err := NewMicSource()
	if err != nil {
		log.Fatalf("Error creating microphone source: %v", err)
	}
	defer micSource.Close()
	
	fmt.Println("Streaming from microphone")

	// Process audio from the microphone
	processAudio(stream, micSource, sig)

	// Close the stream properly
	stream.CloseSend()
	
	fmt.Println("Disconnected from server")
}

// Process audio from microphone using VAD logic
func processAudio(stream pb.WhisperService_StreamAudioClient, source *MicSource, sig chan os.Signal) {
	fmt.Println("=== Streaming from Microphone with VAD ===")
	fmt.Println("Processing audio... (Press Ctrl+C to exit)")

	// State for VAD and buffering
	var accum []byte
	accum = make([]byte, 0, framesPerBuffer*bytesPerSample*4)
	minSize := framesPerBuffer * bytesPerSample
	timeout := 200 * time.Millisecond
	lastSend := time.Now()

	// Create VAD state
	vad := newVADState()

	// Circular buffer for preroll
	prerollIndex := 0
	
	// Buffer for reading samples
	audioBuf := make([]int16, framesPerBuffer)

	for {
		select {
		case <-sig:
			fmt.Println("\nShutting down...")
			
			// If we're in an active speech segment, send end marker
			if vad.active {
				if len(accum) > 0 {
					// Send any remaining audio
					err := stream.Send(&pb.AudioChunk{
						Data: accum,
					})
					if err != nil {
						log.Printf("send final audio: %v", err)
					}
				}
				
				// Send speech end marker
				err := stream.Send(&pb.AudioChunk{
					SpeechEnd: true,
				})
				if err != nil {
					log.Printf("send end marker: %v", err)
				}
				fmt.Println("[Speech end marker sent]")
			}
			
			return
		default:
			// Read samples from the microphone
			n, err := source.ReadSamples(audioBuf)
			if err != nil {
				log.Printf("read error: %v", err)
				continue
			}

			if n == 0 {
				continue
			}

			// Pack int16 samples into bytes for sending
			sendBuf := make([]byte, framesPerBuffer*bytesPerSample)
			for i, s := range audioBuf {
				sendBuf[i*2] = byte(s & 0xFF)
				sendBuf[i*2+1] = byte((s >> 8) & 0xFF)
			}

			// Store in preroll buffer
			if vad.prerollBuffer[prerollIndex] == nil {
				vad.prerollBuffer[prerollIndex] = make([]byte, len(sendBuf))
			}
			copy(vad.prerollBuffer[prerollIndex], sendBuf)
			prerollIndex = (prerollIndex + 1) % preRollFrames

			// Compute RMS for energy detection
			rms := calcRMS(audioBuf[:n])

			// Smooth the energy level
			vad.lastEnergy = vad.lastEnergy*energySmoothing + rms*(1-energySmoothing)

			// Initial adaptation period to establish noise floor
			if vad.adaptationCount < initialAdapt {
				vad.adaptationCount++
				// During initial adaptation, just accumulate noise floor
				vad.noiseFloor = (vad.noiseFloor*(float64(vad.adaptationCount)-1) + rms) / float64(vad.adaptationCount)
				vad.threshold = vad.noiseFloor * thresholdFactor
				if vad.adaptationCount == initialAdapt {
					fmt.Printf("\n[VAD calibrated: noise=%.1f, threshold=%.1f]\n", vad.noiseFloor, vad.threshold)
				}
				continue
			}

			// Dynamic threshold adjustment when not in speech
			if dynamicThreshold && !vad.active {
				// Slowly adapt noise floor during silence
				vad.noiseFloor = vad.noiseFloor*(1-adaptRate) + rms*adaptRate
				vad.threshold = vad.noiseFloor * thresholdFactor
			}

			isSpeech := vad.lastEnergy >= vad.threshold

			// VAD state machine
			if isSpeech {
				// Speech detected
				vad.speechFrames++
				vad.silentFrames = 0
				vad.lastSpeechTime = time.Now()

				if !vad.active && vad.speechFrames >= 2 { // Require at least 2 frames of speech to trigger
					// Speech start
					vad.active = true
					fmt.Println("\n[Speech detected]")

					// Send speech start marker using gRPC message
					fmt.Println("[Sending speech start marker]")
					err := stream.Send(&pb.AudioChunk{
						SpeechStart: true,
					})
					if err != nil {
						log.Printf("send start marker: %v", err)
						return
					}
					fmt.Println("[Speech start marker sent]")

					// Add preroll buffer to accumulator
					for i := 0; i < preRollFrames; i++ {
						idx := (prerollIndex + i) % preRollFrames
						if vad.prerollBuffer[idx] != nil {
							accum = append(accum, vad.prerollBuffer[idx]...)
						}
					}
				}

				if vad.active {
					accum = append(accum, sendBuf...)
				}
			} else {
				// Silence detected
				vad.speechFrames = 0

				if vad.active {
					// Check for end of speech
					holdTimeExpired := time.Since(vad.lastSpeechTime) > time.Duration(vadHoldTime)*time.Millisecond

					if holdTimeExpired {
						vad.silentFrames++
					} else {
						// Still within hold time, treat as speech
						accum = append(accum, sendBuf...)
					}

					if vad.silentFrames > maxSilenceFrames {
						// End of speech detected
						speechDur := time.Since(vad.lastSpeechTime).Milliseconds() + vadHoldTime

						if speechDur < minSpeechDur {
							fmt.Println("\n[Too short, discarding]")
							accum = accum[:0] // Discard too short utterances
						} else {
							fmt.Println("\n[Speech ended]")

							// Send any remaining audio before the end marker
							if len(accum) > 0 {
								err := stream.Send(&pb.AudioChunk{
									Data: accum,
								})
								if err != nil {
									log.Printf("send final audio: %v", err)
								}
								accum = accum[:0]
							}

							// Send speech end marker
							fmt.Println("[Sending speech end marker]")
							err := stream.Send(&pb.AudioChunk{
								SpeechEnd: true,
							})
							if err != nil {
								log.Printf("send end marker: %v", err)
							}
							fmt.Println("[Speech end marker sent]")
						}

						vad.active = false
						vad.silentFrames = 0
						vad.speechFrames = 0
					} else if holdTimeExpired {
						// During silence counting, still include frames
						accum = append(accum, sendBuf...)
					}
				}
			}

			// Send logic
			if vad.active {
				// Send during active speech periodically
				if len(accum) >= minSize || time.Since(lastSend) >= timeout {
					err := stream.Send(&pb.AudioChunk{
						Data: accum,
					})
					if err != nil {
						log.Printf("send audio: %v", err)
						return
					}
					accum = accum[:0]
					lastSend = time.Now()
				}
			}
		}
	}
}

// receiveTranscriptions reads and displays transcripts from the gRPC stream
func receiveTranscriptions(stream pb.WhisperService_StreamAudioClient) {
	for {
		resp, err := stream.Recv()
		if err == io.EOF {
			return
		}
		if err != nil {
			log.Printf("Stream receive error: %v", err)
			return
		}

		// Clear the current line and move to start
		fmt.Print("\r\033[K")  // \r moves cursor to start of line, \033[K clears to end of line
		
		// Print the transcription with a prefix
		text := resp.GetText()
		if text != "" {
			fmt.Print("TRANSCRIPT: " + text)
			// If it doesn't end with a newline, add one
			if len(text) > 0 && text[len(text)-1] != '\n' {
				fmt.Println()
			}
		}
	}
}

// calcRMS computes the root-mean-square of samples
func calcRMS(buf []int16) float64 {
	if len(buf) == 0 {
		return 0
	}
	var sum float64
	for _, s := range buf {
		sum += float64(s) * float64(s)
	}
	mean := sum / float64(len(buf))
	return math.Sqrt(mean)
}