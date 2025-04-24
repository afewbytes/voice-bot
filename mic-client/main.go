package main

import (
	"fmt"
	"log"
	"math"
	"net"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gordonklaus/portaudio"
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
	initialAdapt     = 30    // Number of initial frames for noise profile
)

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
	// Initialize PortAudio
	if err := portaudio.Initialize(); err != nil {
		log.Fatalf("portaudio init: %v", err)
	}
	defer portaudio.Terminate()

	// Connect to our local audio-to-text proxy
	fmt.Printf("Dialing %s…\n", serverAddr)
	conn, err := net.Dial("tcp", serverAddr)
	if err != nil {
		log.Fatalf("dial error: %v", err)
	}
	defer conn.Close()
	fmt.Println("Connected.")

	// Start transcript reader
	go pumpTranscripts(conn)

	// Set up microphone stream
	audioBuf := make([]int16, framesPerBuffer*channels)
	stream, err := portaudio.OpenDefaultStream(
		channels, 0, float64(sampleRate), framesPerBuffer, audioBuf,
	)
	if err != nil {
		log.Fatalf("open mic: %v", err)
	}
	defer stream.Close()
	if err := stream.Start(); err != nil {
		log.Fatalf("start mic: %v", err)
	}
	defer stream.Stop()

	// Handle Ctrl+C for graceful shutdown
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)

	fmt.Println("=== Microphone Streaming Client with Enhanced VAD ===")
	fmt.Println("Start speaking (Press Ctrl+C to exit)...")

	// State for VAD and buffering
	var accum []byte
	accum = make([]byte, 0, framesPerBuffer*bytesPerSample*4)
	minSize := framesPerBuffer * bytesPerSample // send by buffer
	timeout := 200 * time.Millisecond
	lastSend := time.Now()

	// Create VAD state
	vad := newVADState()

	// Circular buffer for preroll
	prerollIndex := 0

loop:
	for {
		select {
		case <-sig:
			fmt.Println("\nShutting down...")
			break loop
		default:
			if err := stream.Read(); err != nil {
				log.Printf("mic read: %v", err)
				continue
			}

			// Pack int16 samples into bytes
			sendBuf := make([]byte, len(audioBuf)*bytesPerSample)
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
			rms := calcRMS(audioBuf)

			// Smooth the energy level
			vad.lastEnergy = vad.lastEnergy*energySmoothing + rms*(1-energySmoothing)

			// Initial adaptation period to establish noise floor
			if vad.adaptationCount < initialAdapt {
				vad.adaptationCount++
				// During initial adaptation, just accumulate noise floor
				vad.noiseFloor = (vad.noiseFloor*(float64(vad.adaptationCount)-1) + rms) / float64(vad.adaptationCount)
				vad.threshold = vad.noiseFloor * thresholdFactor
				if vad.adaptationCount == initialAdapt {
					fmt.Printf("[VAD calibrated: noise=%.1f, threshold=%.1f]\n", vad.noiseFloor, vad.threshold)
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
					fmt.Println("[Speech detected]")

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
							fmt.Println("[Too short, discarding]")
							accum = accum[:0] // Discard too short utterances
						} else {
							fmt.Println("[Speech ended]")
							// Final packet will be sent below
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
					if _, err := conn.Write(accum); err != nil {
						log.Printf("send audio: %v", err)
						break loop
					}
					accum = accum[:0]
					lastSend = time.Now()
				}
			} else if len(accum) > 0 {
				// Send final packet after speech ends
				if _, err := conn.Write(accum); err != nil {
					log.Printf("send audio: %v", err)
				}
				accum = accum[:0]
			}
		}
	}

	fmt.Println("Disconnected from server")
}

// pumpTranscripts reads and displays in-place updates from the server
type rawConn = net.Conn

func pumpTranscripts(conn net.Conn) {
	buf := make([]byte, 4096)
	for {
		n, err := conn.Read(buf)
		if err != nil {
			return
		}
		os.Stdout.Write(buf[:n])
	}
}

// calcRMS computes the root-mean-square of samples
type sampleBuf = []int16

func calcRMS(buf sampleBuf) float64 {
	var sum float64
	for _, s := range buf {
		sum += float64(s) * float64(s)
	}
	mean := sum / float64(len(buf))
	return math.Sqrt(mean)
}
