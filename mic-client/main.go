package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"io"
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

// Protocol markers for speech boundaries
var (
	speechStartMarker = []byte{0xFF, 0x00, 0x00, 0x01} // Unique marker for speech start
	speechEndMarker   = []byte{0xFF, 0x00, 0x00, 0x02} // Unique marker for speech end
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

// WAV file header structure
type wavHeader struct {
	RiffMarker      [4]byte  // "RIFF"
	FileSize        uint32   // File size - 8
	WaveMarker      [4]byte  // "WAVE"
	FmtMarker       [4]byte  // "fmt "
	FmtLength       uint32   // Length of format chunk
	FormatType      uint16   // Format type (1 = PCM)
	NumChannels     uint16   // Number of channels
	SampleRate      uint32   // Sample rate
	ByteRate        uint32   // Byte rate
	BlockAlign      uint16   // Block align
	BitsPerSample   uint16   // Bits per sample
	DataMarker      [4]byte  // "data"
	DataSize        uint32   // Size of data chunk
}

func main() {
	// Parse command line arguments
	wavFile := flag.String("file", "", "Path to WAV file (optional)")
	flag.Parse()

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

	// Handle Ctrl+C for graceful shutdown
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)

	// Check if WAV file was provided
	if *wavFile != "" {
		// Stream WAV file to server
		fmt.Printf("Streaming WAV file: %s\n", *wavFile)
		streamWavFile(*wavFile, conn, sig)
	} else {
		// Initialize PortAudio for microphone input
		if err := portaudio.Initialize(); err != nil {
			log.Fatalf("portaudio init: %v", err)
		}
		defer portaudio.Terminate()

		// Stream from microphone
		streamFromMic(conn, sig)
	}

	fmt.Println("Disconnected from server")
}

// streamWavFile reads a WAV file and streams it to the server as a single speech segment
func streamWavFile(wavFilePath string, conn net.Conn, sig chan os.Signal) {
	// Open the WAV file
	file, err := os.Open(wavFilePath)
	if err != nil {
		log.Fatalf("failed to open WAV file: %v", err)
	}
	defer file.Close()

	// Read and validate WAV header
	var header wavHeader
	if err := binary.Read(file, binary.LittleEndian, &header); err != nil {
		log.Fatalf("failed to read WAV header: %v", err)
	}

	// Validate WAV format
	if string(header.RiffMarker[:]) != "RIFF" || 
	   string(header.WaveMarker[:]) != "WAVE" || 
	   string(header.FmtMarker[:]) != "fmt " || 
	   header.FormatType != 1 { // PCM format
		log.Fatalf("unsupported WAV format or invalid WAV file")
	}

	// Find data chunk if not immediately following format chunk
	if string(header.DataMarker[:]) != "data" {
		// Skip until we find the data marker
		marker := make([]byte, 4)
		var dataSize uint32
		for {
			if _, err := file.Read(marker); err != nil {
				log.Fatalf("failed to find data chunk: %v", err)
			}
			if string(marker) == "data" {
				// Read data size
				if err := binary.Read(file, binary.LittleEndian, &dataSize); err != nil {
					log.Fatalf("failed to read data size: %v", err)
				}
				break
			}
			// Skip one byte and try again
			if _, err := file.Seek(-3, io.SeekCurrent); err != nil {
				log.Fatalf("seek error: %v", err)
			}
		}
		header.DataSize = dataSize
	}

	fmt.Printf("WAV details: %d Hz, %d channels, %d bits per sample\n", 
		header.SampleRate, header.NumChannels, header.BitsPerSample)

	// If sample rate doesn't match our expected rate, warn user
	if header.SampleRate != sampleRate {
		fmt.Printf("Warning: WAV sample rate (%d) differs from expected rate (%d). Results may be affected.\n",
			header.SampleRate, sampleRate)
	}

	// If not 16-bit samples, warn user
	if header.BitsPerSample != 16 {
		fmt.Printf("Warning: WAV bits per sample (%d) is not 16-bit. Results may be affected.\n",
			header.BitsPerSample)
	}

	// If not mono, warn user
	if header.NumChannels != 1 {
		fmt.Printf("Warning: WAV is not mono (%d channels). Only first channel will be used.\n",
			header.NumChannels)
	}

	fmt.Println("=== WAV File Streaming ===")
	fmt.Println("Streaming file as a single speech segment... (Press Ctrl+C to exit)")

	// First, send a speech start marker to indicate beginning of speech
	fmt.Println("[Sending speech start marker]")
	if _, err := conn.Write(speechStartMarker); err != nil {
		log.Fatalf("failed to send speech start marker: %v", err)
	}
	
	// Wait a moment to ensure the start marker is processed
	time.Sleep(100 * time.Millisecond)
	
	// Buffer for reading samples - read smaller chunks to avoid overwhelming the connection
	chunkSize := framesPerBuffer * int(header.NumChannels) * bytesPerSample
	bytesBuf := make([]byte, chunkSize)
	
	// Stream data from file in chunks
	totalSent := 0
	var readErr error
	for {
		select {
		case <-sig:
			fmt.Println("\nShutting down...")
			// Send speech end marker before exiting
			if _, err := conn.Write(speechEndMarker); err != nil {
				log.Printf("failed to send speech end marker: %v", err)
			}
			return
		default:
			// Read a chunk of WAV data
			n, err := file.Read(bytesBuf)
			readErr = err
			
			if err != nil && err != io.EOF {
				log.Printf("error reading WAV file: %v", err)
				return
			}
			
			if n == 0 {
				break
			}
			
			// If multi-channel, extract only the first channel
			if header.NumChannels > 1 {
				singleChannelData := make([]byte, n/int(header.NumChannels))
				for i := 0; i < n/int(header.NumChannels)/bytesPerSample; i++ {
					// Copy only the first channel's data
					singleChannelData[i*bytesPerSample] = bytesBuf[i*int(header.NumChannels)*bytesPerSample]
					singleChannelData[i*bytesPerSample+1] = bytesBuf[i*int(header.NumChannels)*bytesPerSample+1]
				}
				// Send the single channel data
				if _, err := conn.Write(singleChannelData); err != nil {
					log.Printf("failed to send audio data: %v", err)
					return
				}
				totalSent += len(singleChannelData)
			} else {
				// Send the mono data directly
				if _, err := conn.Write(bytesBuf[:n]); err != nil {
					log.Printf("failed to send audio data: %v", err)
					return
				}
				totalSent += n
			}
			
			// Print progress
			seconds := float64(totalSent) / float64(sampleRate*bytesPerSample)
			fmt.Printf("\rStreaming... %.2f seconds of audio sent", seconds)
			
			// Simulate real-time pace to avoid overloading the server
			frameCount := n / (int(header.NumChannels) * bytesPerSample)
			sleepDuration := time.Duration(frameCount) * time.Second / time.Duration(sampleRate)
			time.Sleep(sleepDuration)
		}
		
		// If we reach EOF, break out of the loop
		if readErr == io.EOF {
			break
		}
	}
	
	fmt.Println("\nFile streaming complete, sending end marker")
	
	// Send speech end marker to indicate end of speech
	if _, err := conn.Write(speechEndMarker); err != nil {
		log.Printf("failed to send speech end marker: %v", err)
	}
	
	fmt.Println("[Speech end marker sent]")
	fmt.Println("Waiting for transcription...")
	
	// Wait for transcription to complete
	time.Sleep(3 * time.Second)
	
	fmt.Println("Done")
}

// streamFromMic streams audio from the microphone to the server
func streamFromMic(conn net.Conn, sig chan os.Signal) {
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

	for {
		select {
		case <-sig:
			fmt.Println("\nShutting down...")
			return
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

					// Send speech start marker
					if _, err := conn.Write(speechStartMarker); err != nil {
						log.Printf("send start marker: %v", err)
						return
					}

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

							// Send any remaining audio before the end marker
							if len(accum) > 0 {
								if _, err := conn.Write(accum); err != nil {
									log.Printf("send final audio: %v", err)
								}
								accum = accum[:0]
							}

							// Send speech end marker
							if _, err := conn.Write(speechEndMarker); err != nil {
								log.Printf("send end marker: %v", err)
							}
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
						return
					}
					accum = accum[:0]
					lastSend = time.Now()
				}
			}
		}
	}
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
		
		// Clear the current line and move to start
		fmt.Print("\r\033[K")  // \r moves cursor to start of line, \033[K clears to end of line
		
		// Print the transcription with a prefix
		fmt.Print("TRANSCRIPT: ")
		os.Stdout.Write(buf[:n])
		
		// If it doesn't end with a newline, add one
		if n > 0 && buf[n-1] != '\n' {
			fmt.Println()
		}
	}
}

// calcRMS computes the root-mean-square of samples
type sampleBuf = []int16

func calcRMS(buf sampleBuf) float64 {
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