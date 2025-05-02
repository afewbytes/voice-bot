// mic-client – streams mic to orchestrator, plays back F5-TTS audio
package main

import (
    "context"
    "encoding/binary"
    "flag"
    "fmt"
    "io"
    "log"
    "math"
    "os"
    "os/signal"
    "sync"
    "syscall"
    "time"
    "strings"

    pb "mic-client/proto"

    "github.com/gordonklaus/portaudio"
    "google.golang.org/grpc"
    "google.golang.org/grpc/credentials/insecure"
)

const (
    serverAddr       = "5.9.83.252:8090"
    sampleRate       = 16000          // mic input rate
    framesPerBuffer  = 1048
    channels         = 1
    bytesPerSample   = 2
    chunkDuration    = 1 * time.Second
    silenceRMS       = 1200.0
    maxSilenceFrames = 10
    vadHoldTime      = 500
    energySmoothing  = 0.7
    minSpeechDur     = 300
    preRollFrames    = 3
    dynamicThreshold = true
    thresholdFactor  = 4.0
    adaptRate        = 0.02
    initialAdapt     = 15
)

// ───────────── microphone source ─────────────────────────────────────────
type MicSource struct {
    stream *portaudio.Stream
    buffer []int16
}

func NewMicSource() (*MicSource, error) {
    buffer := make([]int16, framesPerBuffer*channels)
    stream, err := portaudio.OpenDefaultStream(
        channels, 0, float64(sampleRate), framesPerBuffer, buffer)
    if err != nil {
        return nil, fmt.Errorf("open mic: %v", err)
    }
    if err := stream.Start(); err != nil {
        _ = stream.Close()
        return nil, fmt.Errorf("start mic: %v", err)
    }
    return &MicSource{stream: stream, buffer: buffer}, nil
}

func (m *MicSource) ReadSamples(samples []int16) (int, error) {
    if err := m.stream.Read(); err != nil {
        return 0, err
    }
    copy(samples, m.buffer)
    return len(m.buffer), nil
}

func (m *MicSource) Close() error {
    if err := m.stream.Stop(); err != nil {
        _ = m.stream.Close()
        return err
    }
    return m.stream.Close()
}

// ───────────── audio player (TTS output) ────────────────────────────────
type AudioPlayer struct {
    stream      *portaudio.Stream
    buffer      []int16
    bufferSize  int
    sampleRate  float64
    mutex       sync.Mutex
    audioQueue  [][]int16
    queueSignal chan struct{}
    isPlaying   bool
    done        chan struct{}
}

func NewAudioPlayer() *AudioPlayer {
    bufSize := 1024
    return &AudioPlayer{
        buffer:      make([]int16, bufSize),
        bufferSize:  bufSize,
        audioQueue:  make([][]int16, 0),
        queueSignal: make(chan struct{}, 1),
        done:        make(chan struct{}),
    }
}

func (ap *AudioPlayer) ensureStream() error {
    // called with ap.mutex LOCKED
    if ap.stream != nil {
        return nil
    }
    s, err := portaudio.OpenDefaultStream(
        0, channels, ap.sampleRate, ap.bufferSize, ap.buffer)
    if err != nil {
        return err
    }
    if err = s.Start(); err != nil {
        _ = s.Close()
        return err
    }
    ap.stream = s
    ap.isPlaying = true
    return nil
}

func (ap *AudioPlayer) EnqueueAudio(audioBytes []byte, sampleRate int32) {
    if len(audioBytes) == 0 {
        return
    }

    // ── bytes → int16 little-endian (portable & sign-correct) ────────────
    n := len(audioBytes) / 2
    samples := make([]int16, n)
    for i := 0; i < n; i++ {
        samples[i] = int16(binary.LittleEndian.Uint16(
            audioBytes[i*2 : i*2+2]))
    }

    ap.mutex.Lock()
    // Re-open stream if rate changed
    if float64(sampleRate) != ap.sampleRate {
        if ap.stream != nil {
            _ = ap.stream.Stop()
            _ = ap.stream.Close()
            ap.stream = nil
            ap.isPlaying = false
        }
        ap.sampleRate = float64(sampleRate)
    }
    ap.audioQueue = append(ap.audioQueue, samples)
    ap.mutex.Unlock()

    select { case ap.queueSignal <- struct{}{}: default: }
}

func (ap *AudioPlayer) processAudioQueue() {
    defer close(ap.done)
    for {
        <-ap.queueSignal

        for {
            ap.mutex.Lock()
            if len(ap.audioQueue) == 0 {
                if ap.stream != nil {
                    _ = ap.stream.Stop()
                    _ = ap.stream.Close()
                    ap.stream = nil
                    ap.isPlaying = false
                }
                ap.mutex.Unlock()
                break
            }

            if err := ap.ensureStream(); err != nil {
                log.Printf("audio stream error: %v", err)
                ap.audioQueue = nil
                ap.mutex.Unlock()
                break
            }

            chunk := ap.audioQueue[0]
            ap.audioQueue = ap.audioQueue[1:]
            ap.mutex.Unlock()

            for i := 0; i < len(chunk); i += ap.bufferSize {
                end := i + ap.bufferSize
                if end > len(chunk) {
                    end = len(chunk)
                }
                copy(ap.buffer, chunk[i:end])
                if end-i < ap.bufferSize {
                    for j := end - i; j < ap.bufferSize; j++ {
                        ap.buffer[j] = 0
                    }
                }
                if err := ap.stream.Write(); err != nil {
                    log.Printf("write stream: %v", err)
                    break
                }
            }
        }
    }
}

func (ap *AudioPlayer) Close() error {
    ap.mutex.Lock()
    defer ap.mutex.Unlock()
    if ap.stream != nil {
        _ = ap.stream.Stop()
        return ap.stream.Close()
    }
    return nil
}

func (ap *AudioPlayer) IsCurrentlyPlaying() bool {
    ap.mutex.Lock()
    defer ap.mutex.Unlock()
    return ap.isPlaying
}

// ───────────── entry point ───────────────────────────────────────────────
func main() {
    flag.Parse()

    // ── gRPC dial ────────────────────────────────────────────────────────
    fmt.Printf("Dialing %s …\n", serverAddr)
    conn, err := grpc.Dial(serverAddr,
        grpc.WithTransportCredentials(insecure.NewCredentials()))
    if err != nil { log.Fatalf("dial: %v", err) }
    defer conn.Close()
    client := pb.NewWhisperServiceClient(conn)
    fmt.Println("Connected.")

    // ── PortAudio init ───────────────────────────────────────────────────
    if err := portaudio.Initialize(); err != nil {
        log.Fatalf("portaudio init: %v", err)
    }
    defer portaudio.Terminate()

    audioPlayer := NewAudioPlayer()
    go audioPlayer.processAudioQueue()
    defer audioPlayer.Close()

    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()
    stream, err := client.StreamAudio(ctx)
    if err != nil { log.Fatalf("stream: %v", err) }

    // ── Ctrl-C handler ───────────────────────────────────────────────────
    sig := make(chan os.Signal, 1)
    signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)

    micSource, err := NewMicSource()
    if err != nil { log.Fatalf("mic: %v", err) }
    defer micSource.Close()

    go receiveResponses(stream, audioPlayer)

    fmt.Println("Streaming microphone …   (Ctrl-C to quit)")
    processAudio(stream, micSource, audioPlayer, sig)

    stream.CloseSend()
    time.Sleep(500 * time.Millisecond)
    fmt.Println("Disconnected.")
}

// Process audio from microphone using VAD logic
func processAudio(stream pb.WhisperService_StreamAudioClient, source *MicSource, player *AudioPlayer, sig chan os.Signal) {
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
			// Skip processing if TTS is playing
			// If TTS is playing we still need to drain the mic buffer
			if player.IsCurrentlyPlaying() {
				_, _ = source.ReadSamples(audioBuf) // throw away the samples
				continue
			}
			
			// Read samples from the microphone
			n, err := source.ReadSamples(audioBuf)
			if err != nil {
				if err == portaudio.InputOverflowed {
					// We fell behind – drop the buffered frames and keep going.
					continue
				}
				log.Printf("mic read: %v", err)
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

			//lastMeterPrint := time.Now()

			//if time.Since(lastMeterPrint) > 250*time.Millisecond {
				fmt.Printf("\rRMS:%6.0f  Noise:%6.0f  Th:%6.0f  Act:%v     ",
						rms, vad.noiseFloor, vad.threshold, vad.active)
				//lastMeterPrint = time.Now()
			//}

			isSpeech := vad.lastEnergy >= vad.threshold

			// VAD state machine
			if isSpeech {
				// Speech detected
				vad.speechFrames++
				vad.silentFrames = 0
				vad.lastSpeechTime = time.Now()

				if !vad.active && vad.speechFrames >= 4 { // Require at least 2 frames of speech to trigger
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

// receiveResponses reads and displays transcripts and plays audio from the gRPC stream
func receiveResponses(stream pb.WhisperService_StreamAudioClient, player *AudioPlayer) {
	// Current state of displayed transcripts
	var currentWhisperText string
	var llamaBuf strings.Builder 

	for {
		resp, err := stream.Recv()
		if err == io.EOF {
			return
		}
		if err != nil {
			log.Printf("Stream receive error: %v", err)
			return
		}

		// Handle different sources
		switch resp.GetSource() {
		case pb.StreamAudioResponse_WHISPER:
			// Update transcription text
			text := resp.GetText()
			if text != "" {
				currentWhisperText = text
				// Clear the current line and prepare to display
				fmt.Print("\r\033[K")  // \r moves cursor to start of line, \033[K clears to end of line
				// Print transcription with prefix
				fmt.Print("TRANSCRIPT: " + currentWhisperText)
				// Add newline if needed
				if len(text) > 0 && text[len(text)-1] != '\n' {
					fmt.Println()
				}
			}
			
		case pb.StreamAudioResponse_LLAMA:
			txt := resp.GetText()
	
			// 1.  Always append whatever text we got (might be empty)
			if txt != "" {
				llamaBuf.WriteString(txt)
			}
	
			// 2.  If this is the final packet for this answer, print & reset
			if resp.GetDone() {
				if llamaBuf.Len() > 0 {                 // ignore spurious empty answers
					fmt.Print("\r\033[K")               // clear VU-meter line
					fmt.Printf("[LLAMA]: %s\n", llamaBuf.String())
					llamaBuf.Reset()                    // ready for the next answer
				}
			}
		
		case pb.StreamAudioResponse_TTS:
			// Play the TTS audio chunks
			audioData := resp.GetAudioData()
			sampleRate := resp.GetSampleRate()
			isEndAudio := resp.GetIsEndAudio()
			
			if len(audioData) > 0 {
				// Enqueue audio for playback
				player.EnqueueAudio(audioData, sampleRate)
			} else if isEndAudio {
				fmt.Println("\r\033[KTTS: Audio playback complete")
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