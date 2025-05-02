// mic-client – streams microphone to the orchestrator and plays back F5-TTS audio
//
// build: go build -o mic-client .
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
	"strings"
	"sync"
	"syscall"
	"time"

	pb "mic-client/proto" // ← adjust import path to your generated stubs

	"github.com/gordonklaus/portaudio"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const (
	serverAddr = "5.9.83.252:8090"

	/* microphone capture */
	micSampleRate   = 16_000
	framesPerBuffer = 1048
	micChannels     = 1
	bytesPerSample  = 2 /* int16 */

	/* VAD parameters (now using a fixed threshold) */
	silenceRMS       = 1200.0
	maxSilenceFrames = 10
	vadHoldTime      = 500 // ms
	energySmoothing  = 0.7
	minSpeechDur     = 300 // ms
	preRollFrames    = 3
	dynamicThreshold = false // ← fixed after initial calibration
	thresholdFactor  = 4.0
	adaptRate        = 0.02  // kept, but unreachable when dynamicThreshold == false
	initialAdapt     = 15
)

/* ───────────────────────── microphone source ───────────────────────────── */

type MicSource struct {
	stream *portaudio.Stream
	buffer []int16
}

func NewMicSource() (*MicSource, error) {
	buf := make([]int16, framesPerBuffer*micChannels)
	stream, err := portaudio.OpenDefaultStream(
		micChannels, 0, float64(micSampleRate), framesPerBuffer, buf,
	)
	if err != nil {
		return nil, fmt.Errorf("open mic: %v", err)
	}
	if err := stream.Start(); err != nil {
		_ = stream.Close()
		return nil, fmt.Errorf("start mic: %v", err)
	}
	return &MicSource{stream: stream, buffer: buf}, nil
}

func (m *MicSource) ReadSamples(dst []int16) (int, error) {
	if err := m.stream.Read(); err != nil {
		return 0, err
	}
	return copy(dst, m.buffer), nil
}

func (m *MicSource) Close() error {
	if err := m.stream.Stop(); err != nil {
		_ = m.stream.Close()
		return err
	}
	return m.stream.Close()
}

/* ───────────────────────── audio player (PortAudio) ────────────────────── */

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
	const defaultBuf = 1024
	ap := &AudioPlayer{
		buffer:      make([]int16, defaultBuf),
		bufferSize:  defaultBuf,
		audioQueue:  make([][]int16, 0),
		queueSignal: make(chan struct{}, 1),
		done:        make(chan struct{}),
	}
	go ap.processAudioQueue()
	return ap
}

// ensureStream opens the PortAudio stream (or re-opens with a new rate)
func (ap *AudioPlayer) ensureStream(rate float64) error {
	if ap.stream != nil && ap.sampleRate == rate {
		return nil
	}
	if ap.stream != nil {
		_ = ap.stream.Stop()
		_ = ap.stream.Close()
		ap.stream = nil
	}
	s, err := portaudio.OpenDefaultStream(0, 1, rate, ap.bufferSize, ap.buffer)
	if err != nil {
		return err
	}
	if err = s.Start(); err != nil {
		_ = s.Close()
		return err
	}
	ap.stream = s
	ap.sampleRate = rate
	ap.isPlaying = true
	return nil
}

// EnqueueAudio is called from receiveResponses goroutine
func (ap *AudioPlayer) EnqueueAudio(audioBytes []byte, sampleRate int32) {
	if len(audioBytes) == 0 {
		return
	}
	n := len(audioBytes) / 2
	samples := make([]int16, n)
	for i := 0; i < n; i++ {
		samples[i] = int16(binary.LittleEndian.Uint16(audioBytes[i*2:]))
	}

	ap.mutex.Lock()
	ap.audioQueue = append(ap.audioQueue, samples)
	if err := ap.ensureStream(float64(sampleRate)); err != nil {
		log.Printf("audio stream open: %v", err)
		ap.audioQueue = nil
	}
	ap.mutex.Unlock()

	select {
	case ap.queueSignal <- struct{}{}:
	default:
	}
}

func (ap *AudioPlayer) processAudioQueue() {
	defer close(ap.done)
	for range ap.queueSignal {
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
			chunk := ap.audioQueue[0]
			ap.audioQueue = ap.audioQueue[1:]
			s := ap.stream
			buf := ap.buffer
			size := ap.bufferSize
			ap.mutex.Unlock()

			for i := 0; i < len(chunk); i += size {
				end := i + size
				if end > len(chunk) {
					end = len(chunk)
				}
				copy(buf, chunk[i:end])
				for j := end - i; j < size; j++ {
					buf[j] = 0
				}
				if err := s.Write(); err != nil {
					log.Printf("PortAudio write: %v", err)
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

/* ───────────────────────── VAD helpers ─────────────────────────────────── */

type VADState struct {
	active          bool
	silentFrames    int
	speechFrames    int
	lastSpeechTime  time.Time
	lastEnergy      float64
	noiseFloor      float64
	threshold       float64
	prerollBuffer   [][]byte
	adaptationCount int
}

func newVADState() *VADState {
	return &VADState{
		noiseFloor:    silenceRMS,
		threshold:     silenceRMS * thresholdFactor,
		prerollBuffer: make([][]byte, preRollFrames),
	}
}

func (v *VADState) reset() {
	v.active = false
	v.silentFrames = 0
	v.speechFrames = 0
	v.lastSpeechTime = time.Time{}
}

/* ───────────────────────── main ────────────────────────────────────────── */

func main() {
	flag.Parse()

	fmt.Printf("Dialing %s …\n", serverAddr)
	conn, err := grpc.Dial(serverAddr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("dial: %v", err)
	}
	defer conn.Close()
	client := pb.NewWhisperServiceClient(conn)
	fmt.Println("Connected.")

	if err := portaudio.Initialize(); err != nil {
		log.Fatalf("portaudio init: %v", err)
	}
	defer portaudio.Terminate()

	audioPlayer := NewAudioPlayer()
	defer audioPlayer.Close()

	mic, err := NewMicSource()
	if err != nil {
		log.Fatalf("mic: %v", err)
	}
	defer mic.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stream, err := client.StreamAudio(ctx)
	if err != nil {
		log.Fatalf("StreamAudio: %v", err)
	}

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)

	go receiveResponses(stream, audioPlayer)

	fmt.Println("Streaming microphone …   (Ctrl-C to quit)")
	processAudio(stream, mic, audioPlayer, sig)
	stream.CloseSend()
	time.Sleep(500 * time.Millisecond)
	fmt.Println("Bye.")
}

/* ───────────────────────── VAD + mic → gRPC ───────────────────────────── */

func processAudio(stream pb.WhisperService_StreamAudioClient,
	source *MicSource, player *AudioPlayer, sig chan os.Signal) {

	fmt.Println("=== VAD running ===")

	accum := make([]byte, 0, framesPerBuffer*bytesPerSample*4)
	minSize := framesPerBuffer * bytesPerSample
	timeout := 200 * time.Millisecond
	lastSend := time.Now()
	vad := newVADState()
	prerollIdx := 0
	audioBuf := make([]int16, framesPerBuffer)

	for {
		select {
		case <-sig:
			fmt.Println("\nShutting down …")
			if vad.active && len(accum) > 0 {
				_ = stream.Send(&pb.AudioChunk{Data: accum})
				_ = stream.Send(&pb.AudioChunk{SpeechEnd: true})
			}
			return
		default:
			if player.IsCurrentlyPlaying() {
				_, _ = source.ReadSamples(audioBuf)
				continue
			}

			n, err := source.ReadSamples(audioBuf)
			if err != nil {
				if err == portaudio.InputOverflowed {
					continue
				}
				log.Printf("mic read: %v", err)
				continue
			}
			if n == 0 {
				continue
			}

			sendBuf := make([]byte, framesPerBuffer*bytesPerSample)
			for i, s := range audioBuf {
				binary.LittleEndian.PutUint16(sendBuf[i*2:], uint16(s))
			}

			if vad.prerollBuffer[prerollIdx] == nil {
				vad.prerollBuffer[prerollIdx] = make([]byte, len(sendBuf))
			}
			copy(vad.prerollBuffer[prerollIdx], sendBuf)
			prerollIdx = (prerollIdx + 1) % preRollFrames

			rms := calcRMS(audioBuf[:n])
			vad.lastEnergy = vad.lastEnergy*energySmoothing + rms*(1-energySmoothing)

			if vad.adaptationCount < initialAdapt {
				vad.adaptationCount++
				vad.noiseFloor = (vad.noiseFloor*float64(vad.adaptationCount-1) + rms) /
					float64(vad.adaptationCount)
				vad.threshold = vad.noiseFloor * thresholdFactor
				if vad.adaptationCount == initialAdapt {
					fmt.Printf("\n[VAD calibrated ⇢ noise=%.1f th=%.1f]\n",
						vad.noiseFloor, vad.threshold)
				}
				continue
			}

			fmt.Printf("\rRMS:%6.0f  Noise:%6.0f Th:%6.0f Act:%v  ",
				rms, vad.noiseFloor, vad.threshold, vad.active)

			isSpeech := vad.lastEnergy >= vad.threshold

			if isSpeech {
				vad.speechFrames++
				vad.silentFrames = 0
				vad.lastSpeechTime = time.Now()

				if !vad.active && vad.speechFrames >= 4 {
					vad.active = true
					fmt.Println("\n[Speech detected]")
					_ = stream.Send(&pb.AudioChunk{SpeechStart: true})
					for i := 0; i < preRollFrames; i++ {
						idx := (prerollIdx + i) % preRollFrames
						if vad.prerollBuffer[idx] != nil {
							accum = append(accum, vad.prerollBuffer[idx]...)
						}
					}
				}
				if vad.active {
					accum = append(accum, sendBuf...)
				}
			} else { /* silence */
				vad.speechFrames = 0
				if vad.active {
					holdExpired := time.Since(vad.lastSpeechTime) >
						time.Duration(vadHoldTime)*time.Millisecond
					if holdExpired {
						vad.silentFrames++
					} else {
						accum = append(accum, sendBuf...)
					}
					if vad.silentFrames > maxSilenceFrames {
						dur := time.Since(vad.lastSpeechTime).Milliseconds() + vadHoldTime
						if dur < minSpeechDur {
							fmt.Println("\n[Discarded – too short]")
							accum = accum[:0]
						} else {
							fmt.Println("\n[Speech ended]")
							if len(accum) > 0 {
								_ = stream.Send(&pb.AudioChunk{Data: accum})
								accum = accum[:0]
							}
							_ = stream.Send(&pb.AudioChunk{SpeechEnd: true})
						}
						vad.reset()
					} else if holdExpired {
						/* quick threshold decay for abrupt cut-offs */
						vad.threshold -= 0.5 * (vad.threshold - rms*thresholdFactor)
						accum = append(accum, sendBuf...)
					}
				}
			}

			if vad.active && (len(accum) >= minSize ||
				time.Since(lastSend) >= timeout) {
				_ = stream.Send(&pb.AudioChunk{Data: accum})
				accum = accum[:0]
				lastSend = time.Now()
			}
		}
	}
}

/* ───────────────────────── orchestrator → playback ─────────────────────── */

func receiveResponses(stream pb.WhisperService_StreamAudioClient,
	player *AudioPlayer) {

	var curWhisper string
	var llamaBuf strings.Builder

	for {
		resp, err := stream.Recv()
		if err == io.EOF {
			return
		}
		if err != nil {
			log.Printf("stream recv: %v", err)
			return
		}

		switch resp.GetSource() {
		case pb.StreamAudioResponse_WHISPER:
			text := resp.GetText()
			if text != "" {
				curWhisper = text
				fmt.Print("\r\033[K")
				fmt.Print("TRANSCRIPT: " + curWhisper)
				if len(text) > 0 && text[len(text)-1] != '\n' {
					fmt.Println()
				}
			}

		case pb.StreamAudioResponse_LLAMA:
			txt := resp.GetText()
			if txt != "" {
				llamaBuf.WriteString(txt)
			}
			if resp.GetDone() && llamaBuf.Len() > 0 {
				fmt.Print("\r\033[K")
				fmt.Printf("[LLAMA] %s\n", llamaBuf.String())
				llamaBuf.Reset()
			}

		case pb.StreamAudioResponse_TTS:
			player.EnqueueAudio(resp.GetAudioData(), resp.GetSampleRate())
			if resp.GetIsEndAudio() {
				fmt.Print("\r\033[K")
				fmt.Println("[TTS done]")
			}
		}
	}
}

/* ───────────────────────── helpers ─────────────────────────────────────── */

func calcRMS(buf []int16) float64 {
	if len(buf) == 0 {
		return 0
	}
	var sum float64
	for _, s := range buf {
		sum += float64(s) * float64(s)
	}
	return math.Sqrt(sum / float64(len(buf)))
}