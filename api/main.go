package main

import (
	"bytes"
	"context"
	"io"
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"
	"time"

	pb "voice-bot/proto"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const (
	// Server configuration
	listenAddr      = ":8090"
	whisperSockPath = "unix:///app/sockets/whisper.sock"

	// Audio parameters
	sampleRate     = 16000
	bytesPerSample = 2

	// Buffer sizes
	readBufSize    = 4096
	initialBufSize = 64 * 1024 // 64KB initial buffer for audio accumulation

	// Timeout values
	readTimeout = 10 * time.Second
	idleTimeout = 30 * time.Second // Connection idle timeout
)

// Protocol markers for speech boundaries (must match client markers)
var (
	speechStartMarker = []byte{0xFF, 0x00, 0x00, 0x01} // Unique marker for speech start
	speechEndMarker   = []byte{0xFF, 0x00, 0x00, 0x02} // Unique marker for speech end
)

// Connection state
type connState struct {
	accumBuffer  bytes.Buffer                        // Buffer for accumulating audio data
	lastActivity time.Time                           // Last time we received data
	isActive     bool                                // Whether we're currently in an active speech segment
	conn         net.Conn                            // TCP connection
	stream       pb.WhisperService_StreamAudioClient // gRPC stream
}

func main() {
	// Connect to whisper service
	conn, err := grpc.Dial(whisperSockPath, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Failed to connect to whisper service: %v", err)
	}
	defer conn.Close()
	client := pb.NewWhisperServiceClient(conn)

	log.Println("=== VAD-Aware Whisper Audio Server ===")
	log.Println("Connecting to whisper service:", whisperSockPath)

	// Start audio receiver
	if err := audioReceiver(client); err != nil {
		log.Fatalf("Error in audio receiver: %v", err)
	}
}

func audioReceiver(client pb.WhisperServiceClient) error {
	listener, err := net.Listen("tcp", listenAddr)
	if err != nil {
		return err
	}
	defer listener.Close()

	log.Println("Listening on", listenAddr, "for VAD-enabled audio clients")

	// Handle Ctrl+C
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigChan
		log.Println("Shutting down listener")
		listener.Close()
	}()

	for {
		conn, err := listener.Accept()
		if err != nil {
			select {
			case <-sigChan:
				return nil
			default:
				log.Printf("Accept error: %v", err)
				continue
			}
		}
		log.Printf("Client %s connected", conn.RemoteAddr())
		go handleAudioConnection(conn, client)
	}
}

func handleAudioConnection(conn net.Conn, client pb.WhisperServiceClient) {
	defer func() {
		log.Printf("Client %s disconnected", conn.RemoteAddr())
		conn.Close()
	}()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Create bidirectional stream
	stream, err := client.StreamAudio(ctx)
	if err != nil {
		log.Printf("StreamAudio error: %v", err)
		return
	}
	defer stream.CloseSend()

	// Set up state
	state := &connState{
		accumBuffer:  bytes.Buffer{},
		lastActivity: time.Now(),
		isActive:     false,
		conn:         conn,
		stream:       stream,
	}
	state.accumBuffer.Grow(initialBufSize)

	// Start a goroutine to read responses from whisper service
	go receiveTranscriptions(state)

	// Read loop for audio data
	readBuf := make([]byte, readBufSize)
	idleTimer := time.NewTimer(idleTimeout)
	defer idleTimer.Stop()

	// Channel to signal when data is received
	dataCh := make(chan struct{})

	for {
		// Set up a goroutine to handle the read with timeout
		go func() {
			conn.SetReadDeadline(time.Now().Add(readTimeout))
			n, err := conn.Read(readBuf)
			if err != nil {
				close(dataCh)
				return
			}
			if n > 0 {
				// Check if we detect VAD markers in the stream
				data := readBuf[:n]
				state.processAudioChunk(data)
				dataCh <- struct{}{}
			}
		}()

		select {
		case _, ok := <-dataCh:
			if !ok {
				// Channel closed, exit
				return
			}
			// Reset idle timer when we get data
			if !idleTimer.Stop() {
				select {
				case <-idleTimer.C:
				default:
				}
			}
			idleTimer.Reset(idleTimeout)
			state.lastActivity = time.Now()

		case <-idleTimer.C:
			log.Printf("Client %s idle timeout", conn.RemoteAddr())
			return
		}
	}
}

// Process audio data and check for speech boundary markers
func (s *connState) processAudioChunk(data []byte) {
	// Check for speech start marker
	if bytes.Equal(data, speechStartMarker) {
		log.Println("Speech start marker detected")

		// Flush any existing audio
		if s.accumBuffer.Len() > 0 {
			s.sendAccumulatedAudio(false, false)
		}

		// Send speech start marker to whisper service
		err := s.stream.Send(&pb.AudioChunk{
			Data:        nil,
			SpeechStart: true,
		})
		if err != nil {
			log.Printf("Error sending speech start marker: %v", err)
		}

		s.isActive = true
		return
	}

	// Check for speech end marker
	if bytes.Equal(data, speechEndMarker) {
		log.Println("Speech end marker detected")

		// Send any accumulated audio with speech end flag
		s.sendAccumulatedAudio(false, true)

		s.isActive = false
		return
	}

	// Regular audio data - add to buffer
	s.accumBuffer.Write(data)

	// Send accumulated audio if buffer gets large enough
	if s.accumBuffer.Len() >= 32*1024 {
		s.sendAccumulatedAudio(false, false)
	}
}

// Helper to send accumulated audio with appropriate flags
func (s *connState) sendAccumulatedAudio(speechStart bool, speechEnd bool) {
	if s.accumBuffer.Len() == 0 {
		return
	}

	// Send the audio with appropriate flags
	err := s.stream.Send(&pb.AudioChunk{
		Data:        s.accumBuffer.Bytes(),
		SpeechStart: speechStart,
		SpeechEnd:   speechEnd,
	})

	if err != nil {
		log.Printf("Error sending audio chunk: %v", err)
	} else {
		// Reset buffer after successful send
		s.accumBuffer.Reset()
	}
}

// Receive and forward transcriptions from the whisper service
func receiveTranscriptions(state *connState) {
	for {
		resp, err := state.stream.Recv()
		if err == io.EOF {
			return
		}
		if err != nil {
			log.Printf("Stream receive error: %v", err)
			return
		}

		text := resp.GetText()
		if text == "" {
			continue
		}

		// Forward the transcription to the client
		_, err = state.conn.Write([]byte(text))
		if err != nil {
			log.Printf("Write error: %v", err)
			return
		}
	}
}
