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
				state.processAudioChunk(data, client)
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

// Process audio data and look for patterns that suggest VAD markers
func (s *connState) processAudioChunk(data []byte, client pb.WhisperServiceClient) {
	// We could inspect audio data to detect VAD markers, but the VAD client
	// is already handling speech detection. We'll focus on efficiently handling
	// the data it sends us.

	// Add the new data to our buffer
	s.accumBuffer.Write(data)

	// If we have accumulated enough data, send it to the whisper service
	// This allows for more efficient batching while still preserving
	// the speech segments detected by the VAD client
	if s.accumBuffer.Len() >= 32*1024 { // 32KB threshold for sending
		audioData := s.accumBuffer.Bytes()
		err := s.stream.Send(&pb.AudioChunk{Data: audioData})
		if err != nil {
			log.Printf("Error sending audio chunk: %v", err)
			return
		}

		// Reset buffer after sending
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
