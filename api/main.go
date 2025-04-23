package main

import (
	"context"
	"fmt"
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
	sampleRate     = 16000
	bytesPerSample = 2
	chunkDuration  = 5 * time.Second
)

// audioReceiver listens for audio data from a client and forwards it to the Whisper service
func audioReceiver(client pb.WhisperServiceClient) error {
	// Create a TCP listener
	listener, err := net.Listen("tcp", ":8090")
	if err != nil {
		return fmt.Errorf("failed to start TCP listener: %v", err)
	}
	defer listener.Close()

	log.Println("Audio receiver started on port 8090")
	log.Println("Waiting for audio connections...")

	// Set up signal channel to detect Ctrl+C
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	// Handle shutdown gracefully
	go func() {
		<-sigChan
		log.Println("Interrupt received, shutting down...")
		listener.Close()
	}()

	for {
		// Accept incoming connections
		conn, err := listener.Accept()
		if err != nil {
			// Check if we're shutting down
			select {
			case <-sigChan:
				return nil
			default:
				log.Printf("Error accepting connection: %v", err)
				continue
			}
		}

		// Handle each connection in a separate goroutine
		go handleAudioConnection(conn, client)
	}
}

// handleAudioConnection processes audio data from a single client connection
func handleAudioConnection(conn net.Conn, client pb.WhisperServiceClient) {
	defer conn.Close()
	log.Printf("New audio connection from %s", conn.RemoteAddr())

	// Create context for this connection
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Create a new whisper stream
	whisperStream, err := client.StreamAudio(ctx)
	if err != nil {
		log.Printf("Error creating whisper stream: %v", err)
		return
	}

	// Start a goroutine to receive transcriptions and send them back to the client
	go func() {
		for {
			resp, err := whisperStream.Recv()
			if err == io.EOF {
				return
			}
			if err != nil {
				if ctx.Err() == context.Canceled {
					return // Context was canceled, just exit
				}
				log.Printf("Error receiving transcription: %v", err)
				return
			}

			// Log the transcription
			transcription := resp.GetText()
			log.Printf("Transcription: %s", transcription)

			// Send the transcription back to the client
			_, err = conn.Write([]byte(transcription + "\n"))
			if err != nil {
				log.Printf("Error sending transcription to client: %v", err)
				return
			}
		}
	}()

	// Calculate chunk size
	chunkBytes := int(chunkDuration.Seconds() * float64(sampleRate) * float64(bytesPerSample))
	buffer := make([]byte, chunkBytes)

	// Read audio data from the connection and send it to whisper
	for {
		// Set read deadline to detect disconnection
		conn.SetReadDeadline(time.Now().Add(10 * time.Second))

		// Read data
		bytesRead, err := conn.Read(buffer)
		if err != nil {
			if err == io.EOF {
				log.Printf("Client disconnected")
			} else if netErr, ok := err.(net.Error); ok && netErr.Timeout() {
				log.Printf("Client timed out")
			} else {
				log.Printf("Error reading from connection: %v", err)
			}
			break
		}

		// Reset deadline after successful read
		conn.SetReadDeadline(time.Time{})

		// Send the audio chunk to whisper
		if err := whisperStream.Send(&pb.AudioChunk{Data: buffer[:bytesRead]}); err != nil {
			log.Printf("Error sending audio chunk: %v", err)
			break
		}
	}

	// Close the send stream when done
	if err := whisperStream.CloseSend(); err != nil {
		log.Printf("Error closing send stream: %v", err)
	}

	log.Printf("Connection from %s closed", conn.RemoteAddr())
}

func main() {
	// Connect to whisper service
	conn, err := grpc.Dial("unix:///app/sockets/whisper.sock", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Failed to connect to whisper service: %v", err)
	}
	defer conn.Close()

	client := pb.NewWhisperServiceClient(conn)

	fmt.Println("=== Whisper Audio Receiver ===")
	fmt.Println("Ready to receive audio from clients")

	// Start the audio receiver
	if err := audioReceiver(client); err != nil {
		log.Fatalf("Error in audio receiver: %v", err)
	}
}
