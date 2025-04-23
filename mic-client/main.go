package main

import (
	"bufio"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gordonklaus/portaudio"
)

const (
	serverAddr      = "localhost:8090" // Change to your Docker host IP if needed
	sampleRate      = 16000
	framesPerBuffer = 1024
	channels        = 1
)

func main() {
	// Initialize PortAudio
	if err := portaudio.Initialize(); err != nil {
		log.Fatalf("Failed to initialize portaudio: %v", err)
	}
	defer portaudio.Terminate()

	// Connect to server
	fmt.Printf("Connecting to server at %s...\n", serverAddr)
	conn, err := net.Dial("tcp", serverAddr)
	if err != nil {
		log.Fatalf("Failed to connect to server: %v", err)
	}
	defer conn.Close()
	fmt.Println("Connected to server!")

	// Set up audio stream
	audioBuffer := make([]int16, framesPerBuffer*channels)
	stream, err := portaudio.OpenDefaultStream(channels, 0, float64(sampleRate), framesPerBuffer, audioBuffer)
	if err != nil {
		log.Fatalf("Failed to open mic stream: %v", err)
	}
	defer stream.Close()

	if err := stream.Start(); err != nil {
		log.Fatalf("Failed to start mic stream: %v", err)
	}
	defer stream.Stop()

	// Set up signal handler for graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	// Start a goroutine to receive transcriptions
	go receiveTranscriptions(conn)

	// Create a channel to signal when we're done
	done := make(chan struct{})

	// Handle interrupt signal
	go func() {
		<-sigChan
		fmt.Println("\nShutting down...")
		close(done)
	}()

	fmt.Println("=== Microphone Streaming Client ===")
	fmt.Println("Start speaking (Press Ctrl+C to exit)...")

	// Main loop to capture and send audio
	sendBuffer := make([]byte, framesPerBuffer*channels*2)
loop:
	for {
		select {
		case <-done:
			break loop
		default:
			// Read from the microphone
			if err := stream.Read(); err != nil {
				log.Printf("Error reading from mic: %v", err)
				continue
			}

			// Convert int16 to bytes
			for i, sample := range audioBuffer {
				sendBuffer[i*2] = byte(sample & 0xFF)
				sendBuffer[i*2+1] = byte((sample >> 8) & 0xFF)
			}

			// Send the audio data to the server
			if _, err := conn.Write(sendBuffer); err != nil {
				log.Printf("Error sending audio data: %v", err)
				break loop
			}
		}
	}

	fmt.Println("Disconnected from server")
}

// receiveTranscriptions reads and prints transcriptions from the server
func receiveTranscriptions(conn net.Conn) {
	scanner := bufio.NewScanner(conn)
	for scanner.Scan() {
		text := scanner.Text()
		if text != "" {
			fmt.Printf("Transcription: %s\n", text)
		}
	}

	if err := scanner.Err(); err != nil {
		// Don't print error if we're shutting down
		select {
		case <-time.After(100 * time.Millisecond):
			log.Printf("Error receiving transcription: %v", err)
		default:
			// We're shutting down, ignore the error
		}
	}
}
