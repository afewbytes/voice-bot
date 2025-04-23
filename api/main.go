package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"math"
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
	// Audio settings
	sampleRate     = 16000
	bytesPerSample = 2
	chunkDuration  = 1 * time.Second // 1-second chunks to match C++ service

	// Silence detection settings
	silenceThreshold = 300  // Amplitude threshold for detecting silence
	silenceRatio     = 0.95 // If more than 95% of samples are below threshold, consider it silence

	// Silence handling settings
	maxConsecutiveSilence = 3  // Send first N silence chunks to maintain context
	silenceSendRate       = 15 // After that, only send 1 out of N silence chunks
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

// detectSilence checks if audio data contains mostly silence
func detectSilence(data []byte) bool {
	if len(data) < bytesPerSample {
		return true
	}

	// Count silent samples
	totalSamples := len(data) / bytesPerSample
	silentSamples := 0

	for i := 0; i < len(data); i += bytesPerSample {
		// Convert bytes to int16 sample
		sample := int16(data[i]) | (int16(data[i+1]) << 8)

		// Check if sample amplitude is below threshold
		if math.Abs(float64(sample)) < silenceThreshold {
			silentSamples++
		}
	}

	// Calculate ratio of silent samples
	ratio := float64(silentSamples) / float64(totalSamples)

	// If debugging silence detection, uncomment the line below
	// log.Printf("Silence ratio: %.2f (silent samples: %d/%d)", ratio, silentSamples, totalSamples)

	return ratio > silenceRatio
}

// calculateRMS calculates the RMS (Root Mean Square) energy of audio data
func calculateRMS(data []byte) float64 {
	if len(data) < bytesPerSample {
		return 0
	}

	totalSamples := len(data) / bytesPerSample
	sumSquares := 0.0

	for i := 0; i < len(data); i += bytesPerSample {
		// Convert bytes to int16 sample
		sample := int16(data[i]) | (int16(data[i+1]) << 8)

		// Add square of sample value
		sumSquares += float64(sample) * float64(sample)
	}

	// Calculate RMS
	meanSquare := sumSquares / float64(totalSamples)
	return math.Sqrt(meanSquare)
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
	lastTranscription := ""

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

			// Get the transcription
			transcription := resp.GetText()

			// Skip [BLANK_AUDIO] entries
			if transcription == "[BLANK_AUDIO]" {
				continue
			}

			// Skip empty transcriptions
			if transcription == "" {
				continue
			}

			// Skip duplicate consecutive transcriptions
			if transcription == lastTranscription {
				continue
			}

			// Update last transcription and log
			lastTranscription = transcription
			log.Printf("Transcription: %s", transcription)

			// Send the transcription back to the client
			_, err = conn.Write([]byte(transcription + "\n"))
			if err != nil {
				log.Printf("Error sending transcription to client: %v", err)
				return
			}
		}
	}()

	// Calculate ideal buffer size for 1-second chunks
	optimalChunkBytes := int(chunkDuration.Seconds() * float64(sampleRate) * float64(bytesPerSample))
	log.Printf("Target chunk size: %d bytes (%.2f seconds of audio)",
		optimalChunkBytes, float64(optimalChunkBytes)/(float64(sampleRate)*float64(bytesPerSample)))

	// Buffer for network reads
	readBuffer := make([]byte, 4096) // 4KB read buffer

	// Accumulation buffer with extra capacity
	accumulatedData := make([]byte, 0, optimalChunkBytes*2)

	// Minimum chunk size (80% of optimal) to avoid sending tiny chunks
	minChunkSize := optimalChunkBytes * 4 / 5 // 80% of optimal size

	// Maximum time to wait before sending data
	maxAccumulationTime := 500 * time.Millisecond

	// Track when we started accumulating the current chunk
	accumulationStart := time.Now()

	// Track consecutive silence chunks to prevent sending too many
	silenceCounter := 0

	// Read audio data from the connection and send it to whisper
	for {
		// Set read deadline to detect disconnection
		conn.SetReadDeadline(time.Now().Add(10 * time.Second))

		// Read data
		bytesRead, err := conn.Read(readBuffer)
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

		// Add new data to accumulated buffer
		accumulatedData = append(accumulatedData, readBuffer[:bytesRead]...)

		// Decide whether to send the accumulated data
		// Send if:
		// 1. We have at least the minimum chunk size, OR
		// 2. We've been accumulating for more than maxAccumulationTime
		timeAccumulating := time.Since(accumulationStart)
		readyToSend := len(accumulatedData) >= minChunkSize || timeAccumulating >= maxAccumulationTime

		if readyToSend && len(accumulatedData) > 0 {
			// Check for silence
			isSilence := detectSilence(accumulatedData)
			rmsValue := calculateRMS(accumulatedData)

			// Calculate audio duration of this chunk
			audioDuration := float64(len(accumulatedData)) / float64(sampleRate*bytesPerSample)

			// Decision logic for sending silent chunks
			sendChunk := true

			if isSilence {
				silenceCounter++

				// Only send occasional silence chunks after reaching threshold
				if silenceCounter > maxConsecutiveSilence {
					// Skip nearly all silence chunks
					sendChunk = (silenceCounter%silenceSendRate == 0)
				}

				if sendChunk {
					log.Printf("Sending silent chunk (%d consecutive) - RMS: %.1f",
						silenceCounter, rmsValue)
				} else {
					log.Printf("Skipping silent chunk (%d consecutive) - RMS: %.1f",
						silenceCounter, rmsValue)
				}
			} else {
				// Reset silence counter when we get actual audio
				silenceCounter = 0
			}

			if sendChunk {
				// Send data to whisper
				if err := whisperStream.Send(&pb.AudioChunk{Data: accumulatedData}); err != nil {
					log.Printf("Error sending audio chunk: %v", err)
					break
				}

				// Log chunk details
				log.Printf("Sent audio chunk: %d bytes (%.2f seconds) after accumulating for %.1fms - %s",
					len(accumulatedData), audioDuration,
					float64(timeAccumulating)/float64(time.Millisecond),
					silenceStatusString(isSilence, rmsValue))
			} else {
				// Log that we're skipping this chunk
				log.Printf("Skipped sending chunk: %d bytes (%.2f seconds) - %s",
					len(accumulatedData), audioDuration,
					silenceStatusString(isSilence, rmsValue))
			}

			// Reset accumulated data and timer
			accumulatedData = accumulatedData[:0]
			accumulationStart = time.Now()
		}
	}

	// Send any remaining data before closing
	if len(accumulatedData) > 0 {
		audioDuration := float64(len(accumulatedData)) / float64(sampleRate*bytesPerSample)
		isSilence := detectSilence(accumulatedData)
		rmsValue := calculateRMS(accumulatedData)

		// Only send final chunk if it's not silence
		if !isSilence {
			if err := whisperStream.Send(&pb.AudioChunk{Data: accumulatedData}); err != nil {
				log.Printf("Error sending final audio chunk: %v", err)
			} else {
				log.Printf("Sent final chunk: %d bytes (%.2f seconds) - %s",
					len(accumulatedData), audioDuration, silenceStatusString(isSilence, rmsValue))
			}
		}
	}

	// Close the send stream when done
	if err := whisperStream.CloseSend(); err != nil {
		log.Printf("Error closing send stream: %v", err)
	}

	log.Printf("Connection from %s closed", conn.RemoteAddr())
}

// Helper function to create a status string for logging
func silenceStatusString(isSilence bool, rms float64) string {
	if isSilence {
		return fmt.Sprintf("SILENCE (RMS: %.1f)", rms)
	}
	return fmt.Sprintf("AUDIO (RMS: %.1f)", rms)
}

func main() {
	// Connect to whisper service
	conn, err := grpc.Dial("unix:///app/sockets/whisper.sock", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Failed to connect to whisper service: %v", err)
	}
	defer conn.Close()

	client := pb.NewWhisperServiceClient(conn)

	fmt.Println("=== Whisper Audio Receiver with Silence Detection ===")
	fmt.Println("Ready to receive audio from clients")

	// Start the audio receiver
	if err := audioReceiver(client); err != nil {
		log.Fatalf("Error in audio receiver: %v", err)
	}
}
