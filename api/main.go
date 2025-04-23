package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"os"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	pb "voice-bot/proto"
)

// readAudioFile reads an audio file in raw PCM format and returns the bytes
func readAudioFile(path string) ([]byte, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("could not open audio file: %v", err)
	}
	defer file.Close()

	// Read entire file
	data, err := io.ReadAll(file)
	if err != nil {
		return nil, fmt.Errorf("could not read audio file: %v", err)
	}

	return data, nil
}

// streamAudioToWhisper streams audio data to the whisper service
func streamAudioToWhisper(client pb.WhisperServiceClient, audioData []byte, chunkSize int) (string, error) {
	// Create a new stream
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	stream, err := client.StreamAudio(ctx)
	if err != nil {
		return "", fmt.Errorf("error creating stream: %v", err)
	}

	// Send audio in chunks
	for i := 0; i < len(audioData); i += chunkSize {
		end := i + chunkSize
		if end > len(audioData) {
			end = len(audioData)
		}

		chunk := audioData[i:end]
		if err := stream.Send(&pb.AudioChunk{Data: chunk}); err != nil {
			if err == io.EOF {
				break
			}
			return "", fmt.Errorf("error sending audio chunk: %v", err)
		}

		// Optional: add a slight delay to simulate real-time streaming
		time.Sleep(10 * time.Millisecond)
	}

	// Get the response
	reply, err := stream.CloseAndRecv()
	if err != nil {
		return "", fmt.Errorf("error receiving transcription: %v", err)
	}

	return reply.Text, nil
}

func main() {
	// Test mode can be one of:
	// - "simulate" - Send simulated audio data
	// - "file" - Stream audio from a file (if provided)
	testMode := "simulate"
	audioFilePath := "/app/audio/test.pcm" // Change this to your audio file path

	// Connect to whisper service
	conn, err := grpc.Dial("unix:///app/sockets/whisper.sock", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Failed to connect to whisper service: %v", err)
	}
	defer conn.Close()

	client := pb.NewWhisperServiceClient(conn)

	switch testMode {
	case "file":
		// Read audio file
		audioData, err := readAudioFile(audioFilePath)
		if err != nil {
			log.Fatalf("Failed to read audio file: %v", err)
		}

		log.Printf("Streaming audio file (%d bytes) to whisper service...", len(audioData))
		
		// Stream to whisper
		transcription, err := streamAudioToWhisper(client, audioData, 4096) // 4KB chunks
		if err != nil {
			log.Fatalf("Failed to stream audio: %v", err)
		}

		fmt.Printf("Transcription: %s\n", transcription)

	case "simulate":
		// Simulate streaming with dummy PCM data
		log.Println("Simulating audio stream with dummy data...")
		
		// Create a context with timeout
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		
		// Create a stream
		stream, err := client.StreamAudio(ctx)
		if err != nil {
			log.Fatalf("Error creating stream: %v", err)
		}
		
		// Generate some dummy PCM data (sine wave)
		// Real implementation would use actual audio data
		const sampleRate = 16000
		const duration = 3 // seconds
		const frequency = 440 // A4 note
		
		// Generate ~3 seconds of 16-bit PCM audio (sine wave)
		numSamples := sampleRate * duration
		audioData := make([]byte, numSamples*2) // 2 bytes per sample (16-bit)
		
		for i := 0; i < numSamples; i++ {
			// Simple sine wave
			amplitude := int16(10000 * float64(i%100) / 100.0)
			
			// Convert to bytes and store in little-endian format
			audioData[i*2] = byte(amplitude & 0xFF)
			audioData[i*2+1] = byte((amplitude >> 8) & 0xFF)
		}
		
		// Send in chunks of 1600 samples (~100ms at 16kHz)
		const chunkSize = 3200 // 1600 samples * 2 bytes per sample
		
		for i := 0; i < len(audioData); i += chunkSize {
			end := i + chunkSize
			if end > len(audioData) {
				end = len(audioData)
			}
			
			chunk := audioData[i:end]
			if err := stream.Send(&pb.AudioChunk{Data: chunk}); err != nil {
				log.Fatalf("Error sending audio chunk: %v", err)
			}
			
			// Simulate real-time streaming rate
			time.Sleep(100 * time.Millisecond)
		}
		
		// Get transcription result
		reply, err := stream.CloseAndRecv()
		if err != nil {
			log.Fatalf("Error receiving transcription: %v", err)
		}
		
		fmt.Printf("Transcription: %s\n", reply.Text)
	}
}