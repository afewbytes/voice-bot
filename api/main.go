package main

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"os"
	"time"

	pb "voice-bot/proto"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// readWAVFile reads a WAV file and returns the PCM data
func readWAVFile(path string) ([]byte, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("could not open audio file: %v", err)
	}
	defer file.Close()

	// Check file size
	fileInfo, err := file.Stat()
	if err != nil {
		return nil, fmt.Errorf("could not get file info: %v", err)
	}
	if fileInfo.Size() < 44 { // Minimum WAV header size
		return nil, fmt.Errorf("file too small to be a valid WAV file")
	}

	// Read RIFF header
	var riffHeader [12]byte
	if _, err := io.ReadFull(file, riffHeader[:]); err != nil {
		return nil, fmt.Errorf("could not read RIFF header: %v", err)
	}

	// Verify "RIFF" and "WAVE" magic numbers
	if string(riffHeader[0:4]) != "RIFF" || string(riffHeader[8:12]) != "WAVE" {
		return nil, fmt.Errorf("not a valid WAV file")
	}

	// Find the data chunk by reading chunks until we find "data"
	for {
		// Read chunk header (8 bytes: 4 for ID, 4 for size)
		var chunkHeader [8]byte
		_, err := io.ReadFull(file, chunkHeader[:])
		if err != nil {
			if err == io.EOF {
				return nil, fmt.Errorf("reached end of file without finding data chunk")
			}
			return nil, fmt.Errorf("could not read chunk header: %v", err)
		}

		chunkID := string(chunkHeader[0:4])
		chunkSize := binary.LittleEndian.Uint32(chunkHeader[4:8])

		log.Printf("Found chunk: %s, size: %d bytes", chunkID, chunkSize)

		// If it's a 'fmt ' chunk, extract audio format info
		if chunkID == "fmt " {
			var fmtChunk struct {
				AudioFormat   uint16
				NumChannels   uint16
				SampleRate    uint32
				ByteRate      uint32
				BlockAlign    uint16
				BitsPerSample uint16
			}
			if err := binary.Read(file, binary.LittleEndian, &fmtChunk); err != nil {
				return nil, fmt.Errorf("could not read fmt chunk: %v", err)
			}

			log.Printf("WAV format: %d channels, %d Hz, %d bits per sample",
				fmtChunk.NumChannels, fmtChunk.SampleRate, fmtChunk.BitsPerSample)

			// Skip any extra format bytes
			if chunkSize > 16 {
				extraBytes := chunkSize - 16
				_, err := file.Seek(int64(extraBytes), io.SeekCurrent)
				if err != nil {
					return nil, fmt.Errorf("could not skip extra fmt bytes: %v", err)
				}
			}
			continue
		}

		// If it's the 'data' chunk, read the audio data
		if chunkID == "data" {
			data := make([]byte, chunkSize)
			_, err := io.ReadFull(file, data)
			if err != nil {
				return nil, fmt.Errorf("could not read audio data: %v", err)
			}
			return data, nil
		}

		// Skip this chunk if it's not 'data'
		_, err = file.Seek(int64(chunkSize), io.SeekCurrent)
		if err != nil {
			return nil, fmt.Errorf("could not skip chunk: %v", err)
		}
	}
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
	testMode := "file"              // Changed default to "file" since we're focusing on WAV reading
	audioFilePath := "/app/jfk.wav" // Change this to your audio file path

	// Log the file path we're trying to open
	log.Printf("Attempting to open audio file: %s", audioFilePath)

	// Check if file exists
	if _, err := os.Stat(audioFilePath); os.IsNotExist(err) {
		log.Fatalf("Audio file does not exist: %s", audioFilePath)
	}

	// Connect to whisper service
	conn, err := grpc.Dial("unix:///app/sockets/whisper.sock", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Failed to connect to whisper service: %v", err)
	}
	defer conn.Close()

	client := pb.NewWhisperServiceClient(conn)

	switch testMode {
	case "file":
		// Read WAV file
		audioData, err := readWAVFile(audioFilePath)
		if err != nil {
			log.Fatalf("Failed to read WAV file: %v", err)
		}

		log.Printf("Successfully read WAV file, got %d bytes of audio data", len(audioData))
		log.Printf("Streaming WAV file to whisper service...")

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
		const duration = 3    // seconds
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
