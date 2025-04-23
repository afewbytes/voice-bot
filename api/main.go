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

			if chunkSize > 16 {
				extraBytes := chunkSize - 16
				_, err := file.Seek(int64(extraBytes), io.SeekCurrent)
				if err != nil {
					return nil, fmt.Errorf("could not skip extra fmt bytes: %v", err)
				}
			}
			continue
		}

		if chunkID == "data" {
			data := make([]byte, chunkSize)
			_, err := io.ReadFull(file, data)
			if err != nil {
				return nil, fmt.Errorf("could not read audio data: %v", err)
			}
			return data, nil
		}

		// Skip unknown chunk
		_, err = file.Seek(int64(chunkSize), io.SeekCurrent)
		if err != nil {
			return nil, fmt.Errorf("could not skip chunk: %v", err)
		}
	}
}

// streamAudioToWhisper streams audio data to the whisper service using server-streaming
func streamAudioToWhisper(client pb.WhisperServiceClient, audioData []byte) (string, error) {
	// Increased timeout to allow more processing time
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Second)
	defer cancel()

	stream, err := client.StreamAudio(ctx)
	if err != nil {
		return "", fmt.Errorf("error creating stream: %v", err)
	}

	// Match server's step size for better processing
	const samplingRate = 16000
	const bytesPerSample = 2
	const chunkSec = 5.0 // Changed from 8.0 to 5.0 to match server step size
	chunkBytes := int(chunkSec * float64(samplingRate) * float64(bytesPerSample))

	log.Printf("Using chunk size of %d bytes (%.2f seconds)", chunkBytes, float64(chunkBytes)/(float64(samplingRate)*float64(bytesPerSample)))

	// Calculate total chunks for progress reporting
	totalChunks := (len(audioData) + chunkBytes - 1) / chunkBytes

	// send audio in chunks
	for i := 0; i < len(audioData); i += chunkBytes {
		end := i + chunkBytes
		if end > len(audioData) {
			end = len(audioData)
		}

		chunk := audioData[i:end]
		// pad final chunk if too small
		if end == len(audioData) && len(chunk) < chunkBytes {
			log.Printf("Final chunk is %d bytes, padding to %d bytes", len(chunk), chunkBytes)
			padded := make([]byte, chunkBytes)
			copy(padded, chunk)
			chunk = padded
			log.Printf("Padded final chunk from %d to %d bytes", len(audioData)-i, chunkBytes)
		}

		currentChunk := (i / chunkBytes) + 1
		log.Printf("Sending chunk %d of %d, size: %d bytes, time range: %.1f-%.1f seconds",
			currentChunk, totalChunks, len(chunk),
			float64(i)/(float64(samplingRate)*float64(bytesPerSample)),
			float64(end)/(float64(samplingRate)*float64(bytesPerSample)))

		if err := stream.Send(&pb.AudioChunk{Data: chunk}); err != nil {
			return "", fmt.Errorf("error sending audio chunk %d/%d: %v", currentChunk, totalChunks, err)
		}

		// throttle to avoid overwhelming server
		time.Sleep(30 * time.Millisecond)
	}

	// signal done
	log.Printf("Finished sending all audio chunks, closing send stream")
	if err := stream.CloseSend(); err != nil {
		return "", fmt.Errorf("error closing send: %v", err)
	}

	// receive partial transcripts
	var fullText string
	segmentCount := 0
	for {
		resp, err := stream.Recv()
		if err == io.EOF {
			log.Printf("Reached end of transcript stream with %d segments", segmentCount)
			break
		}
		if err != nil {
			return "", fmt.Errorf("error receiving transcription: %v", err)
		}
		segmentCount++
		part := resp.GetText()
		log.Printf("Partial transcript #%d received at %s: %q",
			segmentCount, time.Now().Format(time.RFC3339), part)
		fullText += part + " "
	}

	return fullText, nil
}

func main() {
	audioFilePath := "/app/jfk.wav"
	if _, err := os.Stat(audioFilePath); os.IsNotExist(err) {
		log.Fatalf("Audio file does not exist: %s", audioFilePath)
	}

	// Debug file size
	fileInfo, err := os.Stat(audioFilePath)
	if err == nil {
		log.Printf("Audio file size: %d bytes", fileInfo.Size())
	}

	conn, err := grpc.Dial("unix:///app/sockets/whisper.sock", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Failed to connect to whisper service: %v", err)
	}
	defer conn.Close()

	client := pb.NewWhisperServiceClient(conn)

	audioData, err := readWAVFile(audioFilePath)
	if err != nil {
		log.Fatalf("Failed to read WAV file: %v", err)
	}
	log.Printf("Read WAV file, %d bytes", len(audioData))

	transcription, err := streamAudioToWhisper(client, audioData)
	if err != nil {
		log.Fatalf("Streaming failed: %v", err)
	}

	fmt.Printf("Transcription: %s\n", transcription)
}
