package main

import (
	"context"
	"io"
	"log"
	"net"
	"os"
	"os/signal"
	"strings"
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
	llamaSockPath   = "unix:///app/llama-sockets/llama.sock"
)

// WhisperServiceServer implements the gRPC server for WhisperService
// It orchestrates between WhisperService (transcription) and LlamaService (generation)
type WhisperServiceServer struct {
	pb.UnimplementedWhisperServiceServer
	whisperClient pb.WhisperServiceClient
	llamaClient   pb.LlamaServiceClient
}

// StreamAudio handles a bi-directional stream: audio chunks in, transcript + generation out
func (s *WhisperServiceServer) StreamAudio(stream pb.WhisperService_StreamAudioServer) error {
	log.Println("New gRPC client connected")

	ctx := stream.Context()

	// 1) Open a stream to Whisper
	log.Println("Opening stream to Whisper service...")
	whisperStream, err := s.whisperClient.StreamAudio(ctx)
	if err != nil {
		log.Printf("ERROR: Failed to open Whisper stream: %v", err)
		return err
	}
	defer whisperStream.CloseSend()
	log.Println("Successfully opened stream to Whisper service")

	// 2) Relay Whisper → client & call LLaMA for each segment
	log.Println("Starting Whisper response relay...")
	errCh := make(chan error, 1)
	go func() {
		transcriptionSegments := 0
		for {
			resp, err := whisperStream.Recv()
			if err == io.EOF {
				log.Printf("Whisper stream closed (EOF) after %d segments", transcriptionSegments)
				errCh <- nil
				return
			}
			if err != nil {
				log.Printf("ERROR: Whisper stream receive error: %v", err)
				errCh <- err
				return
			}

			transcriptionText := resp.GetText()
			transcriptionSegments++
			log.Printf("Received Whisper segment #%d: %q", transcriptionSegments, transcriptionText)

			// Skip empty transcriptions
			if strings.TrimSpace(transcriptionText) == "" {
				log.Println("Skipping empty transcription segment")
				continue
			}

			// Send partial transcription to client
			if err := stream.Send(&pb.StreamAudioResponse{
				Text:   transcriptionText,
				Source: pb.StreamAudioResponse_WHISPER,
			}); err != nil {
				log.Printf("ERROR: Failed to send transcription to client: %v", err)
				errCh <- err
				return
			}

			// Call LLaMA with this segment
			log.Printf("Calling LLaMA with segment: %q", transcriptionText)
			llamaReq := &pb.GenerateRequest{
				Prompt:      transcriptionText,
				MaxTokens:   128, // Reduced max tokens for each segment
				Temperature: 0.7,
				TopP:        0.9,
			}
			
			// Process response from LLaMA
			llamaStream, err := s.llamaClient.Generate(ctx, llamaReq)
			if err != nil {
				log.Printf("WARNING: Failed to call LLaMA for segment: %v", err)
				continue // Continue with next segment even if this one fails
			}
			
			// Stream LLaMA tokens to client
			tokenCount := 0
			for {
				genResp, err := llamaStream.Recv()
				if err == io.EOF {
					log.Printf("LLaMA stream for segment #%d ended (EOF) after %d tokens", 
						transcriptionSegments, tokenCount)
					break
				}
				if err != nil {
					log.Printf("WARNING: Error receiving from LLaMA for segment #%d: %v", 
						transcriptionSegments, err)
					break
				}

				tokenText := genResp.GetText()
				isDone := genResp.GetDone()
				tokenCount++
				
				if tokenCount%20 == 0 {
					log.Printf("Received %d LLaMA tokens for segment #%d", 
						tokenCount, transcriptionSegments)
				}

				// Send token to client
				if err := stream.Send(&pb.StreamAudioResponse{
					Text:   tokenText,
					Done:   isDone,
					Source: pb.StreamAudioResponse_LLAMA,
				}); err != nil {
					log.Printf("ERROR: Failed to send LLaMA token to client: %v", err)
					errCh <- err
					return
				}
				
				if isDone {
					log.Printf("LLaMA generation for segment #%d complete", transcriptionSegments)
					break
				}
			}
		}
	}()

	// 3) Forward incoming audio chunks from client → Whisper
	log.Println("Forwarding audio chunks to Whisper...")
	audioChunks := 0
	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			log.Printf("Client audio stream ended (EOF) after %d chunks", audioChunks)
			break
		}
		if err != nil {
			log.Printf("ERROR: Failed to receive audio chunk from client: %v", err)
			return err
		}
		audioChunks++
		if audioChunks%50 == 0 {
			log.Printf("Processed %d audio chunks so far", audioChunks)
		}
		
		if err := whisperStream.Send(chunk); err != nil {
			log.Printf("ERROR: Failed to send audio chunk to Whisper: %v", err)
			return err
		}
	}
	log.Printf("Completed forwarding %d audio chunks to Whisper", audioChunks)
	
	// Close the send direction of the Whisper stream
	log.Println("Closing send direction of Whisper stream...")
	if err := whisperStream.CloseSend(); err != nil {
		log.Printf("WARNING: Error closing send direction of Whisper stream: %v", err)
	}

	// Wait for the whisper goroutine to finish
	log.Println("Waiting for Whisper processing to complete...")
	if err := <-errCh; err != nil {
		log.Printf("ERROR: Whisper processing failed: %v", err)
		return err
	}
	log.Println("Whisper processing completed successfully")

	return nil
}

func main() {
	log.Println("=== gRPC Audio Orchestrator Starting ===")
	
	// Dial Whisper
	log.Printf("Connecting to Whisper service at %s...", whisperSockPath)
	wConn, err := grpc.Dial(whisperSockPath, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Failed to connect to Whisper: %v", err)
	}
	defer wConn.Close()
	whisperClient := pb.NewWhisperServiceClient(wConn)
	log.Println("Successfully connected to Whisper service")

	// Dial Llama
	log.Printf("Connecting to LLaMA service at %s...", llamaSockPath)
	lConn, err := grpc.Dial(llamaSockPath, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Failed to connect to LLaMA: %v", err)
	}
	defer lConn.Close()
	llamaClient := pb.NewLlamaServiceClient(lConn)
	log.Println("Successfully connected to LLaMA service")
	
	// Test LLaMA connection
	log.Println("Testing LLaMA service with a simple request...")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	testReq := &pb.GenerateRequest{
		Prompt:      "Hello",
		MaxTokens:   1,
		Temperature: 0.0,
	}
	testStream, testErr := llamaClient.Generate(ctx, testReq)
	if testErr != nil {
		log.Printf("WARNING: LLaMA test request failed: %v", testErr)
	} else {
		log.Println("LLaMA test request succeeded, checking for response...")
		resp, err := testStream.Recv()
		if err != nil && err != io.EOF {
			log.Printf("WARNING: Error receiving LLaMA test response: %v", err)
		} else if err == io.EOF {
			log.Println("WARNING: LLaMA test gave EOF without data")
		} else {
			log.Printf("LLaMA test response: %q", resp.GetText())
		}
		// Drain the rest
		for {
			_, err := testStream.Recv()
			if err != nil {
				break
			}
		}
	}

	log.Println("=== gRPC Audio Orchestrator ===")
	log.Println("Whisper at", whisperSockPath, "  LLaMA at", llamaSockPath)

	grpcServer := grpc.NewServer()
	pb.RegisterWhisperServiceServer(grpcServer, &WhisperServiceServer{
		whisperClient: whisperClient,
		llamaClient:   llamaClient,
	})

	lis, err := net.Listen("tcp", listenAddr)
	if err != nil {
		log.Fatalf("Failed to listen: %v", err)
	}
	log.Println("Listening on", listenAddr)

	// Graceful shutdown
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		log.Println("Shutting down")
		grpcServer.GracefulStop()
	}()

	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("Serve error: %v", err)
	}
}