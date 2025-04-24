package main

import (
	"io"
	"log"
	"net"
	"os"
	"os/signal"
	"strings"
	"syscall"

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
	whisperStream, err := s.whisperClient.StreamAudio(ctx)
	if err != nil {
		return err
	}
	defer whisperStream.CloseSend()

	// Buffer for the full transcription
	var fullTranscription strings.Builder

	// 2) Relay Whisper → client & accumulate text
	errCh := make(chan error, 1)
	go func() {
		for {
			resp, err := whisperStream.Recv()
			if err == io.EOF {
				errCh <- nil
				return
			}
			if err != nil {
				errCh <- err
				return
			}

			// Send partial transcription
			if err := stream.Send(&pb.StreamAudioResponse{
				Text:   resp.GetText(),
				Source: pb.StreamAudioResponse_WHISPER,
			}); err != nil {
				errCh <- err
				return
			}

			// Accumulate into full transcription buffer
			fullTranscription.WriteString(resp.GetText())
			fullTranscription.WriteRune(' ')
		}
	}()

	// 3) Forward incoming audio chunks from client → Whisper
	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		if err := whisperStream.Send(chunk); err != nil {
			return err
		}
	}
	// Wait for the whisper goroutine to finish
	if err := <-errCh; err != nil {
		return err
	}

	// 4) Call Llama with the full transcription
	prompt := fullTranscription.String()
	log.Printf("Full transcription: %q", prompt)

	llamaReq := &pb.GenerateRequest{
		Prompt:      prompt,
		MaxTokens:   256,
		Temperature: 0.7,
		TopP:        0.9,
	}
	llamaStream, err := s.llamaClient.Generate(ctx, llamaReq)
	if err != nil {
		return err
	}

	// 5) Stream Llama → client
	for {
		genResp, err := llamaStream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}

		if err := stream.Send(&pb.StreamAudioResponse{
			Text:   genResp.GetText(),
			Done:   genResp.GetDone(),
			Source: pb.StreamAudioResponse_LLAMA,
		}); err != nil {
			return err
		}
	}

	log.Println("Completed Llama generation; closing stream")
	return nil
}

func main() {
	// Dial Whisper
	wConn, err := grpc.Dial(whisperSockPath, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Failed to connect to Whisper: %v", err)
	}
	defer wConn.Close()
	whisperClient := pb.NewWhisperServiceClient(wConn)

	// Dial Llama
	lConn, err := grpc.Dial(llamaSockPath, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Failed to connect to Llama: %v", err)
	}
	defer lConn.Close()
	llamaClient := pb.NewLlamaServiceClient(lConn)

	log.Println("=== gRPC Audio Orchestrator ===")
	log.Println("Whisper at", whisperSockPath, "  Llama at", llamaSockPath)

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
