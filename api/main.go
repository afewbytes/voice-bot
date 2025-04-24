package main

import (
	"io"
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"

	pb "voice-bot/proto"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const (
	// Server configuration
	listenAddr      = ":8090"
	whisperSockPath = "unix:///app/sockets/whisper.sock"
)

// WhisperServiceServer implements the gRPC server for WhisperService
type WhisperServiceServer struct {
	pb.UnimplementedWhisperServiceServer
	whisperClient pb.WhisperServiceClient
}

// StreamAudio implements the gRPC StreamAudio method
func (s *WhisperServiceServer) StreamAudio(stream pb.WhisperService_StreamAudioServer) error {
	log.Println("New gRPC client connected")
	
	// Use the stream's context
	ctx := stream.Context()
	
	// Create a stream to the whisper service
	whisperStream, err := s.whisperClient.StreamAudio(ctx)
	if err != nil {
		log.Printf("Failed to create whisper stream: %v", err)
		return err
	}
	defer whisperStream.CloseSend()
	
	// Channel to collect errors from goroutines
	errCh := make(chan error, 1)
	
	// Start goroutine to forward responses from whisper service back to client
	go func() {
		for {
			resp, err := whisperStream.Recv()
			if err == io.EOF {
				errCh <- nil
				return
			}
			if err != nil {
				log.Printf("Error receiving from whisper: %v", err)
				errCh <- err
				return
			}
			
			// Send transcription to client
			if err := stream.Send(resp); err != nil {
				log.Printf("Error sending to client: %v", err)
				errCh <- err
				return
			}
			
			log.Printf("Sent transcription: %s", resp.GetText())
		}
	}()
	
	// Process incoming audio chunks from client
	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Printf("Error receiving from client: %v", err)
			return err
		}
		
		// Log speech markers
		if chunk.SpeechStart {
			log.Println("Speech start marker received")
		}
		if chunk.SpeechEnd {
			log.Println("Speech end marker received")
		}
		
		// Forward the chunk to whisper service
		if err := whisperStream.Send(chunk); err != nil {
			log.Printf("Error forwarding to whisper: %v", err)
			return err
		}
		
		// Log data size if present
		if len(chunk.Data) > 0 {
			log.Printf("Forwarded %d bytes of audio data", len(chunk.Data))
		}
	}
	
	// Wait for transcription goroutine to complete
	if err := <-errCh; err != nil {
		return err
	}
	
	log.Println("Client disconnected")
	return nil
}

func main() {
	// Connect to whisper service
	conn, err := grpc.Dial(whisperSockPath, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Failed to connect to whisper service: %v", err)
	}
	defer conn.Close()
	whisperClient := pb.NewWhisperServiceClient(conn)

	log.Println("=== gRPC Whisper Audio Server ===")
	log.Println("Connected to whisper service:", whisperSockPath)

	// Create gRPC server
	grpcServer := grpc.NewServer()
	pb.RegisterWhisperServiceServer(grpcServer, &WhisperServiceServer{
		whisperClient: whisperClient,
	})

	// Listen for gRPC connections
	lis, err := net.Listen("tcp", listenAddr)
	if err != nil {
		log.Fatalf("Failed to listen: %v", err)
	}
	defer lis.Close()
	
	log.Println("Listening on", listenAddr, "for gRPC clients")

	// Handle graceful shutdown
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	
	go func() {
		<-sigCh
		log.Println("Shutting down gRPC server")
		grpcServer.GracefulStop()
	}()

	// Serve gRPC requests
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("Failed to serve: %v", err)
	}
}