package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	pb "voice-bot/proto"
)

func main() {
	conn, err := grpc.Dial("unix:///app/sockets/whisper.sock", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("Failed to connect: %v", err)
	}
	defer conn.Close()

	client := pb.NewWhisperServiceClient(conn)
	stream, err := client.StreamAudio(context.Background())
	if err != nil {
		log.Fatalf("Error starting stream: %v", err)
	}

	// Simulate streaming 100ms audio chunks
	for i := 0; i < 10; i++ {
		chunk := &pb.AudioChunk{Data: []byte{0, 1, 2}} // Dummy data
		if err := stream.Send(chunk); err != nil {
			log.Fatalf("Send error: %v", err)
		}
		time.Sleep(100 * time.Millisecond)
	}

	reply, err := stream.CloseAndRecv()
	if err != nil {
		log.Fatalf("Recv error: %v", err)
	}
	fmt.Println("Transcription:", reply.Text)
}
