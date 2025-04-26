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

    "github.com/google/uuid"
    "google.golang.org/grpc"
    "google.golang.org/grpc/credentials/insecure"
)

const (
    // Server configuration
    listenAddr      = ":8090"
    whisperSockPath = "unix:///app/sockets/whisper.sock"
    llamaSockPath   = "unix:///app/llama-sockets/llama.sock"
    piperSockPath   = "unix:///app/piper-sockets/piper.sock"
)

// WhisperServiceServer orchestrates Whisper (ASR), Llama (LLM) and Piper (TTS).
// Each gRPC client connection gets its own in‑memory transcript so Llama sees the whole dialog.
type WhisperServiceServer struct {
    pb.UnimplementedWhisperServiceServer
    whisperClient pb.WhisperServiceClient
    llamaClient   pb.LlamaServiceClient
    ttsClient     pb.TextToSpeechClient
}

// StreamAudio – bidirectional: client mic  → Whisper  → Llama → Piper → client speaker.
func (s *WhisperServiceServer) StreamAudio(stream pb.WhisperService_StreamAudioServer) error {
    convID := uuid.NewString()
    log.Printf("[conv %s] client connected", convID)

    ctx := stream.Context()
    transcript := make([]string, 0, 64) // freed when stream ends

    // ─── 1. open Whisper stream ────────────────────────────────────────────────
    whisperStream, err := s.whisperClient.StreamAudio(ctx)
    if err != nil {
        return err
    }
    defer whisperStream.CloseSend()

    // ─── 2. goroutine: Whisper → Llama → Piper, sending only FINAL Llama reply ──
    errCh := make(chan error, 1)
    go func() {
        seg := 0
        for {
            // a) read ASR segment
            wResp, err := whisperStream.Recv()
            if err == io.EOF {
                errCh <- nil; return
            }
            if err != nil {
                errCh <- err; return
            }
            text := strings.TrimSpace(wResp.GetText())
            if text == "" {
                continue
            }
            seg++
            log.Printf("[conv %s] ASR #%d: %q", convID, seg, text)

            // b) append USER turn to transcript and forward to client (optional)
            transcript = append(transcript, "user: "+text)
            if err := stream.Send(&pb.StreamAudioResponse{Text: text, Source: pb.StreamAudioResponse_WHISPER}); err != nil {
                errCh <- err; return
            }

            // c) build prompt & call Llama
            prompt := strings.Join(transcript, "\n") + "\nassistant:"
            llamaReq := &pb.GenerateRequest{Prompt: prompt, MaxTokens: 256, Temperature: 0.7, TopP: 0.9}
            lStream, err := s.llamaClient.Generate(ctx, llamaReq)
            if err != nil {
                log.Printf("[conv %s] Llama error: %v", convID, err)
                continue
            }
            var fullReply strings.Builder
            for {
                lResp, err := lStream.Recv()
                if err == io.EOF {
                    break
                }
                if err != nil {
                    log.Printf("[conv %s] Llama stream error: %v", convID, err); break
                }
                fullReply.WriteString(lResp.GetText())
                if lResp.GetDone() {
                    break
                }
            }
            reply := strings.TrimSpace(fullReply.String())
            if reply == "" {
                continue
            }

            // d) append ASSISTANT turn to transcript
            transcript = append(transcript, "assistant: "+reply)

            // e) **single** message back to client with complete reply
            if err := stream.Send(&pb.StreamAudioResponse{
                Text:   reply,
                Source: pb.StreamAudioResponse_LLAMA,
                Done:   true, // single packet, marked done
            }); err != nil {
                errCh <- err; return
            }

            // f) synthesize TTS
            ttsReq := &pb.TextRequest{Text: reply, SpeakingRate: 1.0}
            tStream, err := s.ttsClient.SynthesizeText(ctx, ttsReq)
            if err != nil {
                log.Printf("[conv %s] TTS error: %v", convID, err)
                continue
            }
            for {
                aResp, err := tStream.Recv()
                if err == io.EOF {
                    break
                }
                if err != nil {
                    log.Printf("[conv %s] TTS stream error: %v", convID, err); break
                }
                if err := stream.Send(&pb.StreamAudioResponse{
                    AudioData:  aResp.GetAudioChunk(),
                    SampleRate: aResp.GetSampleRate(),
                    IsEndAudio: aResp.GetIsEnd(),
                    Source:     pb.StreamAudioResponse_TTS,
                }); err != nil {
                    errCh <- err; return
                }
                if aResp.GetIsEnd() {
                    break
                }
            }
        }
    }()

    // ─── 3. client mic → Whisper ───────────────────────────────────────────────
    for {
        cChunk, err := stream.Recv()
        if err == io.EOF {
            break
        }
        if err != nil {
            return err
        }
        if err := whisperStream.Send(cChunk); err != nil {
            return err
        }
    }
    whisperStream.CloseSend()
    if err := <-errCh; err != nil {
        return err
    }
    return nil
}

// main unchanged except for imports / log strings ─────────────────────────────
func main() {
    log.Println("voice‑bot orchestrator starting…")

    wConn, err := grpc.Dial(whisperSockPath, grpc.WithTransportCredentials(insecure.NewCredentials()))
    if err != nil {
        log.Fatalf("Whisper dial: %v", err)
    }
    defer wConn.Close()
    lConn, err := grpc.Dial(llamaSockPath, grpc.WithTransportCredentials(insecure.NewCredentials()))
    if err != nil {
        log.Fatalf("Llama dial: %v", err)
    }
    defer lConn.Close()
    pConn, err := grpc.Dial(piperSockPath, grpc.WithTransportCredentials(insecure.NewCredentials()))
    if err != nil {
        log.Fatalf("Piper dial: %v", err)
    }
    defer pConn.Close()

    grpcServer := grpc.NewServer()
    pb.RegisterWhisperServiceServer(grpcServer, &WhisperServiceServer{
        whisperClient: pb.NewWhisperServiceClient(wConn),
        llamaClient:   pb.NewLlamaServiceClient(lConn),
        ttsClient:     pb.NewTextToSpeechClient(pConn),
    })

    lis, err := net.Listen("tcp", listenAddr)
    if err != nil {
        log.Fatalf("listen: %v", err)
    }
    log.Println("listening on", listenAddr)

    sigCh := make(chan os.Signal, 1)
    signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
    go func() { <-sigCh; grpcServer.GracefulStop() }()

    if err := grpcServer.Serve(lis); err != nil {
        log.Fatalf("serve: %v", err)
    }
}
