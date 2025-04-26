// api_server.go – orchestrator for Whisper, llama‑cpp and Piper
// Uses per‑conversation llama KV‑cache.  Sends only incremental user turn.

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
    "google.golang.org/grpc/metadata"
)

const (
    listenAddr      = ":8090"
    whisperSockPath = "unix:///app/sockets/whisper.sock"
    llamaSockPath   = "unix:///app/llama-sockets/llama.sock"
    piperSockPath   = "unix:///app/piper-sockets/piper.sock"
)

// system prompt sent only on the first user turn
const systemPrompt = "system: You are a concise conversational voice assistant. DO NOT invent user turns or ask follow‑up questions. Just answer briefly."

// ─────────────────────────────────────────────────────────────────────────────
type WhisperServiceServer struct {
    pb.UnimplementedWhisperServiceServer
    whisperClient pb.WhisperServiceClient
    llamaClient   pb.LlamaServiceClient
    ttsClient     pb.TextToSpeechClient
}

func (s *WhisperServiceServer) StreamAudio(stream pb.WhisperService_StreamAudioServer) error {
    convID := uuid.NewString()
    log.Printf("[conv %s] client connected", convID)

    ctx := stream.Context()

    whisperStream, err := s.whisperClient.StreamAudio(ctx)
    if err != nil {
        return err
    }
    defer whisperStream.CloseSend()

    errCh := make(chan error, 1)
    go func() {
        firstTurn := true
        for seg := 1; ; seg++ {
            // ── receive ASR ────────────────────────────────────────────────
            wResp, err := whisperStream.Recv()
            if err == io.EOF {
                errCh <- nil
                return
            }
            if err != nil {
                errCh <- err
                return
            }

            text := strings.TrimSpace(wResp.GetText())
            if text == "" {
                seg--
                continue
            }
            log.Printf("[conv %s] ASR #%d: %q", convID, seg, text)

            if err := stream.Send(&pb.StreamAudioResponse{Text: text, Source: pb.StreamAudioResponse_WHISPER}); err != nil {
                errCh <- err
                return
            }

            // ── build **incremental** prompt ─────────────────────────────
            var prompt string
            if firstTurn {
                prompt = systemPrompt + "\nuser: " + text + "\nassistant:"
                firstTurn = false
            } else {
                prompt = "user: " + text + "\nassistant:"
            }

            llamaReq := &pb.GenerateRequest{Prompt: prompt, MaxTokens: 128, Temperature: 0.7, TopP: 0.9}
            md := metadata.Pairs("conv-id", convID)
            lStream, err := s.llamaClient.Generate(metadata.NewOutgoingContext(ctx, md), llamaReq)
            if err != nil {
                log.Printf("[conv %s] Llama error: %v", convID, err)
                continue
            }

            var full strings.Builder
            for {
                lResp, err := lStream.Recv()
                if err == io.EOF {
                    break
                }
                if err != nil {
                    log.Printf("[conv %s] Llama stream error: %v", convID, err)
                    break
                }
                if lResp.GetDone() {
                    break
                }
                full.WriteString(lResp.GetText())
            }
            reply := strings.TrimSpace(full.String())
            if reply == "" {
                continue
            }

            if err := stream.Send(&pb.StreamAudioResponse{Text: reply, Source: pb.StreamAudioResponse_LLAMA, Done: true}); err != nil {
                errCh <- err
                return
            }

            // ── TTS synthesis ────────────────────────────────────────────
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
                    log.Printf("[conv %s] TTS stream error: %v", convID, err)
                    break
                }
                if err := stream.Send(&pb.StreamAudioResponse{AudioData: aResp.GetAudioChunk(), SampleRate: aResp.GetSampleRate(), IsEndAudio: aResp.GetIsEnd(), Source: pb.StreamAudioResponse_TTS}); err != nil {
                    errCh <- err
                    return
                }
                if aResp.GetIsEnd() {
                    break
                }
            }
        }
    }()

    // ── client mic → Whisper ────────────────────────────────────────────────
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

// ─────────────────────────────────────────────────────────────────────────────
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
    go func() {
        <-sigCh
        grpcServer.GracefulStop()
    }()

    if err := grpcServer.Serve(lis); err != nil {
        log.Fatalf("serve: %v", err)
    }
}
