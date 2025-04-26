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
    listenAddr      = ":8090"
    whisperSockPath = "unix:///app/sockets/whisper.sock"
    llamaSockPath   = "unix:///app/llama-sockets/llama.sock"
    piperSockPath   = "unix:///app/piper-sockets/piper.sock"
)

// single line system prompt
const systemPrompt = "system: You are a concise conversational voice assistant. DO NOT invent user turns or ask follow‑up questions. Just answer briefly."

// server orchestration
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
    transcript := []string{systemPrompt}

    whisperStream, err := s.whisperClient.StreamAudio(ctx)
    if err != nil {
        return err
    }
    defer whisperStream.CloseSend()

    errCh := make(chan error, 1)
    go func() {
        for seg := 1; ; seg++ {
            // ── receive ASR ───────────────────────────────────────────────────
            wResp, err := whisperStream.Recv()
            if err == io.EOF { errCh <- nil; return }
            if err != nil     { errCh <- err; return }

            text := strings.TrimSpace(wResp.GetText())
            if text == "" { seg--; continue }
            log.Printf("[conv %s] ASR #%d: %q", convID, seg, text)

            transcript = append(transcript, "user: "+text)
            if err := stream.Send(&pb.StreamAudioResponse{Text: text, Source: pb.StreamAudioResponse_WHISPER}); err != nil {
                errCh <- err; return
            }

            // ── build prompt & call Llama ────────────────────────────────────
            prompt := strings.Join(transcript, "\n") + "\nassistant:"
            llamaReq := &pb.GenerateRequest{Prompt: prompt, MaxTokens: 128, Temperature: 0.7, TopP: 0.9}
            lStream, err := s.llamaClient.Generate(ctx, llamaReq)
            if err != nil { log.Printf("[conv %s] Llama error: %v", convID, err); continue }

            var full strings.Builder
            outer: for {
                lResp, err := lStream.Recv()
                if err == io.EOF { break }
                if err != nil   { log.Printf("[conv %s] Llama stream error: %v", convID, err); break }

                piece := lResp.GetText()
                full.WriteString(piece)

                // stop if model starts a new turn
                if idx := strings.Index(full.String(), "\nuser:"); idx != -1 {
                    tmp := full.String()[:idx]
					full.Reset()
					full.WriteString(tmp)
                    break outer
                }
                if idx := strings.Index(full.String(), "\nassistant:"); idx != -1 && idx != 0 {
                    tmp := full.String()[:idx]
					full.Reset()
					full.WriteString(tmp)
                    break outer
                }

                if lResp.GetDone() { break }
            }
            reply := strings.TrimSpace(full.String())
            if reply == "" { continue }

            transcript = append(transcript, "assistant: "+reply)

            if err := stream.Send(&pb.StreamAudioResponse{Text: reply, Source: pb.StreamAudioResponse_LLAMA, Done: true}); err != nil {
                errCh <- err; return
            }

            // ── TTS synthesis ────────────────────────────────────────────────
            ttsReq := &pb.TextRequest{Text: reply, SpeakingRate: 1.0}
            tStream, err := s.ttsClient.SynthesizeText(ctx, ttsReq)
            if err != nil { log.Printf("[conv %s] TTS error: %v", convID, err); continue }
            for {
                aResp, err := tStream.Recv()
                if err == io.EOF { break }
                if err != nil   { log.Printf("[conv %s] TTS stream error: %v", convID, err); break }
                if err := stream.Send(&pb.StreamAudioResponse{AudioData: aResp.GetAudioChunk(), SampleRate: aResp.GetSampleRate(), IsEndAudio: aResp.GetIsEnd(), Source: pb.StreamAudioResponse_TTS}); err != nil {
                    errCh <- err; return
                }
                if aResp.GetIsEnd() { break }
            }
        }
    }()

    // ── client mic → Whisper ─────────────────────────────────────────────────
    for {
        cChunk, err := stream.Recv()
        if err == io.EOF { break }
        if err != nil   { return err }
        if err := whisperStream.Send(cChunk); err != nil { return err }
    }
    whisperStream.CloseSend()
    if err := <-errCh; err != nil { return err }
    return nil
}

// ─────────────────────────────────────────────────────────────────────────────
func main() {
    log.Println("voice‑bot orchestrator starting…")

    wConn, err := grpc.Dial(whisperSockPath, grpc.WithTransportCredentials(insecure.NewCredentials()))
    if err != nil { log.Fatalf("Whisper dial: %v", err) }
    defer wConn.Close()

    lConn, err := grpc.Dial(llamaSockPath, grpc.WithTransportCredentials(insecure.NewCredentials()))
    if err != nil { log.Fatalf("Llama dial: %v", err) }
    defer lConn.Close()

    pConn, err := grpc.Dial(piperSockPath, grpc.WithTransportCredentials(insecure.NewCredentials()))
    if err != nil { log.Fatalf("Piper dial: %v", err) }
    defer pConn.Close()

    grpcServer := grpc.NewServer()
    pb.RegisterWhisperServiceServer(grpcServer, &WhisperServiceServer{
        whisperClient: pb.NewWhisperServiceClient(wConn),
        llamaClient:   pb.NewLlamaServiceClient(lConn),
        ttsClient:     pb.NewTextToSpeechClient(pConn),
    })

    lis, err := net.Listen("tcp", listenAddr)
    if err != nil { log.Fatalf("listen: %v", err) }
    log.Println("listening on", listenAddr)

    sigCh := make(chan os.Signal, 1)
    signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
    go func() { <-sigCh; grpcServer.GracefulStop() }()

    if err := grpcServer.Serve(lis); err != nil { log.Fatalf("serve: %v", err) }
}
