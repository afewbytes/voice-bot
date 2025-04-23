#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <fstream>
#include <thread>
#include <unistd.h>

#include <grpcpp/grpcpp.h>
#include "whisper.grpc.pb.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/resource.h>   

// Include whisper.h
#include "whisper.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using grpc::ServerReader;
using whisper::WhisperService;
using whisper::AudioChunk;
using whisper::Transcription;

// Socket path constant
const std::string SOCKET_PATH = "/app/sockets/whisper.sock";

class WhisperServiceImpl final : public WhisperService::Service {
private:
    struct whisper_context* ctx;
    
    bool init_whisper() {
        if (ctx != nullptr) {
            return true; // Already initialized
        }
        
        // Path to the whisper model
        const char* model_path = "/app/models/ggml-base.en.bin";
        
        // Check if model exists
        std::ifstream f(model_path);
        if (!f.good()) {
            std::cerr << "Model file does not exist: " << model_path << std::endl;
            return false;
        }
        
        std::cout << "Loading whisper model: " << model_path << std::endl;
        
        // Initialize whisper context with appropriate parameters
        whisper_context_params params = whisper_context_default_params();
        params.use_gpu = false;  // Enable GPU if available
        
        ctx = whisper_init_from_file_with_params(model_path, params);
        
        if (ctx == nullptr) {
            std::cerr << "Failed to initialize whisper context" << std::endl;
            return false;
        }
        
        std::cout << "Whisper model loaded successfully" << std::endl;
        return true;
    }
    
    // Convert int16 PCM audio bytes to float samples that whisper can process
    std::vector<float> convert_audio_bytes_to_float(const std::string& audio_bytes) {
        std::vector<float> pcm;
        
        // Check if we have enough bytes for at least one sample
        if (audio_bytes.size() < 2) {
            return pcm;
        }
        
        // Assuming 16-bit PCM audio (int16_t)
        const int16_t* samples = reinterpret_cast<const int16_t*>(audio_bytes.data());
        size_t n_samples = audio_bytes.size() / sizeof(int16_t);
        
        pcm.reserve(n_samples);
        
        // Convert int16 to float and normalize to [-1, 1]
        for (size_t i = 0; i < n_samples; ++i) {
            pcm.push_back(samples[i] / 32768.0f);
        }
        
        return pcm;
    }
    
public:
    WhisperServiceImpl() : ctx(nullptr) {
        init_whisper();
    }
    
    ~WhisperServiceImpl() {
        if (ctx) {
            whisper_free(ctx);
            ctx = nullptr;
        }
    }

    Status StreamAudio(ServerContext* context, ServerReader<AudioChunk>* reader, Transcription* response) override {
        if (!init_whisper()) {
            return Status(grpc::StatusCode::INTERNAL, "Failed to initialize whisper model");
        }
        
        AudioChunk chunk;
        std::vector<float> audio_data;
        
        std::cout << "Starting to receive audio chunks..." << std::endl;
        
        // Process incoming audio chunks
        while (reader->Read(&chunk)) {
            std::string chunk_data = chunk.data();
            std::cout << "Received audio chunk of size: " << chunk_data.size() << " bytes" << std::endl;
            
            if (chunk_data.empty()) {
                continue;
            }
            
            // Convert and append to our audio buffer
            auto pcm_chunk = convert_audio_bytes_to_float(chunk_data);
            audio_data.insert(audio_data.end(), pcm_chunk.begin(), pcm_chunk.end());
        }
        
        std::cout << "Finished receiving audio. Total samples: " << audio_data.size() << std::endl;
        
        if (audio_data.empty()) {
            response->set_text("No audio data received");
            return Status::OK;
        }
        
        // Set up whisper parameters for transcription
        whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        
        // Configure the parameters as needed
        wparams.print_realtime   = false;
        wparams.print_progress   = false;
        wparams.print_timestamps = false;
        wparams.print_special    = false;
        wparams.translate        = false;
        wparams.language         = "en";  // Set to desired language or auto-detect
        wparams.n_threads        = 1;
        wparams.offset_ms        = 0;
        wparams.no_context       = true;
        wparams.single_segment   = false; // Process as a single segment
        
        std::cout << "Processing audio with whisper..." << std::endl;
        
        // Process the audio
        if (whisper_full(ctx, wparams, audio_data.data(), audio_data.size()) != 0) {
            std::cerr << "Failed to process audio with whisper" << std::endl;
            response->set_text("Failed to process audio");
            return Status(grpc::StatusCode::INTERNAL, "Whisper processing failed");
        }
        
        // Extract the transcription results
        const int n_segments = whisper_full_n_segments(ctx);
        std::string transcription;
        
        for (int i = 0; i < n_segments; ++i) {
            const char* segment_text = whisper_full_get_segment_text(ctx, i);
            transcription += segment_text;
            
            // Add spacing between segments
            if (i < n_segments - 1) {
                transcription += " ";
            }
        }
        
        std::cout << "Transcription result: " << transcription << std::endl;
        response->set_text(transcription);
        return Status::OK;
    }
};

void cleanup_socket() {
    // Remove socket file if it exists
    if (access(SOCKET_PATH.c_str(), F_OK) != -1) {
        std::cout << "Removing existing socket file: " << SOCKET_PATH << std::endl;
        unlink(SOCKET_PATH.c_str());
    }
}

void RunServer() {
    std::string socket_addr = "unix:" + SOCKET_PATH;
    std::cout << whisper_print_system_info() << std::endl;

    struct rlimit rlim{RLIM_INFINITY, RLIM_INFINITY};
    setrlimit(RLIMIT_CORE, &rlim);
    
    // Ensure directory exists
    mkdir("/app/sockets", 0777);
    
    // Clean up any existing socket file
    cleanup_socket();
    
    WhisperServiceImpl service;

    ServerBuilder builder;
    builder.AddListeningPort(socket_addr, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    
    std::unique_ptr<Server> server(builder.BuildAndStart());
    if (server == nullptr) {
        std::cerr << "Failed to start server on " << socket_addr << std::endl;
        exit(1);
    }
    
    std::cout << "Server listening on " << socket_addr << std::endl;
    
    // Create a simple file to signal that the server is ready
    std::ofstream ready_file("/app/sockets/ready");
    ready_file.close();
    
    server->Wait();
}

int main(int argc, char** argv) {
    // Install signal handlers for graceful shutdown
    signal(SIGINT, [](int sig) {
        std::cout << "Received SIGINT, cleaning up..." << std::endl;
        cleanup_socket();
        exit(0);
    });
    
    signal(SIGTERM, [](int sig) {
        std::cout << "Received SIGTERM, cleaning up..." << std::endl;
        cleanup_socket();
        exit(0);
    });
    
    RunServer();
    return 0;
}