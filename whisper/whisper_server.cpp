#include "whisper_server.h"
#include <fstream>
#include <thread>
#include <csignal>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/resource.h>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <string>
#include <deque>
#include <iostream>
#include <iomanip>  // Added missing header for std::setprecision

static const std::string SOCKET_PATH = "/app/sockets/whisper.sock";
static const char* MODEL_PATH        = "/app/models/ggml-base.en.bin";

// Buffer parameters
static const size_t MAX_AUDIO_BUFFER = 30 * 16000 * sizeof(int16_t); // 30 seconds max buffer
static const size_t MIN_PROCESS_SIZE = 0.5 * 16000 * sizeof(int16_t); // Min 0.5 seconds for processing

// Logging helper
#define LOG_INFO(msg) do { std::cout << "[INFO] " << msg << std::endl; } while(0)
#define LOG_ERROR(msg) do { std::cerr << "[ERROR] " << msg << std::endl; } while(0)
#define LOG_DEBUG(msg) do { std::cout << "[DEBUG] " << msg << std::endl; } while(0)

// ContextGuard (unchanged)
ContextGuard::ContextGuard(whisper_context* ctx, WhisperContextPool* pool)
    : ctx_(ctx), pool_(pool) {}
ContextGuard::~ContextGuard() { pool_->release(ctx_); }

// WhisperContextPool (unchanged)
WhisperContextPool::WhisperContextPool(size_t count) {
    auto params = whisper_context_default_params();
    params.use_gpu = false;
    for (size_t i = 0; i < count; ++i) {
        auto* ctx = whisper_init_from_file_with_params(MODEL_PATH, params);
        if (!ctx) exit(1);
        pool_.push(ctx);
    }
}
WhisperContextPool::~WhisperContextPool() {
    while (!pool_.empty()) {
        whisper_free(pool_.front());
        pool_.pop();
    }
}
ContextGuardPtr WhisperContextPool::acquire() {
    std::unique_lock<std::mutex> lk(mux_);
    cond_.wait(lk, [&]{ return !pool_.empty(); });
    auto* ctx = pool_.front(); pool_.pop();
    return std::make_unique<ContextGuard>(ctx, this);
}
void WhisperContextPool::release(whisper_context* ctx) {
    std::lock_guard<std::mutex> lk(mux_);
    pool_.push(ctx);
    cond_.notify_one();
}

// Updated Service Implementation with explicit speech boundary handling
WhisperServiceImpl::WhisperServiceImpl(WhisperContextPool& p) : pool_(p) {}

grpc::Status WhisperServiceImpl::StreamAudio(
    grpc::ServerContext*,
    grpc::ServerReaderWriter<whisper::Transcription, whisper::AudioChunk>* stream
) {
    static int session_counter = 0;
    int session_id = ++session_counter;
    
    LOG_INFO("Session " << session_id << ": New audio stream connection started");
    
    auto guard = pool_.acquire();
    auto* ctx = guard->get();

    // Prepare whisper parameters 
    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.print_realtime   = false;
    wparams.print_progress   = false;
    wparams.print_timestamps = true;
    wparams.translate        = false;
    wparams.language         = "en";
    wparams.n_threads        = int(std::thread::hardware_concurrency());
    wparams.no_context       = false;
    wparams.single_segment   = false; // Process as a whole utterance
    wparams.max_tokens       = 0;     // Use full segment output
    wparams.audio_ctx        = 0;

    LOG_INFO("Session " << session_id << ": Parameters configured");

    // Token context storage
    std::vector<whisper_token> prompt_tokens;
    const int N_CTX = whisper_n_text_ctx(ctx);

    // Audio buffer for complete utterances
    std::vector<float> audio_buffer;
    audio_buffer.reserve(MAX_AUDIO_BUFFER / sizeof(float));
    
    // Speech detection state
    bool in_speech_segment = false;
    bool processed_segment = false;
    
    // Helper: convert incoming int16 data to float
    auto to_f32 = [&](const std::string& in) {
        size_t n = in.size() / sizeof(int16_t);
        std::vector<float> out(n);
        auto d = reinterpret_cast<const int16_t*>(in.data());
        for (size_t i = 0; i < n; ++i) {
            out[i] = d[i] / 32768.0f;
        }
        return out;
    };

    // Helper: process current audio buffer
    auto process_buffer = [&](bool is_final) -> bool {
        if (audio_buffer.empty() || audio_buffer.size() < MIN_PROCESS_SIZE / sizeof(float)) {
            LOG_INFO("Session " << session_id << ": Buffer too small to process, size=" 
                     << audio_buffer.size() << " samples ("
                     << (audio_buffer.size() / WHISPER_SAMPLE_RATE) << "s)");
            return false;
        }

        LOG_INFO("Session " << session_id << ": Processing buffer with " 
                 << audio_buffer.size() << " samples ("
                 << std::fixed << std::setprecision(2) 
                 << static_cast<float>(audio_buffer.size()) / WHISPER_SAMPLE_RATE << "s)");
        
        auto process_start = std::chrono::steady_clock::now();
        
        // Determine whether to use prompt tokens based on current speech context
        wparams.prompt_n_tokens = prompt_tokens.size();
        wparams.prompt_tokens   = prompt_tokens.empty() ? nullptr : prompt_tokens.data();
        
        if (prompt_tokens.size() > 0) {
            LOG_DEBUG("Session " << session_id << ": Using " << prompt_tokens.size() 
                      << " context tokens");
        }
        
        if (whisper_full(ctx, wparams, audio_buffer.data(), audio_buffer.size()) != 0) {
            LOG_ERROR("Session " << session_id << ": whisper_full() failed");
            return false;
        }

        auto process_end = std::chrono::steady_clock::now();
        auto process_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            process_end - process_start).count();
        
        // Prepare output text
        std::string out_text = "";
        int n_segments = whisper_full_n_segments(ctx);
        
        LOG_INFO("Session " << session_id << ": Processing completed in " << process_ms 
                  << "ms, found " << n_segments << " segment(s)");
        
        for (int i = 0; i < n_segments; ++i) {
            const char* seg = whisper_full_get_segment_text(ctx, i);
            if (seg && *seg) {
                out_text += seg;
                LOG_DEBUG("Session " << session_id << ": Segment " << i << ": \"" << seg << "\"");
            }
        }

        // If we have text, send it to client with a proper line ending
        if (!out_text.empty()) {
            whisper::Transcription out;
            out.set_text(out_text + "\n");
            
            LOG_INFO("Session " << session_id << ": Sending transcription: \""
                      << out_text << "\"");
            
            if (!stream->Write(out)) {
                LOG_ERROR("Session " << session_id << ": Failed to send transcription to client");
                return false;
            }
        } else {
            LOG_INFO("Session " << session_id << ": No transcription text to send");
        }

        // Update prompt tokens for context if continuing in the same speech segment
        if (!is_final) {
            prompt_tokens.clear();
            for (int i = 0; i < n_segments; ++i) {
                int nt = whisper_full_n_tokens(ctx, i);
                for (int j = 0; j < nt; ++j) {
                    prompt_tokens.push_back(whisper_full_get_token_id(ctx, i, j));
                }
            }
            if ((int)prompt_tokens.size() > N_CTX) {
                LOG_DEBUG("Session " << session_id << ": Truncating context tokens from " 
                         << prompt_tokens.size() << " to " << N_CTX);
                prompt_tokens.erase(prompt_tokens.begin(), prompt_tokens.end() - N_CTX);
            }
        } else {
            // Clear context tokens at the end of a speech segment
            prompt_tokens.clear();
            LOG_INFO("Session " << session_id << ": Cleared context tokens (end of segment)");
        }

        processed_segment = true;
        return true;
    };

    whisper::AudioChunk chunk;
    while (stream->Read(&chunk)) {
        // Print marker status for every chunk
        LOG_DEBUG("Session " << session_id << ": Received chunk - speech_start: " 
                 << (chunk.speech_start() ? "true" : "false") 
                 << ", speech_end: " << (chunk.speech_end() ? "true" : "false")
                 << ", data size: " << chunk.data().size() << " bytes");
                 
        // Check for speech boundary markers
        if (chunk.speech_start()) {
            LOG_INFO("Session " << session_id << ": Received SPEECH_START marker");
            
            // Reset for new speech segment
            in_speech_segment = true;
            prompt_tokens.clear(); // Clear any previous context
            audio_buffer.clear();  // Start with fresh buffer
            processed_segment = false;
            
            continue; // Skip to next chunk
        }
        
        if (chunk.speech_end()) {
            LOG_INFO("Session " << session_id << ": Received SPEECH_END marker");
            
            // Process any remaining audio with finality
            if (!audio_buffer.empty()) {
                LOG_INFO("Session " << session_id << ": Processing final buffer for speech segment");
                process_buffer(true); // true = final processing
            } else if (!processed_segment) {
                LOG_INFO("Session " << session_id << ": Empty buffer at speech end, nothing to process");
            }
            
            // Reset state
            in_speech_segment = false;
            audio_buffer.clear();
            
            continue; // Skip to next chunk
        }
        
        // Handle regular audio data
        auto data = to_f32(chunk.data());
        if (!data.empty()) {
            // Only accumulate if we're in an active speech segment or it came with a start marker
            if (in_speech_segment || chunk.speech_start()) {
                size_t prev_size = audio_buffer.size();
                audio_buffer.insert(audio_buffer.end(), data.begin(), data.end());
                
                LOG_DEBUG("Session " << session_id << ": Added " << data.size() 
                         << " samples to buffer, now " << audio_buffer.size() 
                         << " samples");
                
                // If buffer gets too large, process intermediate results
                if (audio_buffer.size() * sizeof(float) > MAX_AUDIO_BUFFER / 4) {
                    LOG_INFO("Session " << session_id << ": Buffer large enough for intermediate processing");
                    process_buffer(false); // false = not final processing
                    
                    // Keep a portion of the buffer for context continuity
                    size_t keep_samples = std::min(audio_buffer.size(), (size_t)(0.5 * WHISPER_SAMPLE_RATE));
                    if (keep_samples < audio_buffer.size()) {
                        std::vector<float> temp(audio_buffer.end() - keep_samples, audio_buffer.end());
                        audio_buffer = temp;
                        LOG_DEBUG("Session " << session_id << ": Kept " << keep_samples 
                                 << " samples for context continuity");
                    }
                }
            } else {
                LOG_DEBUG("Session " << session_id << ": Received audio outside speech segment, ignoring");
            }
        }
    }
    
    // Process any remaining audio at the end
    if (!audio_buffer.empty() && in_speech_segment) {
        LOG_INFO("Session " << session_id << ": Processing remaining buffer before closing");
        process_buffer(true); // true = final processing
    }

    LOG_INFO("Session " << session_id << ": Stream closed");
    return grpc::Status::OK;
}

// socket setup / main() (unchanged)
void cleanup_socket() {
    if (access(SOCKET_PATH.c_str(), F_OK) == 0) unlink(SOCKET_PATH.c_str());
}
void RunServer() {
    mkdir("/app/sockets", 0777);
    cleanup_socket();
    std::string addr = "unix:" + SOCKET_PATH;
    WhisperContextPool pool(std::max(1, int(std::thread::hardware_concurrency())));
    WhisperServiceImpl svc(pool);
    grpc::ServerBuilder b;
    b.AddListeningPort(addr, grpc::InsecureServerCredentials());
    b.RegisterService(&svc);
    auto s = b.BuildAndStart();
    std::ofstream("/app/sockets/ready").close();
    LOG_INFO("Server ready to accept connections");
    s->Wait();
}
int main() {
    std::signal(SIGINT,  [](int){ cleanup_socket(); exit(0); });
    std::signal(SIGTERM, [](int){ cleanup_socket(); exit(0); });
    setpriority(PRIO_PROCESS, 0, -10);
    RunServer();
    return 0;
}