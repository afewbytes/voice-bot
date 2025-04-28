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
#include <iomanip>  // std::setprecision

static const std::string SOCKET_PATH = "/app/sockets/whisper.sock";
static const char* MODEL_PATH        = "/app/models/kb-ggml-model.bin";

// Buffer parameters
static const size_t MAX_AUDIO_BUFFER = 30 * 16000 * sizeof(int16_t); // 30 seconds max buffer
static const size_t MIN_PROCESS_SIZE = 0.5 * 16000 * sizeof(int16_t); // Min 0.5 seconds for processing

// Logging helper
#define LOG_INFO(msg)  do { std::cout << "[INFO] "  << msg << std::endl; } while(0)
#define LOG_ERROR(msg) do { std::cerr << "[ERROR] " << msg << std::endl; } while(0)
#define LOG_DEBUG(msg) do { std::cout << "[DEBUG] "<< msg << std::endl; } while(0)

//-------------------------------------------------------------------
// ContextGuard (unchanged)
ContextGuard::ContextGuard(whisper_context* ctx, WhisperContextPool* pool)
    : ctx_(ctx), pool_(pool) {}
ContextGuard::~ContextGuard() { pool_->release(ctx_); }

//-------------------------------------------------------------------
// WhisperContextPool (unchanged)
// ------------------------------------------------------------------
WhisperContextPool::WhisperContextPool(size_t count) {

    // pre-flight: make sure the model file is present
    struct stat st;
    if (stat(MODEL_PATH, &st) != 0 || st.st_size == 0) {
        LOG_ERROR("Model file '" << MODEL_PATH << "' is missing or empty");
        std::exit(EXIT_FAILURE);
    }

    // GPU-enabled context
    whisper_context_params params = whisper_context_default_params();
    params.use_gpu    = true;   // enable CUDA backend 🔥
    params.gpu_device = 0;      // first visible GPU (change if you bind multiple)

    for (size_t i = 0; i < count; ++i) {
        whisper_context *ctx = whisper_init_from_file_with_params(MODEL_PATH, params);
        if (!ctx) {
            LOG_ERROR("whisper_init_from_file_with_params() failed");
            std::exit(EXIT_FAILURE);
        }
        pool_.push(ctx);
    }
}
WhisperContextPool::~WhisperContextPool() {
    while (!pool_.empty()) { whisper_free(pool_.front()); pool_.pop(); }
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

//-------------------------------------------------------------------
// Service implementation
WhisperServiceImpl::WhisperServiceImpl(WhisperContextPool& p) : pool_(p) {}

grpc::Status WhisperServiceImpl::StreamAudio(
    grpc::ServerContext*,
    grpc::ServerReaderWriter<voice::StreamAudioResponse, voice::AudioChunk>* stream) {

    static int session_counter = 0;
    int session_id = ++session_counter;
    LOG_INFO("Session " << session_id << ": new audio stream");

    auto guard = pool_.acquire();
    auto* ctx = guard->get();

    // Whisper params
    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.print_realtime=false; wparams.print_progress=false;
    wparams.print_timestamps=true; wparams.language="en";
    wparams.n_threads = int(std::thread::hardware_concurrency());
    wparams.language = "sv";

    std::vector<whisper_token> prompt_tokens;
    const int N_CTX = whisper_n_text_ctx(ctx);

    std::vector<float> audio_buffer; audio_buffer.reserve(MAX_AUDIO_BUFFER/sizeof(float));

    bool in_speech=false, processed_segment=false;
    int  last_sent_segment=0;   // << NEW: remember what we already streamed

    auto to_f32 = [&](const std::string& in){
        size_t n = in.size()/sizeof(int16_t);
        std::vector<float> out(n);
        auto d=reinterpret_cast<const int16_t*>(in.data());
        for(size_t i=0;i<n;++i) out[i]=d[i]/32768.0f;
        return out;
    };

    auto process_buffer=[&](bool is_final)->bool {
        if(audio_buffer.size()<MIN_PROCESS_SIZE/sizeof(float)) return false;

        if(whisper_full(ctx,wparams,audio_buffer.data(),audio_buffer.size())!=0){
            LOG_ERROR("whisper_full failed"); return false; }

        int n_segments = whisper_full_n_segments(ctx);
        std::string out_text;
        for(int i=last_sent_segment;i<n_segments;++i){
            const char* seg=whisper_full_get_segment_text(ctx,i);
            if(seg&&*seg){ out_text+=seg; }
        }
        last_sent_segment=n_segments;          // update progress
        if(is_final) last_sent_segment=0;      // reset for next utterance

        if(!out_text.empty()){
            voice::StreamAudioResponse resp;
            resp.set_text(out_text+"\n");
            resp.set_source(voice::StreamAudioResponse_Source_WHISPER);
            resp.set_done(is_final);
            stream->Write(resp);
        }

        // manage context tokens for partial results
        if(!is_final){
            prompt_tokens.clear();
            for(int i=0;i<n_segments;++i){
                int nt=whisper_full_n_tokens(ctx,i);
                for(int j=0;j<nt;++j) prompt_tokens.push_back(whisper_full_get_token_id(ctx,i,j));
            }
            if((int)prompt_tokens.size()>N_CTX)
                prompt_tokens.erase(prompt_tokens.begin(), prompt_tokens.end()-N_CTX);
        } else {
            prompt_tokens.clear();
        }
        processed_segment=true; return true;
    };

    voice::AudioChunk chunk;
    while(stream->Read(&chunk)){
        if(chunk.speech_start()){
            in_speech=true; processed_segment=false; audio_buffer.clear();
            prompt_tokens.clear(); last_sent_segment=0; continue;
        }
        if(chunk.speech_end()){
            if(!audio_buffer.empty()) process_buffer(true);
            in_speech=false; audio_buffer.clear(); last_sent_segment=0; continue;
        }
        auto data=to_f32(chunk.data());
        if(!data.empty() && in_speech){
            audio_buffer.insert(audio_buffer.end(),data.begin(),data.end());
            if(audio_buffer.size()*sizeof(float) > MAX_AUDIO_BUFFER/4){
                process_buffer(false);
                size_t keep=std::min(audio_buffer.size(), (size_t)(0.5*WHISPER_SAMPLE_RATE));
                if(keep<audio_buffer.size())
                    audio_buffer.assign(audio_buffer.end()-keep, audio_buffer.end());
            }
        }
    }
    if(!audio_buffer.empty()&&in_speech) process_buffer(true);
    LOG_INFO("Session "<<session_id<<": stream closed");
    return grpc::Status::OK;
}

//-------------------------------------------------------------------
void cleanup_socket(){ if(access(SOCKET_PATH.c_str(),F_OK)==0) unlink(SOCKET_PATH.c_str()); }
void RunServer(){
    mkdir("/app/sockets",0777); cleanup_socket();
    std::string addr="unix:" + SOCKET_PATH;
    //WhisperContextPool pool(std::max(1,int(std::thread::hardware_concurrency())));
    WhisperContextPool pool(1);
    WhisperServiceImpl svc(pool);
    grpc::ServerBuilder b; b.AddListeningPort(addr,grpc::InsecureServerCredentials());
    b.RegisterService(&svc);
    auto s=b.BuildAndStart(); std::ofstream("/app/sockets/ready").close();
    LOG_INFO("Whisper server ready"); s->Wait();
}
int main(){ std::signal(SIGINT,[](int){cleanup_socket();exit(0);}); std::signal(SIGTERM,[](int){cleanup_socket();exit(0);}); setpriority(PRIO_PROCESS,0,-10); RunServer(); return 0; }
