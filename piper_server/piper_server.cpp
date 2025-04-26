#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <optional>
#include <functional>
#include <unistd.h>
#include <sys/stat.h>
#include <cerrno>
#include <cstring>
#include <chrono>
#include <fstream>

#include <grpcpp/grpcpp.h>

// Piper TTS
#include "piper.hpp"

// generated gRPC code
#include "voice.grpc.pb.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerReaderWriter;
using grpc::ServerWriter;
using grpc::Status;

// Forward declaration
bool run_piper_startup_test(piper::PiperConfig& piper_config, piper::Voice& voice);

class PiperTTSService final : public voice::TextToSpeech::Service {
public:
    PiperTTSService(const std::string& model_path, const std::string& config_path) {
        std::cout << "Loading Piper voice\n  model  : " << model_path
                  << "\n  config : " << config_path << std::endl;

        auto start = std::chrono::high_resolution_clock::now();

        /* ------------------------------------------------------------------ */
        /*  Initialise the global Piper runtime (eSpeak, OnnxRuntime, etc.)   */
        /* ------------------------------------------------------------------ */
        piper_config_ = piper::PiperConfig{};
        piper::initialize(piper_config_);  // throws on failure

        /* ------------------------------------------------------------------ */
        /*  Load the voice                                                    */
        /* ------------------------------------------------------------------ */
        try {
            std::optional<long> speakerId;   // leave empty for single-speaker models

            /* loadVoice is now void and has a 6-th parameter (useCuda)      */
            piper::loadVoice(piper_config_,
                             model_path,
                             config_path,
                             voice_,          // out-param
                             speakerId,
                             /* useCuda = */ false);

            std::cout << "Piper voice initialized successfully" << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Error initializing Piper TTS: " << e.what() << std::endl;
            throw;
        }

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        std::cout << "Voice loaded in " << elapsed << " seconds" << std::endl;

        /* ------------------------------------------------------------------ */
        /*  Run startup test to verify TTS functionality                      */
        /* ------------------------------------------------------------------ */
        if (!run_piper_startup_test(piper_config_, voice_)) {
            std::cerr << "Startup test failed. Exiting..." << std::endl;
            throw std::runtime_error("Piper TTS startup test failed");
        }
    }

    ~PiperTTSService() override {
        std::cout << "Shutting down Piper" << std::endl;
        piper::terminate(piper_config_);   // Frees ORT sessions & eSpeak
    }

    /* ---------------------------------------------------------------------- */
    /*  Unary synthesis returning a stream                                    */
    /* ---------------------------------------------------------------------- */
    Status SynthesizeText(ServerContext* /*context*/,
                          const voice::TextRequest* request,
                          ServerWriter<voice::AudioResponse>* writer) override {

        std::cout << "SynthesizeText: " << request->text() << std::endl;

        try {
            /* tune synthesis parameters */
            voice_.synthesisConfig.lengthScale = request->speaking_rate();
            voice_.synthesisConfig.noiseScale  = 0.667f;
            voice_.synthesisConfig.noiseW      = 0.8f;

            std::vector<int16_t> audio_samples;
            piper::SynthesisResult synth;

            auto start = std::chrono::high_resolution_clock::now();

            /* NEW API ----------------------------------------------------- */
            piper::textToAudio(piper_config_,
                               voice_,
                               request->text(),
                               audio_samples,
                               synth,
                               nullptr);            // no streaming callback

            auto end   = std::chrono::high_resolution_clock::now();
            double sec = std::chrono::duration<double>(end - start).count();

            std::cout << "Generated " << audio_samples.size()
                      << " samples in " << sec << " s (RTF "
                      << sec / (audio_samples.size() / 22050.0) << ")\n";

            /* send in ~4 kB pages */
            constexpr size_t page_size   = 4096;
            const size_t     total_bytes = audio_samples.size() * sizeof(int16_t);
            const uint8_t*   data        =
                reinterpret_cast<const uint8_t*>(audio_samples.data());

            for (size_t offset = 0; offset < total_bytes; offset += page_size) {
                size_t bytes = std::min(page_size, total_bytes - offset);

                voice::AudioResponse response;
                response.set_audio_chunk(
                    reinterpret_cast<const char*>(data + offset), bytes);
                response.set_sample_rate(22050);
                response.set_is_end(offset + bytes == total_bytes);

                if (!writer->Write(response))
                    return Status::CANCELLED;  // client hung up
            }

            return Status::OK;
        }
        catch (const std::exception& e) {
            std::cerr << "Error synthesizing speech: " << e.what() << std::endl;
            return Status(grpc::StatusCode::INTERNAL, "TTS synthesis failed");
        }
    }

    /* ---------------------------------------------------------------------- */
    /*  Streaming synthesis                                                   */
    /* ---------------------------------------------------------------------- */
    Status SynthesizeStreamingText(
        ServerContext* /*context*/,
        ServerReaderWriter<voice::AudioResponse, voice::TextRequest>* stream) override {

        voice::TextRequest request;

        while (stream->Read(&request)) {
            std::cout << "Streaming text chunk: \"" << request.text() << "\""
                      << std::endl;

            try {
                voice_.synthesisConfig.lengthScale = request.speaking_rate();
                voice_.synthesisConfig.noiseScale  = 0.667f;
                voice_.synthesisConfig.noiseW      = 0.8f;

                std::vector<int16_t> audio_samples;
                piper::SynthesisResult synth;

                piper::textToAudio(piper_config_,
                                   voice_,
                                   request.text(),
                                   audio_samples,
                                   synth,
                                   nullptr);

                /* send in ~4 kB pages */
                constexpr size_t page_size   = 4096;
                const size_t     total_bytes = audio_samples.size() * sizeof(int16_t);
                const uint8_t*   data        =
                    reinterpret_cast<const uint8_t*>(audio_samples.data());

                for (size_t offset = 0; offset < total_bytes; offset += page_size) {
                    size_t bytes = std::min(page_size, total_bytes - offset);

                    voice::AudioResponse response;
                    response.set_audio_chunk(
                        reinterpret_cast<const char*>(data + offset), bytes);
                    response.set_sample_rate(22050);
                    response.set_is_end(offset + bytes == total_bytes);

                    if (!stream->Write(response))
                        return Status::CANCELLED;  // client hung up
                }
            }
            catch (const std::exception& e) {
                std::cerr << "Streaming TTS error: " << e.what() << std::endl;
                return Status(grpc::StatusCode::INTERNAL,
                              "TTS streaming synthesis failed");
            }
        }
        return Status::OK;
    }

private:
    piper::PiperConfig piper_config_;
    piper::Voice       voice_;   // holds model + runtime state
};

/* ========================================================================== */
/*  Startup Test for Piper TTS                                                */
/* ========================================================================== */
bool run_piper_startup_test(piper::PiperConfig& piper_config, piper::Voice& voice) {
    std::cout << "Running Piper TTS startup test..." << std::endl;
    
    const std::string test_text = "This is a test of the Piper text-to-speech system.";
    std::cout << "Startup test prompt: \"" << test_text << "\"" << std::endl;
    
    try {
        // Keep default synthesis parameters for the test
        std::vector<int16_t> audio_samples;
        piper::SynthesisResult synth;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Attempt to synthesize speech
        piper::textToAudio(piper_config,
                          voice,
                          test_text,
                          audio_samples,
                          synth,
                          nullptr);  // no streaming callback
                          
        auto end = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(end - start).count();
        
        // Verify we got samples
        if (audio_samples.empty()) {
            std::cerr << "Startup test failed: No audio samples generated" << std::endl;
            return false;
        }
        
        std::cout << "Startup test generated " << audio_samples.size()
                  << " samples in " << sec << " s (RTF "
                  << sec / (audio_samples.size() / 22050.0) << ")" << std::endl;
        
        // Optional: Save samples to a file for debugging (wav header would be needed for proper playback)
        // Uncomment if needed for debugging
        /*
        std::ofstream test_file("/tmp/piper_test.raw", std::ios::binary);
        if (test_file) {
            test_file.write(reinterpret_cast<const char*>(audio_samples.data()), 
                            audio_samples.size() * sizeof(int16_t));
            std::cout << "Saved test audio to /tmp/piper_test.raw" << std::endl;
        }
        */
        
        std::cout << "Piper TTS startup test completed successfully" << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Startup test failed with exception: " << e.what() << std::endl;
        return false;
    }
    catch (...) {
        std::cerr << "Startup test failed with unknown exception" << std::endl;
        return false;
    }
}

/* ========================================================================== */
/*  main                                                                      */
/* ========================================================================== */
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx> <config.json>\n";
        return 1;
    }

    std::string model_path  = argv[1];
    std::string config_path = argv[2];
    std::string socket_path = "/app/piper-sockets/piper.sock";

    if (access(socket_path.c_str(), F_OK) == 0) {
        std::cout << "Removing existing socket: " << socket_path << std::endl;
        if (unlink(socket_path.c_str()) != 0) {
            std::cerr << "Failed to remove socket: " << strerror(errno) << '\n';
            return 1;
        }
    }

    try {
        // Create sockets directory if it doesn't exist
        mkdir("/app/piper-sockets", 0777);
        
        PiperTTSService service(model_path, config_path);

        ServerBuilder builder;
        builder.AddListeningPort("unix://" + socket_path,
                                 grpc::InsecureServerCredentials());
        builder.RegisterService(&service);

        std::unique_ptr<Server> server(builder.BuildAndStart());
        std::cout << "gRPC server listening on unix://" << socket_path
                  << std::endl;

        if (chmod(socket_path.c_str(), 0777) != 0) {
            std::cerr << "Warning: Failed to change socket permissions: "
                      << strerror(errno) << '\n';
        }
        
        // Create a ready file to signal that the server is ready
        std::ofstream("/app/piper-sockets/ready").close();
        std::cout << "Created ready file: /app/piper-sockets/ready" << std::endl;

        server->Wait();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}