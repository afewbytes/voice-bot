#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <cmath>

#include <grpcpp/grpcpp.h>
#include "voice.grpc.pb.h"
#include "llama.h"  // adjust include path as needed

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using grpc::ServerWriter;

using voice::LlamaService;
using voice::GenerateRequest;
using voice::GenerateResponse;

bool run_startup_test(llama_model* model, llama_context* context, const llama_vocab* vocab) {
    std::cout << "Running startup test..." << std::endl;
  
    const std::string test_prompt = "Hello, my name is";
    std::cout << "Startup test prompt: \"" << test_prompt << "\"" << std::endl;
  
    int n_tokens = -llama_tokenize(vocab, test_prompt.c_str(), (int)test_prompt.size(), nullptr, 0, true, true);
    if (n_tokens <= 0) {
      std::cerr << "Startup test failed: Tokenization error" << std::endl;
      return false;
    }
  
    std::vector<llama_token> tokens(n_tokens);
    if (llama_tokenize(vocab, test_prompt.c_str(), (int)test_prompt.size(), tokens.data(), n_tokens, true, true) < 0) {
      std::cerr << "Startup test failed: Tokenization error" << std::endl;
      return false;
    }
  
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(context, batch) != 0) {
      std::cerr << "Startup test failed: Prompt evaluation error" << std::endl;
      return false;
    }
  
    // Check logits before sampling
    const float* logits = llama_get_logits(context);
    int vocab_size = llama_vocab_n_tokens(vocab);
    bool all_invalid = true;
    for (int i = 0; i < vocab_size; ++i) {
      if (!std::isnan(logits[i]) && logits[i] > -INFINITY) {
        all_invalid = false;
        break;
      }
    }
  
    if (all_invalid) {
      std::cerr << "Startup test failed: All logits are NaN or invalid" << std::endl;
      return false;
    }
  
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler *sampler = llama_sampler_chain_init(sparams);
  
    uint32_t seed = LLAMA_DEFAULT_SEED;

    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));       // keep 40 best
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.95f, 1)); // keep top-95 %
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7f));      // soften probs
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed));      // *selects* one
  
    llama_token token_id = llama_sampler_sample(sampler, context, -1);
  
    if (token_id < 0 || llama_vocab_is_eog(vocab, token_id)) {
      std::cerr << "Startup test failed: Could not generate a valid token (token_id = " << token_id << ")" << std::endl;
      llama_sampler_free(sampler);
      return false;
    }
  
    char buf[16];
    int pcsz = llama_token_to_piece(vocab, token_id, buf, sizeof(buf), 0, true);
    if (pcsz <= 0) {
      std::cerr << "Startup test failed: Could not convert token to piece" << std::endl;
      llama_sampler_free(sampler);
      return false;
    }
  
    std::string first_piece(buf, pcsz);
    std::cout << "First token generated: \"" << first_piece << "\"" << std::endl;
  
    std::string full_output = first_piece;
  
    // Do NOT call llama_decode(context, batch) again – instead feed new token
    batch = llama_batch_get_one(&token_id, 1);
    if (llama_decode(context, batch) != 0) {
      std::cerr << "Startup test failed: Decoding error after first token" << std::endl;
      llama_sampler_free(sampler);
      return false;
    }
  
    const int more_tokens = 2;
    std::cout << "Generating " << more_tokens << " more tokens: ";
    bool generation_complete = false;
  
    for (int i = 0; i < more_tokens && !generation_complete; ++i) {
      token_id = llama_sampler_sample(sampler, context, -1);
  
      if (token_id < 0 || llama_vocab_is_eog(vocab, token_id)) {
        std::cout << "[end]";
        generation_complete = true;
        continue;
      }
  
      pcsz = llama_token_to_piece(vocab, token_id, buf, sizeof(buf), 0, true);
      if (pcsz <= 0) {
        std::cout << "[invalid]";
        continue;
      }
  
      std::string piece(buf, pcsz);
      std::cout << piece;
      full_output += piece;
  
      batch = llama_batch_get_one(&token_id, 1);
      if (llama_decode(context, batch) != 0) {
        std::cout << "[error]";
        break;
      }
    }
  
    std::cout << std::endl;
    std::cout << "Startup test completed successfully" << std::endl;
  
    llama_sampler_free(sampler);
    return true;
  }

class LlamaServiceImpl final : public LlamaService::Service {
public:
  explicit LlamaServiceImpl(const std::string &model_path) {
    ggml_backend_load_all();

    llama_model_params mparams = llama_model_default_params();
    model_ = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model_) {
      std::cerr << "Failed to load LLaMA model: " << model_path << "\n";
      std::exit(1);
    }

    cparams_ = llama_context_default_params();
    cparams_.n_ctx   = 1024;
    cparams_.n_batch = 8;
    context_ = llama_init_from_model(model_, cparams_);
    if (!context_) {
      std::cerr << "Failed to create llama_context\n";
      llama_model_free(model_);
      std::exit(1);
    }

    vocab_ = llama_model_get_vocab(model_);
    if (!vocab_) {
      std::cerr << "Failed to get vocab from model\n";
      llama_free(context_);
      llama_model_free(model_);
      std::exit(1);
    }

    if (!run_startup_test(model_, context_, vocab_)) {
      std::cerr << "Startup test failed. Exiting...\n";
      llama_free(context_);
      llama_model_free(model_);
      std::exit(1);
    }
  }

  ~LlamaServiceImpl() override {
    if (context_)   llama_free(context_);
    if (model_)     llama_model_free(model_);
  }

  Status Generate(ServerContext* /*ctx*/, const GenerateRequest* req,
                  ServerWriter<GenerateResponse>* writer) override {
    llama_free(context_);
    context_ = llama_init_from_model(model_, cparams_);
    if (!context_) {
      return Status(grpc::StatusCode::INTERNAL, "Failed to reset llama context");
    }

    const std::string prompt = req->prompt();
    const int max_tokens = req->max_tokens();
    const float temp   = req->temperature();
    const float top_p  = req->top_p();

    int n_tokens = -llama_tokenize(
        vocab_, prompt.c_str(), (int)prompt.size(),
        nullptr, 0,
        true, true
    );
    if (n_tokens <= 0) {
      return Status(grpc::StatusCode::INTERNAL, "tokenization failure");
    }

    std::vector<llama_token> tokens(n_tokens);
    if (llama_tokenize(
            vocab_, prompt.c_str(), (int)prompt.size(),
            tokens.data(), n_tokens,
            true, true
        ) < 0) {
      return Status(grpc::StatusCode::INTERNAL, "tokenization failure");
    }

    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(context_, batch) != 0) {
      return Status(grpc::StatusCode::INTERNAL, "prompt evaluation failed");
    }

    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler *sampler = llama_sampler_chain_init(sparams);

    uint32_t seed = LLAMA_DEFAULT_SEED;

    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));       // keep 40 best
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.95f, 1)); // keep top-95 %
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7f));      // soften probs
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed));      // *selects* one

    GenerateResponse resp;
    for (int i = 0; i < max_tokens; ++i) {
      llama_token token_id = llama_sampler_sample(sampler, context_, -1);
      if (llama_vocab_is_eog(vocab_, token_id)) {
        break;
      }
      char buf[8];
      int pcsz = llama_token_to_piece(vocab_, token_id, buf, sizeof(buf), 0, true);
      std::string piece = (pcsz > 0 ? std::string(buf, pcsz) : "");

      resp.set_text(piece);
      resp.set_done(false);
      writer->Write(resp);

      batch = llama_batch_get_one(&token_id, 1);
      if (llama_decode(context_, batch) != 0) {
        llama_sampler_free(sampler);
        return Status(grpc::StatusCode::INTERNAL, "decoding failure");
      }
    }

    resp.set_text("");
    resp.set_done(true);
    writer->Write(resp);

    llama_sampler_free(sampler);
    return Status::OK;
  }

private:
  llama_model*    model_{nullptr};
  llama_context*  context_{nullptr};
  const llama_vocab* vocab_{nullptr};
  llama_context_params cparams_;
};

int main(int argc, char** argv) {
  const std::string server_address("unix:///app/llama-sockets/llama.sock");
  std::string model_path = "/app/models/llama-2-7b.Q4_K_M.gguf";
  if (argc > 1) {
    model_path = argv[1];
  }

  LlamaServiceImpl service(model_path);

  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);

  std::unique_ptr<Server> server(builder.BuildAndStart());
  std::cout << "Llama server listening on " << server_address << "\n";
  server->Wait();
  return 0;
}
