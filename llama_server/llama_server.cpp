#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <cmath>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/resource.h>
#include <cerrno>
#include <cstring>

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

// Forward declarations
class LlamaContextPool;

// Guard for automatic context release
class ContextGuard {
public:
  ContextGuard(llama_context* ctx, LlamaContextPool* pool);
  ~ContextGuard();
  llama_context* get() { return ctx_; }

private:
  llama_context* ctx_;
  LlamaContextPool* pool_;
};

using ContextGuardPtr = std::unique_ptr<ContextGuard>;

// Context pool for handling multiple concurrent requests
class LlamaContextPool {
public:
  LlamaContextPool(llama_model* model, size_t count);
  ~LlamaContextPool();
  
  ContextGuardPtr acquire();
  void release(llama_context* ctx);
  
private:
  llama_model* model_;  // Shared model reference, NOT owned by the pool
  std::queue<llama_context*> pool_;
  std::mutex mux_;
  std::condition_variable cond_;
  llama_context_params cparams_;
};

// ContextGuard implementation
ContextGuard::ContextGuard(llama_context* ctx, LlamaContextPool* pool)
  : ctx_(ctx), pool_(pool) {}

ContextGuard::~ContextGuard() { 
  pool_->release(ctx_); 
}

// LlamaContextPool implementation
LlamaContextPool::LlamaContextPool(llama_model* model, size_t count) : model_(model) {
  cparams_ = llama_context_default_params();
  cparams_.n_ctx   = 1024;
  cparams_.n_batch = 8;
  
  std::cout << "Initializing pool with " << count << " contexts (sharing the same model)..." << std::endl;
  for (size_t i = 0; i < count; ++i) {
    auto* ctx = llama_init_from_model(model_, cparams_);
    if (!ctx) {
      std::cerr << "Failed to create context " << i << std::endl;
      exit(1);
    }
    pool_.push(ctx);
    std::cout << "Context " << i+1 << "/" << count << " initialized" << std::endl;
  }
  std::cout << "Successfully initialized context pool" << std::endl;
}

LlamaContextPool::~LlamaContextPool() {
  while (!pool_.empty()) {
    llama_free(pool_.front());
    pool_.pop();
  }
}

ContextGuardPtr LlamaContextPool::acquire() {
  std::unique_lock<std::mutex> lk(mux_);
  cond_.wait(lk, [&]{ return !pool_.empty(); });
  auto* ctx = pool_.front(); 
  pool_.pop();
  return std::make_unique<ContextGuard>(ctx, this);
}

void LlamaContextPool::release(llama_context* ctx) {
  std::lock_guard<std::mutex> lk(mux_);
  pool_.push(ctx);
  cond_.notify_one();
}

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
    std::cout << "Loading model: " << model_path << std::endl;
    ggml_backend_load_all();

    // Load the model only once
    llama_model_params mparams = llama_model_default_params();
    model_ = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model_) {
      std::cerr << "Failed to load LLaMA model: " << model_path << "\n";
      std::exit(1);
    }
    std::cout << "Model loaded successfully" << std::endl;

    // Get vocab from the model (shared resource)
    vocab_ = llama_model_get_vocab(model_);
    if (!vocab_) {
      std::cerr << "Failed to get vocab from model\n";
      llama_model_free(model_);
      std::exit(1);
    }

    // Create a single context for startup test
    std::cout << "Creating test context for startup validation" << std::endl;
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 1024;
    cparams.n_batch = 8;
    auto* test_context = llama_init_from_model(model_, cparams);
    if (!test_context) {
      std::cerr << "Failed to create test context\n";
      llama_model_free(model_);
      std::exit(1);
    }

    // Run startup test
    if (!run_startup_test(model_, test_context, vocab_)) {
      std::cerr << "Startup test failed. Exiting...\n";
      llama_free(test_context);
      llama_model_free(model_);
      std::exit(1);
    }
    
    // Free the test context
    llama_free(test_context);
    
    // Create the context pool
    // Each context shares the same model, which is loaded only once
    size_t num_contexts = std::max(1, static_cast<int>(2));
    std::cout << "Creating context pool with " << num_contexts << " contexts (sharing single model)" << std::endl;
    context_pool_ = std::make_unique<LlamaContextPool>(model_, num_contexts);
  }

  ~LlamaServiceImpl() override {
    context_pool_.reset(); // Free all contexts first
    if (model_) llama_model_free(model_);
  }

  Status Generate(ServerContext* /*ctx*/, const GenerateRequest* req,
                  ServerWriter<GenerateResponse>* writer) override {
    static int session_counter = 0;
    int session_id = ++session_counter;
    
    std::cout << "[INFO] Session " << session_id << ": New generation request" << std::endl;
    
    // Acquire a context from the pool for this request
    auto guard = context_pool_->acquire();
    auto* context = guard->get();
    
    const std::string prompt = req->prompt();
    const int max_tokens = req->max_tokens();
    const float temp = req->temperature();
    const float top_p = req->top_p();

    std::cout << "[INFO] Session " << session_id 
              << ": Processing request with prompt length " << prompt.size() 
              << ", max_tokens=" << max_tokens 
              << ", temp=" << temp 
              << ", top_p=" << top_p << std::endl;

    int n_tokens = -llama_tokenize(
        vocab_, prompt.c_str(), (int)prompt.size(),
        nullptr, 0,
        true, true
    );
    if (n_tokens <= 0) {
      std::cout << "[ERROR] Session " << session_id << ": Tokenization failure" << std::endl;
      return Status(grpc::StatusCode::INTERNAL, "tokenization failure");
    }

    std::vector<llama_token> tokens(n_tokens);
    if (llama_tokenize(
            vocab_, prompt.c_str(), (int)prompt.size(),
            tokens.data(), n_tokens,
            true, true
        ) < 0) {
      std::cout << "[ERROR] Session " << session_id << ": Tokenization failure" << std::endl;
      return Status(grpc::StatusCode::INTERNAL, "tokenization failure");
    }

    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(context, batch) != 0) {
      std::cout << "[ERROR] Session " << session_id << ": Prompt evaluation failed" << std::endl;
      return Status(grpc::StatusCode::INTERNAL, "prompt evaluation failed");
    }

    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler *sampler = llama_sampler_chain_init(sparams);

    uint32_t seed = LLAMA_DEFAULT_SEED;

    // Use the provided temperature and top_p if they're set
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));       // keep 40 best
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p > 0 ? top_p : 0.95f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(temp > 0 ? temp : 0.7f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed));      // *selects* one

    GenerateResponse resp;
    std::cout << "[INFO] Session " << session_id << ": Starting token generation" << std::endl;
    
    // Simple flag to check if we should stop after finding a period
    bool foundSentenceEnd = false;
    
    for (int i = 0; i < max_tokens && !foundSentenceEnd; ++i) {
      llama_token token_id = llama_sampler_sample(sampler, context, -1);
      if (llama_vocab_is_eog(vocab_, token_id)) {
        std::cout << "[INFO] Session " << session_id << ": Reached EOG token at position " << i << std::endl;
        break;
      }
      
      char buf[8];
      int pcsz = llama_token_to_piece(vocab_, token_id, buf, sizeof(buf), 0, true);
      std::string piece = (pcsz > 0 ? std::string(buf, pcsz) : "");

      // Check if this piece contains a sentence-ending character
      if (piece.find('.') != std::string::npos || 
          piece.find('!') != std::string::npos || 
          piece.find('?') != std::string::npos) {
        // We found a sentence ending, will stop after sending this token
        foundSentenceEnd = true;
        std::cout << "[INFO] Session " << session_id 
                  << ": Found sentence ending at position " << i 
                  << ", will stop generation" << std::endl;
      }

      resp.set_text(piece);
      resp.set_done(false);
      writer->Write(resp);

      batch = llama_batch_get_one(&token_id, 1);
      if (llama_decode(context, batch) != 0) {
        std::cout << "[ERROR] Session " << session_id << ": Decoding failure at position " << i << std::endl;
        llama_sampler_free(sampler);
        return Status(grpc::StatusCode::INTERNAL, "decoding failure");
      }
    }

    resp.set_text("");
    resp.set_done(true);
    writer->Write(resp);
    
    std::cout << "[INFO] Session " << session_id << ": Generation completed successfully" << std::endl;
    llama_sampler_free(sampler);
    return Status::OK;
  }

private:
  llama_model* model_{nullptr};
  const llama_vocab* vocab_{nullptr};
  std::unique_ptr<LlamaContextPool> context_pool_;
};

int main(int argc, char** argv) {
  const std::string server_address("unix:///app/llama-sockets/llama.sock");
  std::string model_path = "/app/models/llama-2-7b.Q4_K_M.gguf";
  if (argc > 1) {
    model_path = argv[1];
  }

  std::cout << "Initializing LlamaService with model: " << model_path << std::endl;
  
  try {
    LlamaServiceImpl service(model_path);
    
    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    
    // Create ready file directory
    mkdir("/app/llama-sockets", 0777);
    
    // Remove old socket if it exists
    unlink(server_address.substr(5).c_str());  // Remove "unix:" prefix

    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "Llama server listening on " << server_address << std::endl;
    
    // Create a ready file to signal the server is ready
    std::ofstream("/app/llama-sockets/ready").close();
    
    server->Wait();
  } catch (const std::exception& e) {
    std::cerr << "Exception in server: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Unknown exception in server" << std::endl;
    return 1;
  }
  
  return 0;
}