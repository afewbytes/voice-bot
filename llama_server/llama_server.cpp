#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <grpcpp/grpcpp.h>

#include "voice.grpc.pb.h"
#include "llama.h"      // adjust path as needed

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using grpc::ServerWriter;

using voice::LlamaService;
using voice::GenerateRequest;
using voice::GenerateResponse;

class LlamaServiceImpl final : public LlamaService::Service {
public:
  LlamaServiceImpl(const std::string &model_path) {
    // 1) initialize backend
    llama_backend_init();

    // 2) load the model
    llama_model_params mparams = llama_model_default_params();
    model_ = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model_) {
      std::cerr << "Failed to load LLaMA model: " << model_path << "\n";
      std::exit(1);
    }

    // Grab the vocab for tokenization and piece conversion
    vocab_ = llama_model_get_vocab(model_);

    // 3) create context
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 1024;
    cparams.n_batch = 8;
    context_ = llama_init_from_model(model_, cparams);
    if (!context_) {
      std::cerr << "Failed to create llama_context\n";
      llama_model_free(model_);
      std::exit(1);
    }
  }

  ~LlamaServiceImpl() {
    if (context_)   llama_free(context_);
    if (model_)     llama_model_free(model_);
    llama_backend_free();
  }

  Status Generate(ServerContext* /*ctx*/,
                  const GenerateRequest* req,
                  ServerWriter<GenerateResponse>* writer) override {
    const std::string prompt = req->prompt();
    const int max_tokens     = req->max_tokens();
    const float temp         = req->temperature();
    const float top_p        = req->top_p();

    // --- tokenize prompt on the vocab ---
    int n_tokens = llama_tokenize(vocab_,
                                  prompt.c_str(),
                                  (int)prompt.size(),
                                  nullptr, 0,
                                  /*add_bos=*/true,
                                  /*special=*/false);
    if (n_tokens < 0) {
      std::cerr << "tokenize length query\n";
      return Status(grpc::StatusCode::INTERNAL, "tokenization failure");
    }
    std::vector<llama_token> tokens(n_tokens);
    llama_tokenize(vocab_,
                   prompt.c_str(),
                   (int)prompt.size(),
                   tokens.data(), n_tokens,
                   /*add_bos=*/true,
                   /*special=*/false);

    // --- evaluate the prompt once ---
    llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
    if (llama_decode(context_, batch) != 0) {
      return Status(grpc::StatusCode::INTERNAL, "Failed to evaluate prompt");
    }

    // --- set up a chain sampler ---
    llama_sampler_chain_params schain = llama_sampler_chain_default_params();
    schain.no_perf = false;
    llama_sampler *sampler = llama_sampler_chain_init(schain);                          //  [oai_citation_attribution:0‡Fossies](https://fossies.org/linux/llama.cpp/examples/simple/simple.cpp)
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, /*min_keep=*/1));   //  [oai_citation_attribution:1‡GitHub](https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h)
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(temp));                    //  [oai_citation_attribution:2‡GitHub](https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h)

    // --- streaming generation ---
    GenerateResponse resp;
    for (int i = 0; i < max_tokens; ++i) {
      llama_token token_id = llama_sampler_sample(sampler, context_, /*idx=*/0);
      if (llama_vocab_is_eog(vocab_, token_id)) {
        break;
      }

      // piece → string
      char buf[8];
      int pcsz = llama_token_to_piece(vocab_, token_id, buf, sizeof(buf), /*pad=*/0, /*special=*/true);
      std::string piece = (pcsz > 0 ? std::string(buf, pcsz) : "");

      // send back
      resp.set_text(piece);
      resp.set_done(false);
      writer->Write(resp);

      // feed token for next step
      llama_batch next = llama_batch_get_one(&token_id, 1);
      if (llama_decode(context_, next) != 0) {
        llama_sampler_free(sampler);
        return Status(grpc::StatusCode::INTERNAL, "Failed to decode token");
      }
    }

    // final EOS marker
    resp.set_text("");
    resp.set_done(true);
    writer->Write(resp);

    llama_sampler_free(sampler);
    return Status::OK;
  }

private:
  llama_model*   model_{nullptr};
  llama_context* context_{nullptr};
  const llama_vocab* vocab_{nullptr};
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

  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  std::cout << "Llama server listening on " << server_address << "\n";
  server->Wait();
  return 0;
}