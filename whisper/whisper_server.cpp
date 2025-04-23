#include <iostream>
#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>
#include "whisper.grpc.pb.h"
#include <sys/stat.h>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using grpc::ServerReader;
using whisper::WhisperService;
using whisper::AudioChunk;
using whisper::Transcription;

class WhisperServiceImpl final : public WhisperService::Service {
	Status StreamAudio(ServerContext* context, ServerReader<AudioChunk>* reader, Transcription* response) override {
		AudioChunk chunk;
		std::string result = "";
		while (reader->Read(&chunk)) {
			// TODO: integrate whisper.cpp streaming API here
			result += "word "; // placeholder transcription
		}
		response->set_text(result);
		return Status::OK;
	}
};

void RunServer() {
	std::string socket_path = "unix:/app/sockets/whisper.sock";

    mkdir("/app/sockets", 0777);
    
	WhisperServiceImpl service;

	ServerBuilder builder;
	builder.AddListeningPort(socket_path, grpc::InsecureServerCredentials());
	builder.RegisterService(&service);
	std::unique_ptr<Server> server(builder.BuildAndStart());
	std::cout << "Server listening on " << socket_path << std::endl;
	server->Wait();
}

int main(int argc, char** argv) {
	RunServer();
	return 0;
}
