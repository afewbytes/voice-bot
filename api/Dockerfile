FROM golang:1.24

# Install required tools
RUN apt-get update && apt-get install -y protobuf-compiler

# Install protoc-gen-go and protoc-gen-go-grpc
RUN go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.31 \
    && go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.3

# Add Go binaries to PATH
ENV PATH="/go/bin:$PATH"

# Set working directory
WORKDIR /app

# Initialize Go module
RUN go mod init voice-bot

# Copy API source code and proto files
COPY api/ .
COPY proto/ /app/proto/

# Generate Go gRPC code from proto
RUN protoc --go_out=. --go-grpc_out=. --proto_path=/app/proto /app/proto/whisper.proto

# Build the Go app
RUN go mod tidy
RUN go build -o server .

CMD ["./server"]