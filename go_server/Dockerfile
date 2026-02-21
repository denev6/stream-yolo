# Stage 1: Builder
FROM gocv/opencv:4.10.0 AS builder

ARG ORT_VERSION=1.21.0

RUN apt-get update && apt-get install -y --no-install-recommends wget \
    && wget -qO /tmp/ort.tgz "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" \
    && tar -C /opt -xf /tmp/ort.tgz \
    && mv "/opt/onnxruntime-linux-x64-${ORT_VERSION}" /opt/ort

ENV CGO_CFLAGS="-I/opt/ort/include"
ENV CGO_LDFLAGS="-L/opt/ort/lib"
ENV GOPROXY=direct

WORKDIR /build
COPY go_server/main.go ./
# 호환성을 위해 onnxruntime_go 버전을 v1.14.0으로 고정
RUN go mod init yolo-server && \
    go get github.com/yalue/onnxruntime_go@v1.14.0 && \
    (go mod tidy || go mod tidy)

RUN CGO_ENABLED=1 go build -trimpath -ldflags="-s -w" -o /out/server .

# Stage 2: Runtime
FROM gocv/opencv:4.10.0

WORKDIR /app

COPY --from=builder /out/server ./server

COPY --from=builder /opt/ort/lib/libonnxruntime.so.1.21.0 /usr/local/lib/libonnxruntime.so
RUN ldconfig

COPY model/yolo26n.onnx .

EXPOSE 8000

CMD ["./server"]