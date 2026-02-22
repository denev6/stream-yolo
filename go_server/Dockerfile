# Stage 1: Builder
FROM gocv/opencv:4.10.0 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends wget tar

RUN wget -qO /tmp/ort.tgz "https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/onnxruntime-linux-aarch64-1.21.0.tgz" && \
    tar -C /opt -xf /tmp/ort.tgz && \
    mv /opt/onnxruntime-linux-aarch64-1.21.0 /opt/ort

ENV CGO_CFLAGS="-I/opt/ort/include"
ENV CGO_LDFLAGS="-L/opt/ort/lib -L/usr/local/lib -ltbb"
ENV GOPROXY=direct

WORKDIR /build
COPY go_server/main.go .

RUN go mod init yolo-server && \
    go get github.com/yalue/onnxruntime_go@v1.14.0 && \
    go mod tidy

RUN CGO_ENABLED=1 go build -trimpath -ldflags="-s -w" -o /out/server .

# Stage 2: Runtime
FROM gocv/opencv:4.10.0

WORKDIR /app

COPY --from=builder /out/server ./server

COPY --from=builder /opt/ort/lib/libonnxruntime.so* /usr/local/lib/
COPY --from=builder /usr/local/lib/libtbb* /usr/local/lib/
COPY --from=builder /usr/local/lib/libopencv* /usr/local/lib/

ENV LD_LIBRARY_PATH=/usr/local/lib:/opt/ort/lib:$LD_LIBRARY_PATH

RUN ldconfig

EXPOSE 8001
CMD ["./server"]