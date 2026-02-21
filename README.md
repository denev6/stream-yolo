# Image Streaming API: Python vs Go

An object detection system using **YOLOv26n** for benchmarking WebSocket-based image streaming latency between Python and Go server implementations.

![bbox](/assets/bbox.gif)

This repository was developed with assistance from Claude Code and Gemini.

## Overview

This project compares the end-to-end latency of real-time image streaming using WebSockets between:

- **Python server** — built with FastAPI and websockets
- **Go server** — built with the standard `net/http` and `gorilla/websocket`

Both servers run the same YOLO inference pipeline and provide an identical WebSocket interface.

## Getting Started

### 1. Download Model Weights

```sh
$ cd py_server
$ uv run utils/get_model.py
```

### 2. Start the Servers

```sh
$ docker compose build
$ docker compose up
```

- Go server: `http://localhost:8001`
- Python server: `http://localhost:8000`

### 3. Run the Client

```sh
$ cd py_server
$ uv run client.py \
    --video-in "assets/test.mp4" \
    --bench-out "results/go_result.csv" \
    --server-type "go" \
    --clients 6
```

`--server-type` must be set to `py` or `go`.

## Results

- OS: Windows 11
- CPU: AMD Ryzen 5 7500F
- Python: 3.12.12
- Go: 1.22.8

### Single Client Benchmark

Measurements are averaged across **three runs** with a single client connection.

| Metric          | Go Server    | Python Server |
| --------------- | ------------ | ------------- |
| Average Latency | **30.51 ms** | 35.13 ms      |
| Average FPS     | **33.47**    | 28.71         |
| P95 Latency     | **41.18 ms** | 41.42 ms      |
| P99 Latency     | **50.39 ms** | 54.60 ms      |

### Multi-Client Benchmark (6 concurrent clients)

| Metric          | Go Server     | Python Server |
| --------------- | ------------- | ------------- |
| Average Latency | **117.87 ms** | 122.74 ms     |
| Average FPS     | **8.83**      | 8.41          |
| P95 Latency     | **155.04 ms** | 162.41 ms     |
| P99 Latency     | 190.62 ms     | **186.82 ms** |
