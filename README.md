# Image Streaming API with YOLO

An object detection system using **YOLOv26n** for benchmarking WebSocket-based image streaming latency between Python and Go server implementations.

![bbox](/assets/bbox.gif)

This repository was developed with assistance from Claude Code and Gemini.

## Overview

This project compares the end-to-end latency of real-time image streaming using WebSockets between:

- **Python server** — built with FastAPI and websockets
- **Go server** — built with the standard `net/http` and `gorilla/websocket`

Both servers run the same YOLO inference pipeline and provide an identical WebSocket interface.

![overview](/assets/flow.png)

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

### 3-1. Run the Test

```sh
$ cd py_server
$ uv run utils/test.py \
    --video-in "assets/test.mp4" \
    --bench-out "results/go_result.csv" \
    --server-type "go" \
    --clients 6
```

`--server-type` must be set to `py` or `go`.

### 3-2. Run the Client

```sh
$ cd client
$ flutter pub get
$ flutter run
```

## Test Results

- OS: macOS 26.2
- CPU: Apple M4
- Python: 3.12.12
- Go: 1.22.8

### Latency

Scenario: 20 concurrent clients

| Metric          | Go Server     | Python Server |
| --------------- | ------------- | ------------- |
| Average Latency | **413.09 ms** | 437.10 ms     |
| Average FPS     | **2.53 ms**   | 2.35 ms       |
| P95 Latency     | **542.26 ms** | 549.43 ms     |
| P99 Latency     | **613.48 ms** | 623.11 ms     |

### CPU Usage

Scenario: 6 concurrent clients

![CPU Usage](/assets/cpu-usage.png)
