# Image Streaming API: Python vs Go

An object detection API server using **YOLOv26n**, designed to benchmark WebSocket-based image streaming latency between Python and Go implementations.

This repository was written with [Claude Code](https://claude.ai/code).

## Overview

This project compares the end-to-end latency of real-time image streaming over WebSocket between:

- **Python server** - built with FastAPI / websockets
- **Go server** - built with the standard `net/http` + `gorilla/websocket`

Both servers run the same YOLO inference pipeline and expose an identical WebSocket interface.

## Getting Started

### 1. Start the Server

#### Python

```sh
docker compose up py-server
```

#### Go

```sh
docker compose up go-server
```

### 2. Run the Client

Edit `VIDEO_IN` in `py_server/client.py` to point to your input video, then:

```sh
cd py_server
uv run client.py
```

Results are saved to the paths defined by `VIDEO_OUT` (annotated video) and `BENCH_OUT` (latency log).

### 3. Deploy to Google Cloud Run

```sh
docker tag yolo-stream gcr.io/YOUR_PROJECT/yolo-stream
docker push gcr.io/YOUR_PROJECT/yolo-stream

gcloud run deploy yolo-stream \
  --image gcr.io/YOUR_PROJECT/yolo-stream \
  --platform managed \
  --region asia-northeast3 \
  --allow-unauthenticated
```
