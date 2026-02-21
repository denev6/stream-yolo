import ast
import asyncio
from contextlib import asynccontextmanager

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

MODEL_PATH = "model/yolo26n.onnx"
INPUT_SIZE = 640
CONF_THRESHOLD = 0.4

session: ort.InferenceSession | None = None
CLASS_NAMES: dict[int, str] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global session, CLASS_NAMES
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

    # ultralytics ONNX 메타데이터에서 클래스 이름 추출
    # custom_metadata_map["names"] 예시: "{0: 'person', 1: 'bicycle', ...}"
    meta = session.get_modelmeta().custom_metadata_map
    if "names" in meta:
        CLASS_NAMES = ast.literal_eval(meta["names"])

    print(f"[YOLO] 모델 로드 완료: {MODEL_PATH} ({len(CLASS_NAMES)}개 클래스)")
    yield
    session = None


app = FastAPI(title="YOLO Stream Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 전처리 ──────────────────────────────────────────────────────────────────
def preprocess(frame: np.ndarray) -> tuple[np.ndarray, float, float]:
    orig_h, orig_w = frame.shape[:2]
    resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    inp = resized.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0
    scale_x = orig_w / INPUT_SIZE
    scale_y = orig_h / INPUT_SIZE
    return inp, scale_x, scale_y


# ── 후처리 ──────────────────────────────────────────────────────────────────
def postprocess(raw: np.ndarray, scale_x: float, scale_y: float) -> list[dict]:
    """
    YOLO26 출력 형태: (N, 6) — [x1, y1, x2, y2, score, label]
    배치 차원이 있으면 (1, N, 6)이므로 squeeze 처리.
    """
    if raw is None or raw.size == 0:
        return []

    dets = raw.squeeze()  # (1,N,6) → (N,6) or (N,6) 그대로
    if dets.ndim == 1:
        dets = dets[np.newaxis]  # 단일 검출 → (1,6)

    results: list[dict] = []
    for row in dets:
        if len(row) < 6:
            continue
        x1, y1, x2, y2, score, label = row[:6]
        if score < CONF_THRESHOLD:
            continue
        results.append(
            {
                "box": [
                    int(x1 * scale_x),
                    int(y1 * scale_y),
                    int(x2 * scale_x),
                    int(y2 * scale_y),
                ],
                "score": round(float(score), 4),
                "label": int(label),
                "name": CLASS_NAMES.get(int(label), f"cls{int(label)}"),
            }
        )
    return results


# ── 공통 추론 로직 ────────────────────────────────────────────────────────────
def infer(frame: np.ndarray) -> list[dict]:
    """프레임 → detections (바운딩 박스 그리기는 클라이언트 담당)"""
    inp, scale_x, scale_y = preprocess(frame)
    outputs = session.run(None, {"images": inp})
    return postprocess(np.asarray(outputs[0]), scale_x, scale_y)


# ── 헬스 체크 ────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "ok", "model_loaded": session is not None}


# ── WebSocket: 검출 결과 JSON 스트리밍 ──────────────────────────────────────
@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    """
    클라이언트 → 서버: JPEG/PNG 바이너리 프레임
    서버 → 클라이언트: JSON — {"detections": [{box, score, label, name}, ...]}
    바운딩 박스 그리기는 클라이언트가 담당한다.
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()

            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                await websocket.send_json({"error": "invalid image"})
                continue

            detections = await asyncio.get_event_loop().run_in_executor(
                None, infer, frame
            )
            await websocket.send_json({"detections": detections})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[ws/stream] 오류: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
