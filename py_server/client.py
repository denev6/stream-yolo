"""
WebSocket 스트리밍 클라이언트

흐름:
  ../assets/test_30.mp4 (프레임 읽기)
      → ws://localhost:8000/ws/stream (JPEG bytes 전송)
      → JSON {"detections": [...]} 수신
      → 클라이언트에서 바운딩 박스 그리기
      → ../assets/test_30_result.mp4 (영상 저장)
      → ../assets/test_30_benchmark.csv (벤치마크 raw data 저장)
"""

import asyncio
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import websockets

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent
VIDEO_IN = str(_ROOT / "../assets/test_30.mp4")
VIDEO_OUT = str(_ROOT / "../assets/test_30_result.mp4")
BENCH_OUT = str(_ROOT / "../assets/test_30_benchmark.csv")
WS_URL = "ws://localhost:8000/ws/stream"

YOLO_INPUT_SIZE = 640  # YOLO 모델 입력 크기
JPEG_ENCODE_QUALITY = 90  # 전송 품질 (1~100)
MAX_WS_MSG_BYTES = 16 * 1024 * 1024  # 수신 메시지 최대 크기 16 MB
MAX_WS_TIMEOUT = 60


# ── 벤치마크 저장 ──────────────────────────────────────────────────────────────
def save_benchmark(records: list[dict], path: str) -> None:
    """frame별 delay_ms / fps raw data를 CSV로 저장"""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "delay_ms", "fps"])
        writer.writeheader()
        writer.writerows(records)


# ── 바운딩 박스 그리기 ──────────────────────────────────────────────────────
def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        name = det.get("name", f"cls{det['label']}")
        score = det["score"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{name}: {score:.2f}",
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return frame


def print_benchmark_summary(records: list[dict]) -> None:
    if not records:
        return
    delays = [r["delay_ms"] for r in records]
    fps_list = [r["fps"] for r in records]
    print(f"  프레임 수  : {len(records)}")
    print(
        f"  delay (ms) : min={min(delays):.1f}  max={max(delays):.1f}  avg={sum(delays) / len(delays):.1f}"
    )
    print(
        f"  fps        : min={min(fps_list):.2f}  max={max(fps_list):.2f}  avg={sum(fps_list) / len(fps_list):.2f}"
    )


# ── 메인 루틴 ─────────────────────────────────────────────────────────────────
async def stream_video() -> None:
    # 1. 입력 영상 열기
    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        sys.exit(f"[오류] 영상을 열 수 없습니다: {VIDEO_IN}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 2. 출력 영상 Writer 초기화 (YOLO 입력 크기로 저장)
    out_size = (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, out_size)
    if not writer.isOpened():
        cap.release()
        sys.exit(f"[오류] VideoWriter를 초기화할 수 없습니다: {VIDEO_OUT}")

    print(f"입력 : {VIDEO_IN}")
    print(
        f"       {w}x{h} → {YOLO_INPUT_SIZE}x{YOLO_INPUT_SIZE}  {fps:.2f}fps  총 {total}프레임"
    )
    print(f"출력 : {VIDEO_OUT}")
    print(f"벤치 : {BENCH_OUT}")
    print(f"서버 : {WS_URL}\n")

    bench: list[dict] = []  # raw benchmark records

    try:
        async with websockets.connect(
            WS_URL, max_size=MAX_WS_MSG_BYTES, open_timeout=MAX_WS_TIMEOUT
        ) as ws:
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 3. 클라이언트 측 리사이즈 → YOLO 입력 크기로 맞춰 전송
                #    서버 preprocess()의 resize 비용 제거 + 전송 데이터 감소
                frame_yolo = cv2.resize(frame, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE))
                _, buf = cv2.imencode(
                    ".jpg", frame_yolo, [cv2.IMWRITE_JPEG_QUALITY, JPEG_ENCODE_QUALITY]
                )
                t0 = time.perf_counter()
                await ws.send(buf.tobytes())

                # 4. 검출 결과 JSON 수신  (타이머 종료)
                msg = await ws.recv()
                t1 = time.perf_counter()

                detections = json.loads(msg).get("detections", [])

                # 5. 클라이언트에서 바운딩 박스 그리기 → 영상 파일에 기록
                annotated = draw_detections(frame_yolo.copy(), detections)
                writer.write(annotated)

                # 6. 벤치마크 기록
                delay_ms = (t1 - t0) * 1000
                inst_fps = 1000.0 / delay_ms if delay_ms > 0 else 0.0
                bench.append(
                    {
                        "frame": idx + 1,
                        "delay_ms": round(delay_ms, 3),
                        "fps": round(inst_fps, 3),
                    }
                )

                idx += 1
                pct = idx / total * 100 if total > 0 else 0
                print(
                    f"\r  [{idx:>5} / {total}]  {pct:5.1f}%"
                    f"  delay={delay_ms:6.1f}ms  fps={inst_fps:5.1f}",
                    end="",
                    flush=True,
                )

    except ConnectionRefusedError as exp:
        print("\n[오류] 서버에 연결할 수 없습니다.")
        print(exp)
        sys.exit(1)
    except websockets.exceptions.ConnectionClosed as e:
        print(f"\n[오류] 연결이 끊어졌습니다: {e}")
        sys.exit(1)
    finally:
        cap.release()
        writer.release()

    # 7. 벤치마크 raw data 저장 + 요약 출력
    save_benchmark(bench, BENCH_OUT)
    print(f"\n\n── 벤치마크 요약 ───────────────────────────────")
    print_benchmark_summary(bench)
    print(f"────────────────────────────────────────────────")
    print(f"\n완료!")
    print(f"  영상 → {VIDEO_OUT}")
    print(f"  벤치 → {BENCH_OUT}")


if __name__ == "__main__":
    asyncio.run(stream_video())
