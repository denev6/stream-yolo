import argparse
import asyncio
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import websockets
from tqdm import tqdm

parser = argparse.ArgumentParser(description="비디오 처리 및 부하 테스트 클라이언트")
parser.add_argument("--video-in", type=str, default="assets/test_30.mp4")
parser.add_argument(
    "--video-out", type=str, default=None, help="생략 시 비디오 저장 안 함"
)
parser.add_argument("--bench-out", type=str, default="results/benchmark.csv")
parser.add_argument("--server-type", type=str, default="go", choices=["go", "py"])
parser.add_argument("--clients", type=int, default=1, help="동시 접속 클라이언트 수")

args = parser.parse_args()

_ROOT = Path(__file__).parent.parent
VIDEO_IN = str(_ROOT / args.video_in)
BENCH_OUT = str(_ROOT / args.bench_out)

if args.server_type.lower() == "go":
    PORT = 8001
elif args.server_type.lower() == "py":
    PORT = 8000
else:
    sys.exit(f"[오류] 지원하지 않는 서버 타입: {args.server_type}")

WS_URL = f"ws://localhost:{PORT}/ws/stream"
YOLO_INPUT_SIZE = 640
JPEG_ENCODE_QUALITY = 90
MAX_WS_MSG_BYTES = 16 * 1024 * 1024
MAX_WS_TIMEOUT = 60


def save_benchmark(records: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["client_id", "frame", "delay_ms", "fps"])
        writer.writeheader()
        writer.writerows(records)


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
    print(f"  총 처리 프레임 : {len(records)}")
    print(
        f"  delay (ms)     : min={min(delays):.1f}  max={max(delays):.1f}  avg={sum(delays) / len(delays):.1f}"
    )
    print(
        f"  fps            : min={min(fps_list):.2f}  max={max(fps_list):.2f}  avg={sum(fps_list) / len(fps_list):.2f}"
    )


async def stream_video(client_id: int) -> list[dict]:
    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None

    # 다중 클라이언트의 경우 파일명에 client_id 추가
    if args.video_out:
        out_path = Path(_ROOT / args.video_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        client_video_out = (
            str(out_path.with_name(f"{out_path.stem}_{client_id}{out_path.suffix}"))
            if args.clients > 1
            else str(out_path)
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            client_video_out, fourcc, fps, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE)
        )

    bench = []

    # 각 클라이언트별 프로그레스 바 위치(position) 분리
    pbar = tqdm(
        total=total,
        position=client_id,
        desc=f"Client {client_id}",
        leave=True,
        ascii=True,
    )

    try:
        async with websockets.connect(
            WS_URL, max_size=MAX_WS_MSG_BYTES, open_timeout=MAX_WS_TIMEOUT
        ) as ws:
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_yolo = cv2.resize(frame, (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE))
                _, buf = cv2.imencode(
                    ".jpg", frame_yolo, [cv2.IMWRITE_JPEG_QUALITY, JPEG_ENCODE_QUALITY]
                )

                t0 = time.perf_counter()
                await ws.send(buf.tobytes())
                msg = await ws.recv()
                t1 = time.perf_counter()

                delay_ms = (t1 - t0) * 1000
                inst_fps = 1000.0 / delay_ms if delay_ms > 0 else 0.0

                if writer:
                    detections = json.loads(msg).get("detections", [])
                    annotated = draw_detections(frame_yolo.copy(), detections)
                    writer.write(annotated)

                bench.append(
                    {
                        "client_id": client_id,
                        "frame": idx + 1,
                        "delay_ms": round(delay_ms, 3),
                        "fps": round(inst_fps, 3),
                    }
                )

                idx += 1
                pbar.update(1)
                pbar.set_postfix(delay=f"{delay_ms:.1f}ms", fps=f"{inst_fps:.1f}")

    except Exception as e:
        pbar.write(f"[Client {client_id} 오류] {e}")
    finally:
        pbar.close()
        cap.release()
        if writer:
            writer.release()

    return bench


async def main():
    print(f"입력 : {VIDEO_IN}")
    print(f"서버 : {WS_URL}")
    print(f"클라이언트 수 : {args.clients}개 동시 접속\n")

    tasks = [stream_video(i) for i in range(args.clients)]
    results = await asyncio.gather(*tasks)

    # 콘솔 출력 겹침 방지용 줄바꿈
    print("\n" * args.clients)

    all_benchmarks = []
    for bench in results:
        all_benchmarks.extend(bench)

    if all_benchmarks:
        save_benchmark(all_benchmarks, BENCH_OUT)
        print(f"── 벤치마크 요약 ({args.clients} Clients) ────────────────")
        print_benchmark_summary(all_benchmarks)
        print(f"────────────────────────────────────────────────")
        print(f"\n완료!")
        if args.video_out:
            print(f"  영상 저장됨")
        print(f"  벤치 → {BENCH_OUT}")


if __name__ == "__main__":
    asyncio.run(main())
