import onnxruntime as ort
import numpy as np
import cv2

# 1. 세션 생성 (CPU 또는 GPU 선택)
# GPU 사용 시 'CUDAExecutionProvider' 추가
session = ort.InferenceSession(
    "../model/yolo26n.onnx", providers=["CPUExecutionProvider"]
)

# 2. 이미지 전처리 (640x640 크기, 정규화 등)
img = cv2.imread("../assets/test.jpg")
img_input = cv2.resize(img, (640, 640))
img_input = img_input.transpose(2, 0, 1)  # HWC -> CHW
img_input = img_input[np.newaxis, :, :, :].astype(np.float32) / 255.0

# 3. 추론 실행
outputs = session.run(None, {"images": img_input})

# 4. 결과 해석
# YOLO26은 NMS 과정 없이 바로 최종 [box, score, label] 형태의 결과가 출력됩니다.
detections = outputs[0]
print(detections)
