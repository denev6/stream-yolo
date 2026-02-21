FROM python:3.12-slim

# Runtime libs required by opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY py_server/pyproject.toml .

# uv for fast dependency resolution; only the runtime subset needed for inference
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache \
    "fastapi[standard]>=0.129.0" \
    "numpy>=2.4.2" \
    "onnxruntime>=1.24.2" \
    "opencv-python-headless>=4.13.0.92"

COPY py_server/main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
