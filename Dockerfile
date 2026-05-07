FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg curl sox \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/andimarafioti/faster-qwen3-tts.git /app/faster-qwen3-tts

WORKDIR /app/faster-qwen3-tts

RUN pip install --break-system-packages --upgrade pip && \
    pip install --break-system-packages -e ".[demo]" pydub

EXPOSE 8880

CMD ["python", "examples/openai_server.py", \
     "--model", "Qwen/Qwen3-TTS-12Hz-1.7B-Base", \
     "--voices", "/voices/voices.json", \
     "--host", "0.0.0.0", \
     "--port", "8880", \
     "--device", "cuda"]