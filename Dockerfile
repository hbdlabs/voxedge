FROM python:3.11-slim

# Install system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl unzip build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Bun (for LiteParse)
RUN curl -fsSL https://bun.sh/install | bash
ENV PATH="/root/.bun/bin:$PATH"

# Install LiteParse CLI
RUN bun install -g @llamaindex/liteparse

# Copy source first (needed for pip install)
WORKDIR /app
COPY pyproject.toml .
COPY src/ src/

# Install Python dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir "."

# Download GGUF model at build time (~2.1 GB)
RUN mkdir -p /data/models && \
    curl -L -o /data/models/tiny-aya-global-q4_k_m.gguf \
    "https://huggingface.co/CohereLabs/tiny-aya-global-GGUF/resolve/main/tiny-aya-global-q4_k_m.gguf"

# Copy baked-in corpus (add docs to data/corpus/ before building)
COPY data/corpus/ /data/corpus/

# Create qdrant storage directory
RUN mkdir -p /data/qdrant

EXPOSE 8080

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
