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

# Install Python dependencies
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir -e "."

# Download GGUF model at build time
# NOTE: Update this URL to the actual quantized GGUF location
RUN mkdir -p /data/models && \
    curl -L -o /data/models/tiny-aya-q4.gguf \
    "https://huggingface.co/CohereLabs/tiny-aya-base-gguf/resolve/main/tiny-aya-base-Q4_K_M.gguf"

# Copy application source
COPY src/ src/

# Copy baked-in corpus (add docs to data/corpus/ before building)
COPY data/corpus/ /data/corpus/

# Create qdrant storage directory
RUN mkdir -p /data/qdrant

EXPOSE 8080

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
