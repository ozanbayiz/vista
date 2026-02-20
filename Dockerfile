# ---------------------------------------------------------------------------
# IDARVE: Investigating Demographic Attribute Representation in Vision Encoders
# Reproducible research environment
# ---------------------------------------------------------------------------
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specification first (cache-friendly layer ordering)
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Copy source code
COPY . .

# Default: show help
CMD ["python", "-m", "src.main", "--help"]
