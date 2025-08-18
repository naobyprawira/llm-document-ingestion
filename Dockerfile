FROM python:3.12-slim

# Install OS dependencies needed for PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libfreetype6 \
    libjpeg62-turbo \
    libopenjp2-7 \
    liblcms2-2 \
    libtiff5 \
    libpng16-16 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Default command prints help for ingestion CLI
CMD ["python", "cli/ingest_pdf.py", "--help"]