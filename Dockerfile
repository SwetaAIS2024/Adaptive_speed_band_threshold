FROM python:3.11-slim

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
# Ensures stdout/stderr are flushed immediately
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app/             app/
COPY clustering/      clustering/
COPY data/            data/
# synthetic_dataset/processed/ is provided at runtime via docker-compose volume mount

# Streamlit configuration — disable telemetry, set server options
RUN mkdir -p /root/.streamlit
RUN echo "\
[browser]\n\
gatherUsageStats = false\n\
[server]\n\
port = 8501\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
" > /root/.streamlit/config.toml

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
