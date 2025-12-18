FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install shared euchre_core package
RUN pip install -e /shared/euchre_core

# Expose port
EXPOSE 5002

# Set PYTHONPATH
ENV PYTHONPATH=/app:$PYTHONPATH

# Run Flask app
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5002"]
