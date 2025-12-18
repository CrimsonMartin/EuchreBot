FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install shared euchre_core package
RUN pip install -e /shared/euchre_core

# Expose port
EXPOSE 5001

# Run Flask app
CMD ["flask", "run", "--host=0.0.0.0", "--port=5001"]
