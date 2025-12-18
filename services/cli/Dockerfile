FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install shared euchre_core package
RUN pip install -e /shared/euchre_core

# Run CLI application
CMD ["python", "app/main.py"]
