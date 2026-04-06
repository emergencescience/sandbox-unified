FROM python:3.11-slim

# Install system dependencies & Node.js
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install global JS/TS tools
RUN npm install -g typescript tsx

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Set environment
ENV PORT=3004
EXPOSE 3004

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3004"]
