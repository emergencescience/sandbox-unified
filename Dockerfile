FROM python:3.11-slim

# ── 部署注意：本服务已开启 Railway Serverless（无出站流量 5–10 分钟后休眠，省内存费）──
# 保持「可休眠」的前提（改动前先读 Railway 文档 deployments/serverless）：
#   1) 不要引入后台心跳/定时轮询（任何周期性出站包都会让服务永远醒着）；
#   2) 不要保持空闲数据库连接（本服务无 DB，请继续保持无状态）；
#   3) 平台健康检查只会探 /health，不会造成持续出站。
# 代价：休眠后第一个请求可能返回 502。调用方（emergence-orchestrator）已加冷启动重试
#       （core/upstream_retry.py），切勿把该重试去掉。

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
