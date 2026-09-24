# 1. Use Python 3.11 slim image
FROM python:3.11-slim

# 2. Set working directory
WORKDIR /app

# 3. Install OpenMP runtime required by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# 4. Copy clean requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the application code and MLflow artifacts
COPY . .

# 6. Expose the port
EXPOSE 8000

# 7. Start Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]