# 1. Stable Base Image
# Using python:3.12-slim to keep the image lightweight and secure
FROM python:3.12-slim

# 2. System Dependencies
# These are installed once and cached. ffmpeg is required for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 3. Environment Configuration
# Setting the working directory and PYTHONPATH to resolve internal modules correctly
WORKDIR /app
ENV PYTHONPATH=/app

# 4. Library Installation (Heavy Layer)
# This layer is only rebuilt if requirements.txt changes, saving build time
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Application Code (Light Layer)
# Copying the source code last allows for near-instant builds when changing logic
COPY . .

# Expose the default Streamlit port
EXPOSE 8501

# Command to execute the application with network bindings for Docker
CMD ["streamlit", "run", "src/frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]