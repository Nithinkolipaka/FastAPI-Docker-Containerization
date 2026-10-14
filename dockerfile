# ---------------------------------------------------------------
# Dockerfile — Instructions to build the QA Evaluator Docker image
#
# What is Docker?
#   Docker packages your application and all its dependencies into a
#   "container" — a lightweight, portable box that runs the same way
#   on any computer, server, or cloud platform.
#
# What is a base image?
#   A base image is a starting point. We use python:3.11-slim, which is
#   an official Python image that is small (slim) and fast to download.
#   It includes Python 3.11 but nothing extra, keeping the image size small.
#
# Why WORKDIR /app?
#   Sets the working directory inside the container to /app.
#   All following commands run from this directory.
#
# Why copy requirements.txt first (before the rest of the code)?
#   Docker builds images in layers. If requirements.txt hasn't changed,
#   Docker reuses the cached pip install layer — making rebuilds much faster.
#   This is a Docker best practice called "layer caching optimization".
#
# Why --no-cache-dir?
#   pip stores downloaded packages in a cache to speed up future installs.
#   Inside Docker, we don't need this cache (the container is rebuilt fresh
#   each time), so --no-cache-dir saves disk space in the image.
#
# Why EXPOSE 8000?
#   Tells Docker (and anyone reading the Dockerfile) that the app listens
#   on port 8000. You still need to map this port when running the container:
#     docker run -p 8000:8000 qa-evaluator-app
#
# Why uvicorn?
#   Uvicorn is a fast Python web server for running FastAPI applications.
#   --host 0.0.0.0 means "listen on all network interfaces" so requests
#   from outside the container can reach the app.
#
# Best Practices:
#   - Use a slim base image to keep the container small.
#   - Copy requirements.txt before source code for better layer caching.
#   - Never copy .env into the image — pass secrets at runtime with --env-file.
#   - Create required directories (data, logs) so the app starts without errors.
# ---------------------------------------------------------------

# Use official Python 3.11 slim image as the base
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements file first (enables Docker layer caching)
COPY requirements.txt .

# Install all Python dependencies (no cache to keep image size small)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application source code into the container
COPY . .

# Create required directories that the app needs at runtime
RUN mkdir -p data logs

# Tell Docker this container listens on port 8000
EXPOSE 8000

# Start the FastAPI app using uvicorn when the container runs
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]