# Use NVIDIA PyTorch container as base (already has CUDA 12.1, Python 3.10, PyTorch)
FROM nvcr.io/nvidia/pytorch:24.05-py3

# Set working directory
WORKDIR /workspace

# Update system packages and install ffmpeg (required by README)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . ./

# Install dependencies exactly as specified in README.md pip section
RUN pip install --no-cache-dir \
    tensorrt==8.6.1 \
    librosa \
    tqdm \
    filetype \
    imageio \
    opencv_python_headless \
    scikit-image \
    cython \
    cuda-python \
    imageio-ffmpeg \
    colored \
    polygraphy \
    numpy==2.0.1

# Install Jupyter Lab and notebook
RUN pip install --no-cache-dir \
    jupyterlab \
    notebook \
    ipykernel

# Expose Jupyter port
EXPOSE 8888

# Set default command to launch Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"] 