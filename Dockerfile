# Use NVIDIA PyTorch container as base (CUDA 12.1, Python 3.10, PyTorch 2.5.1)
FROM nvcr.io/nvidia/pytorch:24.05-py3

# Set working directory
WORKDIR /workspace

# Update system packages and install dependencies
RUN apt-get update && apt-get install -y \
    wget \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . ./

# Install Python dependencies directly with pip (more reliable than conda in containers)
RUN pip install --no-cache-dir \
    jupyterlab \
    notebook \
    ipykernel \
    librosa==0.10.2.post1 \
    opencv-python-headless==4.10.0.84 \
    scikit-image==0.25.0 \
    scikit-learn==1.6.0 \
    imageio==2.36.1 \
    imageio-ffmpeg==0.5.1 \
    soundfile==0.13.0 \
    tensorrt==8.6.1 \
    cython==3.0.11 \
    cuda-python==12.6.2.post1 \
    filetype==1.2.0 \
    tqdm==4.67.1 \
    colored \
    polygraphy

# Expose Jupyter port
EXPOSE 8888

# Set default command to launch Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"] 