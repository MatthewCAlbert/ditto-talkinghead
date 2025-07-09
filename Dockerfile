# Use NVIDIA PyTorch container as base (CUDA 12.1, Python 3.10, PyTorch 2.5.1)
FROM nvcr.io/nvidia/pytorch:24.05-py3

# Set working directory
WORKDIR /workspace

# Copy environment file and project code
COPY environment.yaml ./
COPY . ./

# Install mamba for fast conda operations
RUN conda install -y mamba -c conda-forge

# Create and activate environment, install dependencies
RUN mamba env update -n base -f environment.yaml && \
    conda clean -afy

# Install Jupyter Lab and notebook extensions
RUN pip install --no-cache-dir jupyterlab notebook ipykernel

# Expose Jupyter port
EXPOSE 8888

# Set default command to launch Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"] 