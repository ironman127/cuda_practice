export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

export CUDA_HOME=/usr/local/cuda-13.0
export CUDACXX=$CUDA_HOME/bin/nvcc
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH

export PATH=/opt/nvidia/nsight-systems/2026.3.1/target-linux-x64:$PATH
export PATH=/opt/nvidia/nsight-compute/2026.2.0:$PATH
