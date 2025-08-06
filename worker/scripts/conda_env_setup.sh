#!/bin/bash
set -e  # exit on error

export DEBIAN_FRONTEND=noninteractive
ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime
echo "Etc/UTC" > /etc/timezone

# Install Rust.
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# 4) "conda init" is typically for interactive shells. In Docker builds,
#    you can often just add /opt/conda/bin to PATH or run /opt/conda/bin/conda directly.
#    But if you do want it available for interactive shells in the container, keep it.
/opt/conda/bin/conda init bash

# Accept Terms of Service for all channels to avoid interactive prompts
/opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
/opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r


# 5) Install all GPU dependencies.
/opt/conda/bin/conda create -y -n worker \
  python=3.12                             \
  pytorch=2.2.*                           \
  torchvision=0.17.*                      \
  torchaudio=2.2.*                        \
  pytorch-cuda=12.1                       \
  gpytorch=1.14                           \
  botorch=0.14.*                          \
  linear_operator=0.6                     \
  pyro-ppl>=1.8.4                         \
  -c pytorch -c nvidia -c gpytorch -c conda-forge

# /opt/conda/bin/conda create -y -n worker \
#     python=3.12 \
#     cudatoolkit=11.8 \
#     pytorch-cuda=12.1 \
#     pytorch \
#     torchvision \
#     torchaudio \
#     -c pytorch -c nvidia -c conda-forge
# 6) Clean up conda cache
/opt/conda/bin/conda clean -afq

# 7) Install pip packages for your worker environment
/opt/conda/envs/worker/bin/pip install pip-tools
/opt/conda/envs/worker/bin/pip install --no-cache-dir --no-deps -r /backend/requirements.txt

# Clean conda & pip
/opt/conda/bin/conda clean -a -y
rm -rf /opt/conda/pkgs
rm -rf /root/.cache/pip

# Clean Rust if not needed
rm -rf /root/.cargo /root/.rustup
