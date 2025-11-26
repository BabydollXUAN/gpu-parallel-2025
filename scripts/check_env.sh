set -e

echo "=== nvidia-smi ==="
nvidia-smi || echo "nvidia-smi not found"

echo
echo "=== nvcc --version ==="
nvcc --version || echo "nvcc not found"
