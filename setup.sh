#!/bin/bash
set -e

echo "========================================"
echo "        RobotAI Setup Script"
echo "========================================"

# 1. System dependencies
echo ""
echo "[1/6] Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    ffmpeg \
    libportaudio2 \
    portaudio19-dev \
    ca-certificates \
    curl

# 2. Python dependencies
echo ""
echo "[2/6] Installing Python dependencies..."
pip install torch==2.9.1+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install nemo-toolkit[asr]==2.4.0
pip install -r requirements.txt
# Stage 5: VachanaTTS (no setup.py — install dependencies manually)
echo "Installing VachanaTTS..."
VACHANA_DIR="/home/ancfhtna/capstone/RobotAI/vachanatts"
git clone https://github.com/VYNCX/VachanaTTS.git "$VACHANA_DIR"
cd "$VACHANA_DIR"
# Install any requirements if present
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi
# Copy module directly to site-packages
pip show torch | grep Location | awk '{print $2}' | xargs -I{} cp -r . {}/vachanatts 2>/dev/null || \
    cp -r . $(python -c "import site; print(site.getsitepackages()[0])")/vachanatts
cd ..
echo "VachanaTTS installed."

# 3. Ollama
echo ""
echo "[3/6] Installing Ollama and pulling model..."
sudo snap install ollama
ollama pull qwen2.5:7b-instruct
ollama serve &
echo "Ollama started in background."

# 4. Docker
echo ""
echo "[4/6] Installing Docker..."
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt-get update
sudo apt-get install -y \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin

# 5. Milvus
echo ""
echo "[5/6] Setting up Milvus..."
wget https://github.com/milvus-io/milvus/releases/download/v2.4.0/milvus-standalone-docker-compose.yml -O docker-compose.yml
mkdir -p ~/milvus && mv docker-compose.yml ~/milvus/

# 6. Models reminder
echo ""
echo "[6/6] Models: Please place your models in the models/ directory."
echo "       See README for download links."


# # 7. Done
# echo ""
# echo "[7/7] Setup complete!"
# echo ""
# echo "Next steps:"
# echo "1. cd ~/milvus"
# echo "2. sudo docker compose up -d"
# echo "3. Place models in the models/ directory"
# echo "4. Enroll a student:  python demos/demo_enrollment.py"
# echo "5. Run the demo:      python demos/demo_end_to_end.py"
# echo "========================================"