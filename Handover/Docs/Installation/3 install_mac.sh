#!/bin/bash

set -e  # Exit on first error

# --- Check if setup.py exists ---
if [[ ! -f "setup.py" ]]; then
    echo "❌ setup.py not found in the current directory. Exiting."
    exit 1
fi

# --- Check if Python3 is installed ---
if ! command -v python3 &> /dev/null; then
    echo "🛠️ Python3 is not installed."

    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "🔧 Installing Python via Homebrew..."
        if ! command -v brew &> /dev/null; then
            echo "🍺 Homebrew not found. Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        brew install python
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "🔧 Installing Python via apt (requires sudo)..."
        sudo apt update
        sudo apt install -y python3 python3-pip python3-tk
    else
        echo "❌ Unsupported OS."
        exit 1
    fi
fi

# --- Check if pip is installed ---
if ! command -v pip3 &> /dev/null; then
    echo "📦 Installing pip..."
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3 get-pip.py
    rm get-pip.py
fi

# --- Run the setup script ---
echo "🚀 Running Python setup..."
python3 setup.py

echo "✅ Setup complete."
