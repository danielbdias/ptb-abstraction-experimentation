#!/bin/bash

# fail if any errors
set -e

# Function to print a title with decoration
print_title() {
    local title="$1"
    local width=50
    local padding=$(( (width - ${#title}) / 2 ))
    
    echo ""
    printf '%*s' "$width" | tr ' ' '='
    printf '%*s%s%*s\n' "$padding" "" "$title" "$padding" ""
    printf '%*s' "$width" | tr ' ' '='
    echo ""
}

# Script to setup the environment for the experiments
print_title "Environment Setup"

echo "Installing dependencies..."
echo ""

print_title "Golang Setup"
echo "Checking Golang installation..."

# Check if go is installed
if command -v go &> /dev/null; then
    echo "Go is already installed: $(go version)"
else
    echo "Installing Golang..."
    # Install go
    sudo apt update && sudo apt upgrade
    sudo apt install golang-go -y

    # Add go to path
    export PATH="$PATH:$HOME/go/bin"
    
    # Add to bashrc
    echo "export PATH=\"\$PATH:$HOME/go/bin\"" >> ~/.bashrc

    echo "Golang installed"
fi

echo ""

print_title "ASDF Setup"
echo "Checking asdf installation..."

# Check if asdf is installed
if command -v asdf &> /dev/null; then
    echo "asdf is already installed: $(asdf --version)"
else
    echo "Installing asdf..."
    
    # Install asdf
    apt install git -y
    go install github.com/asdf-vm/asdf/cmd/asdf@v0.17.0
    
    # Add to bashrc
    export PATH="${ASDF_DATA_DIR:-$HOME/.asdf}/shims:$PATH"
    echo "export PATH=\"\${ASDF_DATA_DIR:-\$HOME/.asdf}/shims:\$PATH\"" >> ~/.bashrc
    
    echo "asdf installed"
fi
echo ""

print_title "Python Setup"
echo "Checking Python installation..."

# Check if python is installed via asdf
if asdf list python | grep -q "3.12.4"; then
    echo "Python 3.12.4 is already installed via asdf"
else
    echo "Installing Python 3.12.4 via asdf..."

    apt update && apt install -y \
        make \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        libncursesw5-dev \
        libgdbm-dev \
        liblzma-dev \
        tk-dev \
        uuid-dev \
        libffi-dev \
        libdb-dev \
        libnss3-dev \
        libxml2-dev \
        libxmlsec1-dev \
        xz-utils \
        wget \
        curl \
        llvm
    
    asdf plugin add python
    asdf install python 3.12.4
    
    echo "Python installed"
fi
echo ""

# Check if python is installed via asdf
if asdf list uv | grep -q "0.7.5"; then
    echo "uv 0.7.5 is already installed via asdf"
else
    echo "Installing uv 0.7.5 via asdf..."
    
    asdf plugin add uv
    asdf install uv 0.7.5
    
    echo "uv installed"
    echo ""
fi

echo "Performing uv sync..."

# Perform uv sync
uv sync

echo "uv sync completed"
echo ""

print_title "Setup Complete"
echo "Environment setup complete!"
echo ""