#!/bin/bash

# fail if any errors
set -e

# Function to get current timestamp in seconds
get_timestamp() {
    date +%s
}

# Function to calculate elapsed time
calculate_elapsed_time() {
    local start_time=$1
    local end_time=$2
    local elapsed=$((end_time - start_time))
    local minutes=$((elapsed / 60))
    local seconds=$((elapsed % 60))
    printf "%02d:%02d" $minutes $seconds
}

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

# Start timing the entire script
SCRIPT_START_TIME=$(get_timestamp)

# Script to setup the environment for the experiments
print_title "Environment Setup"

echo "Installing dependencies..."
echo ""

# Start timing Golang setup
GOLANG_START_TIME=$(get_timestamp)
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

# End timing Golang setup
GOLANG_END_TIME=$(get_timestamp)
echo "Golang setup completed in $(calculate_elapsed_time $GOLANG_START_TIME $GOLANG_END_TIME)"
echo ""

# Start timing ASDF setup
ASDF_START_TIME=$(get_timestamp)
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

# End timing ASDF setup
ASDF_END_TIME=$(get_timestamp)
echo "ASDF setup completed in $(calculate_elapsed_time $ASDF_START_TIME $ASDF_END_TIME)"
echo ""

# Start timing Python setup
PYTHON_START_TIME=$(get_timestamp)
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

# End timing Python setup
PYTHON_END_TIME=$(get_timestamp)
echo "Python setup completed in $(calculate_elapsed_time $PYTHON_START_TIME $PYTHON_END_TIME)"
echo ""

# Start timing UV setup
UV_START_TIME=$(get_timestamp)
print_title "UV Setup"
echo "Checking UV installation..."

# End timing UV setup
UV_END_TIME=$(get_timestamp)
echo "UV setup completed in $(calculate_elapsed_time $UV_START_TIME $UV_END_TIME)"
echo ""

# Calculate total script execution time
SCRIPT_END_TIME=$(get_timestamp)
print_title "Setup Complete"
echo "Environment setup completed in $(calculate_elapsed_time $SCRIPT_START_TIME $SCRIPT_END_TIME)"
echo ""