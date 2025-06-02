#!/bin/bash

# fail if any errors
set -ex

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
    export PATH="${ASDF_DATA_DIR:-$HOME/.asdf}/shims:$PATH" > ~/.bashrc
    echo "asdf installed"
fi
echo ""

print_title "Python Setup"
echo "Installing python..."

# Install python
asdf plugin add python
asdf install python 3.12.4

echo "python installed"
echo ""

# Install uv
asdf plugin add uv
asdf install uv 0.7.5

echo "uv installed"
echo ""

echo "Performing uv sync..."

# Perform uv sync
uv sync

echo "uv sync completed"
echo ""

print_title "Setup Complete"
echo "Environment setup complete!"
echo ""