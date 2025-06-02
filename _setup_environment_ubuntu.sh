#!/bin/bash

# Script to setup the environment for the experiments

echo "Installing dependencies..."
echo ""

echo "Installing Golang..."

# Install go
sudo apt update && sudo apt upgrade
sudo apt install golang-go

echo "Golang installed"
echo ""

echo "Installing asdf..."

# Install asdf
apt install git
go install github.com/asdf-vm/asdf/cmd/asdf@v0.17.0

echo "asdf installed"
echo ""

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

echo "Environment setup complete!"
echo ""