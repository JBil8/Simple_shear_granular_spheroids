#!/bin/bash

# Check if .venv directory exists in the current directory
if [ -d ".venv" ]; then
    echo "Virtual environment '.venv' already exists."
else
    echo "Creating virtual environment '.venv'..."
    python3 -m venv .venv
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Check if requirements.txt exists and install dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "Error: requirements.txt not found in the current directory."
    exit 1
fi

echo "Virtual environment setup and dependencies installed successfully."