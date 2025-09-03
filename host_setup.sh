#!/bin/bash

# This script sets up direnv and Python environment with flag file for Python setup only

set -e  # Exit on any error

source "$(dirname "$0")/bash_scripts/colors.sh"

# Unified direnv Setup Function
setup_direnv() {
    # Function to install direnv using apt
    install_via_apt() {
        log_info "Installing direnv using apt..."
        sudo apt update && sudo apt install -y direnv
    }

    # Check if direnv is already installed
    if command -v direnv >/dev/null 2>&1; then
        log_info "direnv is already installed: $(direnv version)"
    else
        log_info "direnv not found. Installing via apt..."

        # Try apt installation only
        if install_via_apt; then
            log_info "direnv installed successfully via apt"
        else
            log_error "Failed to install direnv via apt"
            return 1
        fi
    fi

    # Verify installation
    if command -v direnv >/dev/null 2>&1; then
        log_info "Installation verified: $(direnv version)"

        # Setup bash hook
        setup_direnv_hook

        # Force allow any .envrc in current directory
        if [ -f ".envrc" ]; then
            log_info "Found .envrc file, allowing it automatically..."
            direnv allow
        fi

        log_info "direnv setup complete!"
        return 0
    else
        log_error "Installation failed - direnv command not found"
        return 1
    fi
}

# Function to setup direnv bash hook
setup_direnv_hook() {
    log_info "Setting up direnv bash hook..."

    local hook_line='eval "$(direnv hook bash)"'
    local bashrc_file="$HOME/.bashrc"

    # Check if hook is already in .bashrc
    if grep -Fxq "$hook_line" "$bashrc_file" 2>/dev/null; then
        log_info "direnv hook already configured in .bashrc"
    else
        log_info "Adding direnv hook to .bashrc..."
        echo "" >> "$bashrc_file"
        echo "# direnv hook" >> "$bashrc_file"
        echo "$hook_line" >> "$bashrc_file"
        log_info "direnv hook added to .bashrc"
    fi

    # Also set it up for the current session
    log_info "Setting up direnv hook for current session..."
    eval "$(direnv hook bash)"
}

# Function to setup Python environment
setup_python_env() {
    export CUDA_HOME=/usr/local/cuda
    log_info "Setting up Python environment..."

    # Set UV cache directory
    export UV_CACHE_DIR="$REMOTE_HOME_DIR/cache"
    log_info "Using UV cache directory: $UV_CACHE_DIR"

    # Create cache directory if it doesn't exist
    mkdir -p "$UV_CACHE_DIR"

    # Install uv with specific version
    log_info "Installing uv==0.7.19..."
    pip install uv==0.7.19

    # Add ~/.local/bin to PATH if it exists and isn't already there
    if [ -d "$HOME/.local/bin" ] && [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
        export PATH="$HOME/.local/bin:$PATH"
        log_info "Added ~/.local/bin to PATH"
    fi

    # Verify uv is accessible
    if ! command -v uv >/dev/null 2>&1; then
        log_error "uv command not found after installation. Please ensure ~/.local/bin is in your PATH"
        return 1
    fi

    log_info "uv version: $(uv --version)"

    uv venv --allow-existing

    # Sync dependencies (excluding cuda group)
    log_info "Syncing dependencies with uv..."
    uv sync --no-group cuda

    # Install bitsandbytes
    log_info "Installing bitsandbytes..."
    uv pip install bitsandbytes==0.46.1

    # Install PyTorch with CUDA support
    log_info "Installing PyTorch with CUDA 12.8 support..."
    CUDA_HOME=/usr/local/cuda uv pip install --force-reinstall torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128

    # Install flash-attn with specific build configuration
    log_info "Installing flash-attn (this may take a while)..."
    log_warn "Using MAX_JOBS=24 to prevent OOM errors during compilation"
    CUDA_HOME=/usr/local/cuda MAX_JOBS=24 uv pip install -v --upgrade flash-attn==2.8.0.post2 --no-build-isolation

    # Download model
    log_info "Downloading Llama model..."
    source .venv/bin/activate

    DIR=./downloaded-models/gradientai/Llama-3-8B-Instruct-262k
    if [ -d "$DIR" ] && [ -z "$(ls -A "$DIR")" ]; then
      log_info "Directory is empty. Downloading model..."
      python scripts/huggingface_downloader.py gradientai/Llama-3-8B-Instruct-262k "$DIR"
    else
      echo "Directory is not empty. Skipping model download."
    fi

    DIR=./downloaded-models/openai/gpt-oss-20b
    if [ -d "$DIR" ] && [ -z "$(ls -A "$DIR")" ]; then
      log_info "Directory is empty. Downloading openai model..."
      python scripts/huggingface_downloader.py openai/gpt-oss-20b "$DIR"
    else
      echo "Directory is not empty. Skipping openai model download."
    fi

    log_info "Python environment setup complete!"


    log_info "Setting up lambda labs agent"
    curl -L https://lambdalabs-guest-agent.s3.us-west-2.amazonaws.com/scripts/install.sh | sudo bash

    return 0
}

# Main entrypoint
main() {
    log_info "Starting combined setup..."

    # Create tmp directory if it doesn't exist
    mkdir -p ./tmp

    # Check flag file for Python setup
    local python_flag_file="/tmp/$REMOTE_HOME_DIR/python_setup_complete"
    mkdir -p "/tmp/$REMOTE_HOME_DIR/"

    if [ -f "$python_flag_file" ]; then
        log_info "Python setup already completed (flag file exists)"
    else
        log_info "Running Python setup..."
        if setup_python_env; then
            # Create flag file to indicate successful completion
            touch "$python_flag_file"
            log_info "Created flag file: $python_flag_file"
        else
            log_error "Python setup failed"
            return 1
        fi
    fi

    # Always run direnv setup (no flag file)
    log_info "Setting up direnv..."
    setup_direnv

    log_info "Combined setup complete!"
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
