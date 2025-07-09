#!/bin/bash

# Combined Setup Script
# This script sets up direnv and Python environment with flag file for Python setup only

set -e  # Exit on any error

source "$(dirname "$0")/bash_scripts/colors.sh"

# Unified direnv Setup Function
setup_direnv() {
    # Function to install direnv using apt
    install_via_apt() {
        print_status "Installing direnv using apt..."
        sudo apt update && sudo apt install -y direnv
    }

    # Check if direnv is already installed
    if command -v direnv >/dev/null 2>&1; then
        print_status "direnv is already installed: $(direnv version)"
    else
        print_status "direnv not found. Installing via apt..."

        # Try apt installation only
        if install_via_apt; then
            print_status "direnv installed successfully via apt"
        else
            log_error "Failed to install direnv via apt"
            return 1
        fi
    fi

    # Verify installation
    if command -v direnv >/dev/null 2>&1; then
        print_status "Installation verified: $(direnv version)"

        # Force allow any .envrc in current directory
        if [ -f ".envrc" ]; then
            print_status "Found .envrc file, allowing it automatically..."
            direnv allow
        fi

        print_status "direnv setup complete!"
        return 0
    else
        log_error "Installation failed - direnv command not found"
        return 1
    fi
}

# Function to setup Python environment
setup_python_env() {
    print_status "Setting up Python environment..."

    # Set UV cache directory
    export UV_CACHE_DIR="/home/ubuntu/mars-arnesen-gh/cache"
    print_status "Using UV cache directory: $UV_CACHE_DIR"

    # Create cache directory if it doesn't exist
    mkdir -p "$UV_CACHE_DIR"

    # Install uv with specific version
    print_status "Installing uv==0.7.19..."
    pip install uv==0.7.19

    # Add ~/.local/bin to PATH if it exists and isn't already there
    if [ -d "$HOME/.local/bin" ] && [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
        export PATH="$HOME/.local/bin:$PATH"
        print_status "Added ~/.local/bin to PATH"
    fi

    # Verify uv is accessible
    if ! command -v uv >/dev/null 2>&1; then
        log_error "uv command not found after installation. Please ensure ~/.local/bin is in your PATH"
        return 1
    fi

    print_status "uv version: $(uv --version)"

    uv venv --allow-existing

    # Sync dependencies (excluding cuda group)
    print_status "Syncing dependencies with uv..."
    uv sync --no-group cuda

    # Install bitsandbytes
    print_status "Installing bitsandbytes..."
    uv pip install bitsandbytes==0.46.1

    # Install PyTorch with CUDA support
    print_status "Installing PyTorch with CUDA 12.8 support..."
    uv pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128

    # Install flash-attn with specific build configuration
    print_status "Installing flash-attn (this may take a while)..."
    log_warn "Using MAX_JOBS=24 to prevent OOM errors during compilation"
    CUDA_HOME=/usr/local/cuda MAX_JOBS=24 uv pip install -v --upgrade flash-attn==2.8.0.post2 --no-build-isolation

    # Download model
    print_status "Downloading Llama model..."
    python scripts/huggingface_downloader.py gradientai/Llama-3-8B-Instruct-262k ./downloaded-models/gradientai/Llama-3-8B-Instruct-262k

    print_status "Python environment setup complete!"
    return 0
}

# Main entrypoint
main() {
    print_status "Starting combined setup..."

    # Create tmp directory if it doesn't exist
    mkdir -p ./tmp

    # Check flag file for Python setup
    local python_flag_file="./tmp/python_setup_complete"

    if [ -f "$python_flag_file" ]; then
        print_status "Python setup already completed (flag file exists)"
    else
        print_status "Running Python setup..."
        if setup_python_env; then
            # Create flag file to indicate successful completion
            touch "$python_flag_file"
            print_status "Created flag file: $python_flag_file"
        else
            log_error "Python setup failed"
            return 1
        fi
    fi

    # Always run direnv setup (no flag file)
    print_status "Setting up direnv..."
    setup_direnv

    print_status "Combined setup complete!"
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
