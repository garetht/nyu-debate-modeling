#!/bin/bash

# direnv Setup Script
# This script checks if direnv is installed and installs it if not present

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to install direnv using apt
install_direnv_apt() {
    print_status "Installing direnv using apt..."
    sudo apt update
    sudo apt install -y direnv
}

# Function to install direnv from binary release
install_direnv_binary() {
    print_status "Installing direnv from binary release..."

    # Detect architecture
    local arch=$(uname -m)
    case $arch in
        x86_64)
            arch="amd64"
            ;;
        aarch64)
            arch="arm64"
            ;;
        armv7l)
            arch="arm"
            ;;
        *)
            print_error "Unsupported architecture: $arch"
            return 1
            ;;
    esac

    # Get latest release URL
    local os="linux"
    local url="https://github.com/direnv/direnv/releases/latest/download/direnv.${os}-${arch}"

    print_status "Downloading direnv for ${os}-${arch}..."

    # Create temporary directory
    local temp_dir=$(mktemp -d)
    cd "$temp_dir"

    # Download binary
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$url" -o direnv
    elif command -v wget >/dev/null 2>&1; then
        wget -q "$url" -O direnv
    else
        print_error "Neither curl nor wget found. Please install one of them."
        return 1
    fi

    # Make executable and install
    chmod +x direnv

    # Install to /usr/local/bin (requires sudo) or ~/.local/bin (user-local)
    if [ -w /usr/local/bin ]; then
        mv direnv /usr/local/bin/
        print_status "Installed direnv to /usr/local/bin/"
    elif [ "$EUID" -eq 0 ]; then
        mv direnv /usr/local/bin/
        print_status "Installed direnv to /usr/local/bin/"
    else
        mkdir -p ~/.local/bin
        mv direnv ~/.local/bin/
        print_status "Installed direnv to ~/.local/bin/"
        print_warning "Make sure ~/.local/bin is in your PATH"
    fi

    # Clean up
    cd - > /dev/null
    rm -rf "$temp_dir"
}

# Function to setup bash shell hook for direnv
setup_bash_hook() {
    local rc_file="$HOME/.bashrc"

    if [ -f "$rc_file" ]; then
        # Check if hook already exists
        if grep -q "direnv hook" "$rc_file"; then
            print_status "direnv hook already exists in $rc_file"
        else
            print_status "Adding direnv hook to $rc_file"
            echo 'eval "$(direnv hook bash)"' >> "$rc_file"
            print_status "Hook added. Please restart your shell or run: source $rc_file"
        fi
    else
        print_warning "$rc_file not found. Please manually add: eval \"\$(direnv hook bash)\""
    fi
}

# Main function
main() {

    # Check if direnv is already installed
    if command -v direnv >/dev/null 2>&1; then
        return 0
    fi

    print_status "direnv not found. Installing..."

    # Install using apt
    if install_direnv_apt; then
        print_status "direnv installed successfully via apt"
    else
        print_warning "apt installation failed. Trying binary installation..."
        if install_direnv_binary; then
            print_status "direnv installed successfully via binary"
        else
            print_error "Failed to install direnv"
            exit 1
        fi
    fi

    # Verify installation
    if command -v direnv >/dev/null 2>&1; then
        print_status "Installation verified: $(direnv version)"

        # Setup bash hook
        setup_bash_hook

        # Force allow any .envrc in current directory
        if [ -f ".envrc" ]; then
            print_status "Found .envrc file, allowing it automatically..."
            direnv allow
        fi

        print_status "Setup complete!"
        print_status "You can now use direnv in your projects by creating .envrc files"
    else
        print_error "Installation failed - direnv command not found"
        exit 1
    fi
}

# Run main function
main "$@"
