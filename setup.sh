#!/bin/bash

# Crypto Trading Bot Dashboard Setup Script for Ubuntu
# This script helps set up the development environment

set -e

echo "🚀 Crypto Trading Bot Dashboard Setup"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Node.js is installed
check_node() {
    print_status "Checking Node.js installation..."
    
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version)
        print_success "Node.js is installed: $NODE_VERSION"
        
        # Check if version is 16 or higher
        NODE_MAJOR_VERSION=$(echo $NODE_VERSION | cut -d'.' -f1 | sed 's/v//')
        if [ "$NODE_MAJOR_VERSION" -lt 16 ]; then
            print_warning "Node.js version is too old. Please upgrade to v16 or higher."
            return 1
        fi
    else
        print_error "Node.js is not installed!"
        print_status "Installing Node.js..."
        
        # Install Node.js using NodeSource repository
        curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
        sudo apt-get install -y nodejs
        
        print_success "Node.js installed successfully!"
    fi
}

# Check if npm is installed
check_npm() {
    print_status "Checking npm installation..."
    
    if command -v npm &> /dev/null; then
        NPM_VERSION=$(npm --version)
        print_success "npm is installed: v$NPM_VERSION"
    else
        print_error "npm is not installed!"
        return 1
    fi
}

# Install dashboard dependencies
install_dependencies() {
    print_status "Installing dashboard dependencies..."
    
    if [ -d "dashboard" ]; then
        cd dashboard
        
        if [ -f "package.json" ]; then
            npm install
            print_success "Dependencies installed successfully!"
        else
            print_error "package.json not found in dashboard directory!"
            return 1
        fi
        
        cd ..
    else
        print_error "Dashboard directory not found!"
        return 1
    fi
}

# Create environment file
setup_environment() {
    print_status "Setting up environment configuration..."
    
    if [ -f "dashboard/.env.example" ] && [ ! -f "dashboard/.env" ]; then
        cp dashboard/.env.example dashboard/.env
        print_success "Environment file created from example"
    else
        print_warning "Environment file already exists or example not found"
    fi
}

# Main setup function
main() {
    echo
    print_status "Starting setup process..."
    echo
    
    # Check prerequisites
    check_node || exit 1
    check_npm || exit 1
    
    echo
    
    # Install dependencies
    install_dependencies || exit 1
    
    # Setup environment
    setup_environment
    
    echo
    print_success "Setup completed successfully! 🎉"
    echo
    print_status "To start the development server:"
    echo -e "  ${YELLOW}cd dashboard${NC}"
    echo -e "  ${YELLOW}npm start${NC}"
    echo
    print_status "The application will be available at: http://localhost:3000"
    echo
}

# Run main function
main