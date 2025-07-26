# Setup Guide for Ubuntu

This guide will help you set up the Crypto Trading Bot Dashboard on Ubuntu.

## Prerequisites

### 1. Install Node.js and npm

```bash
# Method 1: Using NodeSource repository (Recommended)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Method 2: Using Ubuntu package manager
sudo apt update
sudo apt install nodejs npm

# Method 3: Using snap
sudo snap install node --classic
```

Verify installation:
```bash
node --version  # Should be v16+ 
npm --version   # Should be v8+
```

### 2. Install Git (if not already installed)

```bash
sudo apt update
sudo apt install git
```

## Installation Steps

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd crypto-trading-bot-dashboard
```

### 2. Navigate to Dashboard Directory

```bash
cd dashboard
```

### 3. Install Dependencies

```bash
npm install
```

This will install all required packages:
- React and TypeScript
- Tailwind CSS for styling
- Recharts for data visualization
- Lucide React for icons
- And other dependencies

### 4. Configure Environment (Optional)

```bash
# Copy example environment file
cp .env.example .env

# Edit configuration if needed
nano .env
```

### 5. Start Development Server

```bash
npm start
```

The application will:
- Start on `http://localhost:3000`
- Automatically open in your default browser
- Enable hot reloading for development

## Troubleshooting

### Common Issues

#### 1. Port Already in Use
```bash
# Use a different port
npm start -- --port 3001
```

#### 2. Permission Issues
```bash
# Fix npm permissions
sudo chown -R $USER:$USER ~/.npm
sudo chown -R $USER:$USER ~/.config
```

#### 3. Node Version Issues
```bash
# Check Node version
node --version

# If version is too old, update Node.js
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
```

#### 4. Build Errors
```bash
# Clear npm cache
npm cache clean --force

# Remove node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

#### 5. Memory Issues
```bash
# Increase Node.js memory limit
export NODE_OPTIONS="--max-old-space-size=4096"
npm start
```

### System Requirements

- **OS**: Ubuntu 18.04+ (or other Linux distributions)
- **Node.js**: v16.0.0 or higher
- **npm**: v8.0.0 or higher
- **RAM**: 2GB minimum, 4GB recommended
- **Storage**: 500MB for dependencies

### Development Tools (Optional)

Install useful development tools:

```bash
# VS Code
sudo snap install code --classic

# Chrome/Chromium for debugging
sudo apt install chromium-browser

# Git GUI (optional)
sudo apt install gitg
```

## Production Build

To create a production build:

```bash
npm run build
```

This creates an optimized build in the `build/` directory.

## Next Steps

1. **Customize Configuration**: Edit `src/config.ts` for your needs
2. **Connect Backend**: Update API endpoints in environment variables
3. **Add Features**: Extend components in `src/components/` and `src/pages/`
4. **Deploy**: Use the production build for deployment

## Getting Help

- Check the main README.md for feature documentation
- Review component code in `src/` directory
- Create issues on GitHub for bugs or questions
- Check browser console for error messages

---

Happy coding! 🚀