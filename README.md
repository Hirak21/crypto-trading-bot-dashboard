# Crypto Trading Bot Dashboard

A modern React-based dashboard for monitoring and controlling cryptocurrency trading bots. Built with TypeScript, Tailwind CSS, and Recharts for real-time data visualization.

## Features

- 📊 **Real-time Dashboard** - Monitor bot performance, P&L, and trading metrics
- 📈 **Interactive Charts** - Visualize portfolio performance and price movements
- 🎯 **Strategy Management** - Configure and monitor multiple trading strategies
- 💼 **Portfolio Tracking** - Track asset allocation and performance
- 🌙 **Dark Theme** - Optimized for trading environments
- 📱 **Responsive Design** - Works on desktop and mobile devices

## Tech Stack

- **Frontend**: React 18, TypeScript
- **Styling**: Tailwind CSS
- **Charts**: Recharts
- **Icons**: Lucide React
- **Routing**: React Router DOM
- **Build Tool**: Create React App

## Quick Start

### Prerequisites

- Node.js 16+ and npm
- Ubuntu/Linux environment

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd crypto-trading-bot-dashboard
```

2. Navigate to dashboard directory:
```bash
cd dashboard
```

3. Install dependencies:
```bash
npm install
```

4. Start the development server:
```bash
npm start
```

5. Open your browser to `http://localhost:3000`

## Project Structure

```
crypto-trading-bot-dashboard/
├── dashboard/                 # React frontend application
│   ├── public/               # Static assets
│   ├── src/
│   │   ├── components/       # Reusable UI components
│   │   │   ├── Navbar.tsx
│   │   │   └── Sidebar.tsx
│   │   ├── pages/           # Page components
│   │   │   ├── Dashboard.tsx
│   │   │   ├── Portfolio.tsx
│   │   │   └── Strategies.tsx
│   │   ├── App.tsx          # Main app component
│   │   ├── index.tsx        # Entry point
│   │   ├── index.css        # Global styles
│   │   └── config.ts        # Configuration
│   ├── package.json
│   └── tailwind.config.js
├── api_server.py            # Python backend (optional)
├── tests/                   # Test files
└── README.md
```

## Available Scripts

In the `dashboard` directory:

- `npm start` - Start development server
- `npm build` - Build for production
- `npm test` - Run tests
- `npm run eject` - Eject from Create React App

## Configuration

The dashboard can be configured via environment variables:

```bash
# .env file in dashboard directory
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000/ws
```

## Features Overview

### Dashboard
- Bot status monitoring
- Real-time P&L tracking
- Performance metrics
- Interactive price charts

### Portfolio
- Asset allocation visualization
- Portfolio performance tracking
- Balance monitoring

### Strategies
- Strategy configuration
- Performance comparison
- Enable/disable strategies
- Parameter tuning

## Development

### Adding New Components

1. Create component in `src/components/` or `src/pages/`
2. Follow TypeScript best practices
3. Use Tailwind CSS for styling
4. Add proper prop types and interfaces

### Styling Guidelines

- Use Tailwind CSS utility classes
- Follow dark theme color scheme
- Maintain responsive design
- Use consistent spacing and typography

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
- Create an issue on GitHub
- Check existing documentation
- Review the code examples

---

Built with ❤️ for the crypto trading community