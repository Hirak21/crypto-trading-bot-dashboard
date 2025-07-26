// Configuration for the dashboard
export const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
export const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws';

export const REFRESH_INTERVALS = {
  MARKET_DATA: 5000,    // 5 seconds
  PORTFOLIO: 10000,     // 10 seconds
  TRADES: 15000,        // 15 seconds
  STATUS: 5000,         // 5 seconds
};

export const CHART_COLORS = {
  PRIMARY: '#3B82F6',
  SUCCESS: '#10B981',
  WARNING: '#F59E0B',
  DANGER: '#EF4444',
  SECONDARY: '#6B7280',
};

export const SYMBOLS = [
  'BTCUSDT',
  'ETHUSDT',
  'ADAUSDT',
  'DOTUSDT',
  'LINKUSDT',
  'BNBUSDT'
];

export const STRATEGIES = [
  'momentum',
  'liquidity',
  'pattern',
  'candlestick'
];