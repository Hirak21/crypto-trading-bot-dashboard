import React, { useState, useEffect } from 'react';
import { Play, Pause, Square, TrendingUp, TrendingDown, Activity } from 'lucide-react';

interface TradingStatus {
  isActive: boolean;
  mode: 'live' | 'paper' | 'backtest';
  totalTrades: number;
  successRate: number;
  dailyPnL: number;
  activePositions: number;
}

interface MarketData {
  symbol: string;
  price: number;
  change24h: number;
  volume: number;
  signal?: 'BUY' | 'SELL' | 'HOLD';
  confidence?: number;
}

const Trading: React.FC = () => {
  const [tradingStatus, setTradingStatus] = useState<TradingStatus>({
    isActive: false,
    mode: 'paper',
    totalTrades: 0,
    successRate: 0,
    dailyPnL: 0,
    activePositions: 0
  });

  const [marketData, setMarketData] = useState<MarketData[]>([
    { symbol: 'BTCUSDT', price: 43250.50, change24h: 2.45, volume: 1250000000, signal: 'BUY', confidence: 0.75 },
    { symbol: 'ETHUSDT', price: 2650.25, change24h: -1.20, volume: 850000000, signal: 'HOLD', confidence: 0.60 },
    { symbol: 'ADAUSDT', price: 0.485, change24h: 3.80, volume: 120000000, signal: 'SELL', confidence: 0.80 },
  ]);

  const handleStartTrading = () => {
    setTradingStatus(prev => ({ ...prev, isActive: true }));
  };

  const handleStopTrading = () => {
    setTradingStatus(prev => ({ ...prev, isActive: false }));
  };

  const handleModeChange = (mode: 'live' | 'paper' | 'backtest') => {
    setTradingStatus(prev => ({ ...prev, mode }));
  };

  const getSignalColor = (signal?: string) => {
    switch (signal) {
      case 'BUY': return 'text-success-600 bg-success-50';
      case 'SELL': return 'text-danger-600 bg-danger-50';
      case 'HOLD': return 'text-warning-600 bg-warning-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-gray-900">Trading Control</h1>
        <div className="flex items-center space-x-4">
          <div className={`status-indicator ${tradingStatus.isActive ? 'status-online' : 'status-offline'}`}>
            <Activity className="w-4 h-4 mr-1" />
            {tradingStatus.isActive ? 'Active' : 'Inactive'}
          </div>
        </div>
      </div>

      {/* Trading Controls */}
      <div className="card">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-xl font-semibold">Trading Controls</h2>
          <div className="flex space-x-2">
            <button
              onClick={() => handleModeChange('paper')}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                tradingStatus.mode === 'paper' 
                  ? 'bg-primary-600 text-white' 
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              Paper Trading
            </button>
            <button
              onClick={() => handleModeChange('live')}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                tradingStatus.mode === 'live' 
                  ? 'bg-danger-600 text-white' 
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              Live Trading
            </button>
            <button
              onClick={() => handleModeChange('backtest')}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                tradingStatus.mode === 'backtest' 
                  ? 'bg-warning-600 text-white' 
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              Backtest
            </button>
          </div>
        </div>

        <div className="flex space-x-4">
          <button
            onClick={handleStartTrading}
            disabled={tradingStatus.isActive}
            className="btn-success flex items-center disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Play className="w-4 h-4 mr-2" />
            Start Trading
          </button>
          <button
            onClick={handleStopTrading}
            disabled={!tradingStatus.isActive}
            className="btn-danger flex items-center disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Pause className="w-4 h-4 mr-2" />
            Pause Trading
          </button>
          <button className="btn-secondary flex items-center">
            <Square className="w-4 h-4 mr-2" />
            Emergency Stop
          </button>
        </div>
      </div>

      {/* Trading Status */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="metric-card">
          <div className="metric-value">{tradingStatus.totalTrades}</div>
          <div className="metric-label">Total Trades</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">{tradingStatus.successRate.toFixed(1)}%</div>
          <div className="metric-label">Success Rate</div>
        </div>
        <div className="metric-card">
          <div className={`metric-value ${tradingStatus.dailyPnL >= 0 ? 'metric-change-positive' : 'metric-change-negative'}`}>
            ${tradingStatus.dailyPnL.toFixed(2)}
          </div>
          <div className="metric-label">Daily P&L</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">{tradingStatus.activePositions}</div>
          <div className="metric-label">Active Positions</div>
        </div>
      </div>

      {/* Market Signals */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-6">Market Signals</h2>
        <div className="table-container">
          <table className="table">
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Price</th>
                <th>24h Change</th>
                <th>Volume</th>
                <th>Signal</th>
                <th>Confidence</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {marketData.map((item) => (
                <tr key={item.symbol}>
                  <td className="font-medium">{item.symbol}</td>
                  <td>${item.price.toLocaleString()}</td>
                  <td className={item.change24h >= 0 ? 'metric-change-positive' : 'metric-change-negative'}>
                    {item.change24h >= 0 ? '+' : ''}{item.change24h.toFixed(2)}%
                  </td>
                  <td>${(item.volume / 1000000).toFixed(1)}M</td>
                  <td>
                    {item.signal && (
                      <span className={`status-indicator ${getSignalColor(item.signal)}`}>
                        {item.signal === 'BUY' && <TrendingUp className="w-3 h-3 mr-1" />}
                        {item.signal === 'SELL' && <TrendingDown className="w-3 h-3 mr-1" />}
                        {item.signal}
                      </span>
                    )}
                  </td>
                  <td>
                    {item.confidence && (
                      <div className="flex items-center">
                        <div className="w-16 bg-gray-200 rounded-full h-2 mr-2">
                          <div 
                            className="bg-primary-600 h-2 rounded-full" 
                            style={{ width: `${item.confidence * 100}%` }}
                          ></div>
                        </div>
                        <span className="text-sm text-gray-600">
                          {(item.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    )}
                  </td>
                  <td>
                    <button className="btn-primary text-sm py-1 px-3">
                      Execute
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Recent Trades */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-6">Recent Trades</h2>
        <div className="space-y-3">
          {[
            { symbol: 'BTCUSDT', side: 'BUY', amount: 0.025, price: 43180.50, time: '2 minutes ago', pnl: '+$125.50' },
            { symbol: 'ETHUSDT', side: 'SELL', amount: 1.5, price: 2655.25, time: '15 minutes ago', pnl: '-$45.20' },
            { symbol: 'ADAUSDT', side: 'BUY', amount: 2000, price: 0.482, time: '1 hour ago', pnl: '+$78.30' },
          ].map((trade, index) => (
            <div key={index} className="flex justify-between items-center p-4 bg-gray-50 rounded-lg">
              <div className="flex items-center space-x-4">
                <span className={`status-indicator ${trade.side === 'BUY' ? 'status-online' : 'status-offline'}`}>
                  {trade.side}
                </span>
                <div>
                  <div className="font-medium">{trade.symbol}</div>
                  <div className="text-sm text-gray-500">{trade.amount} @ ${trade.price}</div>
                </div>
              </div>
              <div className="text-right">
                <div className={`font-medium ${trade.pnl.startsWith('+') ? 'metric-change-positive' : 'metric-change-negative'}`}>
                  {trade.pnl}
                </div>
                <div className="text-sm text-gray-500">{trade.time}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Trading;