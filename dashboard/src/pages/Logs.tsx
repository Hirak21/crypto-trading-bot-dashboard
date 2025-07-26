import React, { useState, useEffect } from 'react';
import { Download, Filter, Search, RefreshCw, AlertCircle, Info, CheckCircle, XCircle } from 'lucide-react';

interface LogEntry {
  id: string;
  timestamp: string;
  level: 'INFO' | 'WARNING' | 'ERROR' | 'DEBUG';
  component: string;
  message: string;
  details?: any;
}

const Logs: React.FC = () => {
  const [logs, setLogs] = useState<LogEntry[]>([
    {
      id: '1',
      timestamp: '2024-01-15 10:30:25',
      level: 'INFO',
      component: 'TradingBot',
      message: 'Trading bot started successfully',
      details: { mode: 'paper', strategies: 3 }
    },
    {
      id: '2',
      timestamp: '2024-01-15 10:30:30',
      level: 'INFO',
      component: 'MarketManager',
      message: 'Connected to Binance WebSocket',
      details: { symbols: ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'] }
    },
    {
      id: '3',
      timestamp: '2024-01-15 10:31:15',
      level: 'WARNING',
      component: 'RiskManager',
      message: 'Position size adjusted due to risk limits',
      details: { symbol: 'BTCUSDT', originalSize: 0.05, adjustedSize: 0.03 }
    },
    {
      id: '4',
      timestamp: '2024-01-15 10:32:00',
      level: 'INFO',
      component: 'StrategyManager',
      message: 'Signal generated: BUY BTCUSDT',
      details: { strategy: 'momentum', confidence: 0.75, price: 43250.50 }
    },
    {
      id: '5',
      timestamp: '2024-01-15 10:32:05',
      level: 'ERROR',
      component: 'TradeManager',
      message: 'Failed to execute trade: Insufficient balance',
      details: { symbol: 'BTCUSDT', requiredBalance: 1000, availableBalance: 850 }
    },
    {
      id: '6',
      timestamp: '2024-01-15 10:33:20',
      level: 'INFO',
      component: 'PortfolioManager',
      message: 'Portfolio updated: +$125.50 unrealized P&L',
      details: { totalValue: 10125.50, positions: 2 }
    },
  ]);

  const [filteredLogs, setFilteredLogs] = useState<LogEntry[]>(logs);
  const [selectedLevel, setSelectedLevel] = useState<string>('ALL');
  const [selectedComponent, setSelectedComponent] = useState<string>('ALL');
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [autoRefresh, setAutoRefresh] = useState<boolean>(true);

  useEffect(() => {
    let filtered = logs;

    if (selectedLevel !== 'ALL') {
      filtered = filtered.filter(log => log.level === selectedLevel);
    }

    if (selectedComponent !== 'ALL') {
      filtered = filtered.filter(log => log.component === selectedComponent);
    }

    if (searchTerm) {
      filtered = filtered.filter(log => 
        log.message.toLowerCase().includes(searchTerm.toLowerCase()) ||
        log.component.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    setFilteredLogs(filtered);
  }, [logs, selectedLevel, selectedComponent, searchTerm]);

  const getLevelIcon = (level: string) => {
    switch (level) {
      case 'INFO': return <Info className="w-4 h-4" />;
      case 'WARNING': return <AlertCircle className="w-4 h-4" />;
      case 'ERROR': return <XCircle className="w-4 h-4" />;
      case 'DEBUG': return <CheckCircle className="w-4 h-4" />;
      default: return <Info className="w-4 h-4" />;
    }
  };

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'INFO': return 'text-primary-600 bg-primary-50';
      case 'WARNING': return 'text-warning-600 bg-warning-50';
      case 'ERROR': return 'text-danger-600 bg-danger-50';
      case 'DEBUG': return 'text-gray-600 bg-gray-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const handleDownloadLogs = () => {
    const logData = filteredLogs.map(log => 
      `${log.timestamp} [${log.level}] ${log.component}: ${log.message}`
    ).join('\n');
    
    const blob = new Blob([logData], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `trading-bot-logs-${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleRefresh = () => {
    // Simulate fetching new logs
    console.log('Refreshing logs...');
  };

  const components = ['ALL', ...Array.from(new Set(logs.map(log => log.component)))];
  const levels = ['ALL', 'INFO', 'WARNING', 'ERROR', 'DEBUG'];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-gray-900">System Logs</h1>
        <div className="flex items-center space-x-3">
          <div className="flex items-center">
            <input
              type="checkbox"
              id="autoRefresh"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="mr-2"
            />
            <label htmlFor="autoRefresh" className="text-sm text-gray-600">
              Auto-refresh
            </label>
          </div>
          <button onClick={handleRefresh} className="btn-secondary">
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </button>
          <button onClick={handleDownloadLogs} className="btn-primary">
            <Download className="w-4 h-4 mr-2" />
            Download
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="card">
        <div className="flex flex-wrap gap-4 items-center">
          <div className="flex items-center space-x-2">
            <Search className="w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search logs..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="form-input w-64"
            />
          </div>
          
          <div className="flex items-center space-x-2">
            <Filter className="w-4 h-4 text-gray-400" />
            <select
              value={selectedLevel}
              onChange={(e) => setSelectedLevel(e.target.value)}
              className="form-input w-32"
            >
              {levels.map(level => (
                <option key={level} value={level}>{level}</option>
              ))}
            </select>
          </div>
          
          <div className="flex items-center space-x-2">
            <select
              value={selectedComponent}
              onChange={(e) => setSelectedComponent(e.target.value)}
              className="form-input w-40"
            >
              {components.map(component => (
                <option key={component} value={component}>{component}</option>
              ))}
            </select>
          </div>
          
          <div className="text-sm text-gray-500">
            Showing {filteredLogs.length} of {logs.length} logs
          </div>
        </div>
      </div>

      {/* Log Entries */}
      <div className="card">
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {filteredLogs.map((log) => (
            <div key={log.id} className="border-l-4 border-gray-200 pl-4 py-2 hover:bg-gray-50 transition-colors">
              <div className="flex items-start justify-between">
                <div className="flex items-start space-x-3 flex-1">
                  <span className={`status-indicator ${getLevelColor(log.level)} flex items-center`}>
                    {getLevelIcon(log.level)}
                    <span className="ml-1">{log.level}</span>
                  </span>
                  
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center space-x-2 mb-1">
                      <span className="font-medium text-gray-900">{log.component}</span>
                      <span className="text-sm text-gray-500">{log.timestamp}</span>
                    </div>
                    <p className="text-gray-700">{log.message}</p>
                    
                    {log.details && (
                      <details className="mt-2">
                        <summary className="text-sm text-primary-600 cursor-pointer hover:text-primary-700">
                          Show details
                        </summary>
                        <pre className="mt-2 p-3 bg-gray-100 rounded text-xs overflow-x-auto">
                          {JSON.stringify(log.details, null, 2)}
                        </pre>
                      </details>
                    )}
                  </div>
                </div>
              </div>
            </div>
          ))}
          
          {filteredLogs.length === 0 && (
            <div className="text-center py-8 text-gray-500">
              <AlertCircle className="w-8 h-8 mx-auto mb-2" />
              <p>No logs found matching your filters.</p>
            </div>
          )}
        </div>
      </div>

      {/* Log Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {levels.slice(1).map(level => {
          const count = logs.filter(log => log.level === level).length;
          return (
            <div key={level} className="metric-card">
              <div className="flex items-center justify-between">
                <div>
                  <div className="metric-value">{count}</div>
                  <div className="metric-label">{level} Logs</div>
                </div>
                <div className={`p-2 rounded-lg ${getLevelColor(level)}`}>
                  {getLevelIcon(level)}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default Logs;