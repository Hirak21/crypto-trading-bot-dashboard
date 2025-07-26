import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Settings, Play, Square } from 'lucide-react';

interface Strategy {
    id: string;
    name: string;
    enabled: boolean;
    confidence: number;
    totalTrades: number;
    winRate: number;
    totalPnl: number;
    avgWin: number;
    avgLoss: number;
    maxDrawdown: number;
    sharpeRatio: number;
    description: string;
    parameters: Record<string, any>;
}

interface StrategyPerformance {
    date: string;
    pnl: number;
    trades: number;
    winRate: number;
}

const Strategies: React.FC = () => {
    const [strategies, setStrategies] = useState<Strategy[]>([]);
    const [selectedStrategy, setSelectedStrategy] = useState<Strategy | null>(null);
    const [configDialogOpen, setConfigDialogOpen] = useState(false);
    const [performanceData, setPerformanceData] = useState<StrategyPerformance[]>([]);

    useEffect(() => {
        fetchStrategies();
        fetchPerformanceData();
    }, []);

    const fetchStrategies = async () => {
        try {
            // Mock data for demonstration
            const mockStrategies: Strategy[] = [
                {
                    id: 'momentum',
                    name: 'Momentum Strategy',
                    enabled: true,
                    confidence: 0.75,
                    totalTrades: 45,
                    winRate: 0.72,
                    totalPnl: 234.50,
                    avgWin: 15.30,
                    avgLoss: -8.20,
                    maxDrawdown: 0.12,
                    sharpeRatio: 1.85,
                    description: 'Trades based on price momentum and RSI indicators',
                    parameters: {
                        rsiPeriod: 14,
                        rsiOverbought: 70,
                        rsiOversold: 30,
                        momentumThreshold: 0.6,
                    },
                },
                {
                    id: 'liquidity',
                    name: 'Liquidity Strategy',
                    enabled: true,
                    confidence: 0.68,
                    totalTrades: 32,
                    winRate: 0.65,
                    totalPnl: 156.25,
                    avgWin: 12.80,
                    avgLoss: -6.90,
                    maxDrawdown: 0.08,
                    sharpeRatio: 1.42,
                    description: 'Exploits liquidity imbalances in the order book',
                    parameters: {
                        minLiquidityThreshold: 0.6,
                        highLiquidityThreshold: 0.8,
                        volumeThreshold: 1000000,
                        spreadThreshold: 0.005,
                    },
                },
                {
                    id: 'pattern',
                    name: 'Pattern Strategy',
                    enabled: false,
                    confidence: 0.45,
                    totalTrades: 12,
                    winRate: 0.58,
                    totalPnl: -23.75,
                    avgWin: 18.50,
                    avgLoss: -12.30,
                    maxDrawdown: 0.18,
                    sharpeRatio: 0.65,
                    description: 'Identifies chart patterns and breakouts',
                    parameters: {
                        minPatternConfidence: 0.7,
                        lookbackPeriods: 50,
                        breakoutConfirmation: true,
                        volumeConfirmation: true,
                    },
                },
                {
                    id: 'candlestick',
                    name: 'Candlestick Strategy',
                    enabled: true,
                    confidence: 0.62,
                    totalTrades: 28,
                    winRate: 0.61,
                    totalPnl: 89.40,
                    avgWin: 11.20,
                    avgLoss: -7.80,
                    maxDrawdown: 0.10,
                    sharpeRatio: 1.15,
                    description: 'Trades based on candlestick patterns',
                    parameters: {
                        minPatternConfidence: 0.6,
                        volumeConfirmation: true,
                        lookbackPeriods: 20,
                        fallbackThreshold: 0.6,
                    },
                },
            ];

            setStrategies(mockStrategies);
        } catch (error) {
            console.error('Error fetching strategies:', error);
        }
    };

    const fetchPerformanceData = () => {
        // Mock performance data
        const mockData: StrategyPerformance[] = [];
        for (let i = 30; i >= 0; i--) {
            const date = new Date();
            date.setDate(date.getDate() - i);
            mockData.push({
                date: date.toISOString().split('T')[0],
                pnl: Math.random() * 100 - 20,
                trades: Math.floor(Math.random() * 10),
                winRate: 0.5 + Math.random() * 0.4,
            });
        }
        setPerformanceData(mockData);
    };

    const handleToggleStrategy = async (strategyId: string) => {
        try {
            setStrategies(prev =>
                prev.map(strategy =>
                    strategy.id === strategyId
                        ? { ...strategy, enabled: !strategy.enabled }
                        : strategy
                )
            );
        } catch (error) {
            console.error('Error toggling strategy:', error);
        }
    };

    const handleConfigureStrategy = (strategy: Strategy) => {
        setSelectedStrategy(strategy);
        setConfigDialogOpen(true);
    };

    const handleSaveConfiguration = async () => {
        if (!selectedStrategy) return;

        try {
            console.log('Saving configuration for:', selectedStrategy.name);
            setConfigDialogOpen(false);
            setSelectedStrategy(null);
        } catch (error) {
            console.error('Error saving configuration:', error);
        }
    };

    const formatCurrency = (value: number) => {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
        }).format(value);
    };

    const formatPercentage = (value: number) => {
        return `${(value * 100).toFixed(1)}%`;
    };

    return (
        <div className="space-y-6">
            <h1 className="text-3xl font-bold text-white">Trading Strategies</h1>

            {/* Strategy Overview */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <div className="bg-gradient-to-br from-slate-800 to-slate-700 p-6 rounded-lg shadow-lg">
                    <p className="text-3xl font-bold text-blue-400">
                        {strategies.filter(s => s.enabled).length}
                    </p>
                    <p className="text-gray-400 text-sm">Active Strategies</p>
                </div>
                
                <div className="bg-gradient-to-br from-slate-800 to-slate-700 p-6 rounded-lg shadow-lg">
                    <p className="text-3xl font-bold text-green-400">
                        {formatCurrency(strategies.reduce((sum, s) => sum + s.totalPnl, 0))}
                    </p>
                    <p className="text-gray-400 text-sm">Total P&L</p>
                </div>
                
                <div className="bg-gradient-to-br from-slate-800 to-slate-700 p-6 rounded-lg shadow-lg">
                    <p className="text-3xl font-bold text-white">
                        {strategies.reduce((sum, s) => sum + s.totalTrades, 0)}
                    </p>
                    <p className="text-gray-400 text-sm">Total Trades</p>
                </div>
                
                <div className="bg-gradient-to-br from-slate-800 to-slate-700 p-6 rounded-lg shadow-lg">
                    <p className="text-3xl font-bold text-white">
                        {formatPercentage(
                            strategies.reduce((sum, s) => sum + s.winRate, 0) / strategies.length
                        )}
                    </p>
                    <p className="text-gray-400 text-sm">Avg Win Rate</p>
                </div>
            </div>

            {/* Performance Chart */}
            <div className="bg-gradient-to-br from-slate-800 to-slate-700 p-6 rounded-lg shadow-lg">
                <h3 className="text-xl font-semibold text-white mb-4">
                    Strategy Performance (30 Days)
                </h3>
                <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={performanceData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
                            <XAxis dataKey="date" stroke="#94a3b8" />
                            <YAxis stroke="#94a3b8" />
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: '#1e293b',
                                    border: '1px solid #475569',
                                    borderRadius: '8px',
                                    color: '#f1f5f9'
                                }}
                            />
                            <Line
                                type="monotone"
                                dataKey="pnl"
                                stroke="#10b981"
                                strokeWidth={2}
                                dot={false}
                            />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Strategy List */}
            <div className="bg-gradient-to-br from-slate-800 to-slate-700 rounded-lg shadow-lg overflow-hidden">
                <div className="p-6">
                    <h3 className="text-xl font-semibold text-white mb-4">Strategy Details</h3>
                </div>
                
                <div className="overflow-x-auto">
                    <table className="w-full">
                        <thead className="bg-slate-700">
                            <tr>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                                    Strategy
                                </th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                                    Status
                                </th>
                                <th className="px-6 py-3 text-right text-xs font-medium text-gray-300 uppercase tracking-wider">
                                    Confidence
                                </th>
                                <th className="px-6 py-3 text-right text-xs font-medium text-gray-300 uppercase tracking-wider">
                                    Trades
                                </th>
                                <th className="px-6 py-3 text-right text-xs font-medium text-gray-300 uppercase tracking-wider">
                                    Win Rate
                                </th>
                                <th className="px-6 py-3 text-right text-xs font-medium text-gray-300 uppercase tracking-wider">
                                    P&L
                                </th>
                                <th className="px-6 py-3 text-right text-xs font-medium text-gray-300 uppercase tracking-wider">
                                    Sharpe
                                </th>
                                <th className="px-6 py-3 text-center text-xs font-medium text-gray-300 uppercase tracking-wider">
                                    Actions
                                </th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-600">
                            {strategies.map((strategy) => (
                                <tr key={strategy.id} className="hover:bg-slate-700/50">
                                    <td className="px-6 py-4">
                                        <div>
                                            <p className="text-sm font-medium text-white">
                                                {strategy.name}
                                            </p>
                                            <p className="text-sm text-gray-400">
                                                {strategy.description}
                                            </p>
                                        </div>
                                    </td>
                                    <td className="px-6 py-4">
                                        <div className="flex items-center">
                                            <button
                                                onClick={() => handleToggleStrategy(strategy.id)}
                                                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                                                    strategy.enabled ? 'bg-emerald-600' : 'bg-gray-600'
                                                }`}
                                            >
                                                <span
                                                    className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                                                        strategy.enabled ? 'translate-x-6' : 'translate-x-1'
                                                    }`}
                                                />
                                            </button>
                                            <span className="ml-2 text-sm text-gray-300">
                                                {strategy.enabled ? 'Active' : 'Inactive'}
                                            </span>
                                        </div>
                                    </td>
                                    <td className="px-6 py-4 text-right">
                                        <div className="flex items-center justify-end">
                                            <div className="w-16 bg-gray-600 rounded-full h-2 mr-2">
                                                <div
                                                    className={`h-2 rounded-full ${
                                                        strategy.confidence > 0.7 
                                                            ? 'bg-green-500' 
                                                            : strategy.confidence > 0.5 
                                                            ? 'bg-yellow-500' 
                                                            : 'bg-red-500'
                                                    }`}
                                                    style={{ width: `${strategy.confidence * 100}%` }}
                                                />
                                            </div>
                                            <span className="text-sm text-white">
                                                {formatPercentage(strategy.confidence)}
                                            </span>
                                        </div>
                                    </td>
                                    <td className="px-6 py-4 text-right text-sm text-white">
                                        {strategy.totalTrades}
                                    </td>
                                    <td className="px-6 py-4 text-right">
                                        <span className={`text-sm ${
                                            strategy.winRate > 0.5 ? 'text-green-400' : 'text-red-400'
                                        }`}>
                                            {formatPercentage(strategy.winRate)}
                                        </span>
                                    </td>
                                    <td className="px-6 py-4 text-right">
                                        <span className={`text-sm ${
                                            strategy.totalPnl >= 0 ? 'text-green-400' : 'text-red-400'
                                        }`}>
                                            {formatCurrency(strategy.totalPnl)}
                                        </span>
                                    </td>
                                    <td className="px-6 py-4 text-right">
                                        <span className={`text-sm ${
                                            strategy.sharpeRatio > 1 ? 'text-green-400' : 'text-red-400'
                                        }`}>
                                            {strategy.sharpeRatio.toFixed(2)}
                                        </span>
                                    </td>
                                    <td className="px-6 py-4 text-center">
                                        <button
                                            onClick={() => handleConfigureStrategy(strategy)}
                                            className="inline-flex items-center px-3 py-1 border border-gray-600 rounded-md text-sm text-gray-300 hover:bg-slate-600 transition-colors"
                                        >
                                            <Settings className="w-4 h-4 mr-1" />
                                            Configure
                                        </button>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* Configuration Modal */}
            {configDialogOpen && selectedStrategy && (
                <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                    <div className="bg-slate-800 rounded-lg p-6 w-full max-w-2xl max-h-[80vh] overflow-y-auto">
                        <h2 className="text-xl font-bold text-white mb-4">
                            Configure {selectedStrategy.name}
                        </h2>
                        
                        <div className="bg-blue-900/20 border border-blue-600 rounded-lg p-4 mb-6">
                            <p className="text-blue-300 text-sm">
                                Adjust the parameters below to optimize the strategy performance.
                                Changes will take effect after saving.
                            </p>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                            {Object.entries(selectedStrategy.parameters).map(([key, value]) => (
                                <div key={key} className="space-y-2">
                                    <label className="block text-sm font-medium text-gray-300">
                                        {key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
                                    </label>
                                    {typeof value === 'number' ? (
                                        <input
                                            type="range"
                                            min={0}
                                            max={key.includes('Period') ? 50 : key.includes('Threshold') ? 1 : 100}
                                            step={key.includes('Threshold') ? 0.01 : 1}
                                            value={value}
                                            onChange={(e) => {
                                                setSelectedStrategy(prev => prev ? {
                                                    ...prev,
                                                    parameters: {
                                                        ...prev.parameters,
                                                        [key]: parseFloat(e.target.value)
                                                    }
                                                } : null);
                                            }}
                                            className="w-full"
                                        />
                                    ) : typeof value === 'boolean' ? (
                                        <label className="flex items-center">
                                            <input
                                                type="checkbox"
                                                checked={value}
                                                onChange={(e) => {
                                                    setSelectedStrategy(prev => prev ? {
                                                        ...prev,
                                                        parameters: {
                                                            ...prev.parameters,
                                                            [key]: e.target.checked
                                                        }
                                                    } : null);
                                                }}
                                                className="mr-2"
                                            />
                                            <span className="text-gray-300">
                                                {key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}
                                            </span>
                                        </label>
                                    ) : (
                                        <input
                                            type="text"
                                            value={value}
                                            onChange={(e) => {
                                                setSelectedStrategy(prev => prev ? {
                                                    ...prev,
                                                    parameters: {
                                                        ...prev.parameters,
                                                        [key]: e.target.value
                                                    }
                                                } : null);
                                            }}
                                            className="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md text-white"
                                        />
                                    )}
                                    {typeof value === 'number' && (
                                        <div className="text-sm text-gray-400">
                                            Current value: {value}
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>

                        <div className="flex justify-end space-x-3">
                            <button
                                onClick={() => setConfigDialogOpen(false)}
                                className="px-4 py-2 text-gray-300 hover:text-white transition-colors"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={handleSaveConfiguration}
                                className="px-4 py-2 bg-emerald-600 text-white rounded-md hover:bg-emerald-700 transition-colors"
                            >
                                Save Configuration
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default Strategies;