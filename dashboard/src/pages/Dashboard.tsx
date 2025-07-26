import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { TrendingUp, TrendingDown, DollarSign, BarChart3 } from 'lucide-react';

interface DashboardStats {
  totalPnL: number;
  dailyPnL: number;
  totalTrades: number;
  winRate: number;
  activePositions: number;
  portfolioValue: number;
}

interface PriceData {
  time: string;
  price: number;
}

const Dashboard: React.FC = () => {
  const [stats] = useState<DashboardStats>({
    totalPnL: 1250.75,
    dailyPnL: 85.30,
    totalTrades: 147,
    winRate: 68.5,
    activePositions: 3,
    portfolioValue: 12450.75,
  });

  const [priceData, setPriceData] = useState<PriceData[]>([]);
  const [botStatus] = useState<'running' | 'stopped' | 'error'>('running');

  // Mock data for price chart
  useEffect(() => {
    const generateMockData = (): PriceData[] => {
      const data: PriceData[] = [];
      const now = new Date();
      for (let i = 23; i >= 0; i--) {
        const time = new Date(now.getTime() - i * 60 * 60 * 1000);
        data.push({
          time: time.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
          price: 50000 + Math.random() * 2000 - 1000,
        });
      }
      return data;
    };

    setPriceData(generateMockData());
  }, []);

  const StatCard: React.FC<{
    title: string;
    value: string | number;
    change?: number;
    icon: React.ReactNode;
    colorClass?: string;
  }> = ({ title, value, change, icon, colorClass = 'text-emerald-400' }) => (
    <div className="bg-gradient-to-br from-slate-800 to-slate-700 p-6 rounded-lg shadow-lg">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-gray-400 text-sm mb-1">{title}</p>
          <p className={`text-2xl font-bold ${colorClass}`}>{value}</p>
          {change !== undefined && (
            <div className="flex items-center mt-2">
              {change >= 0 ? (
                <TrendingUp className="w-4 h-4 text-green-500 mr-1" />
              ) : (
                <TrendingDown className="w-4 h-4 text-red-500 mr-1" />
              )}
              <span className={`text-sm ${change >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                {Math.abs(change)}%
              </span>
            </div>
          )}
        </div>
        <div className={`${colorClass} opacity-70`}>
          {icon}
        </div>
      </div>
    </div>
  );

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-white mb-6">Dashboard</h1>

      {/* Status Bar */}
      <div className="bg-gradient-to-r from-slate-800 to-slate-700 p-4 rounded-lg shadow-lg">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <h2 className="text-lg font-semibold text-white">Bot Status:</h2>
            <span className={`px-3 py-1 rounded-full text-sm font-medium border ${
              botStatus === 'running' 
                ? 'bg-green-900 text-green-300 border-green-600' 
                : botStatus === 'error' 
                ? 'bg-red-900 text-red-300 border-red-600' 
                : 'bg-gray-900 text-gray-300 border-gray-600'
            }`}>
              {botStatus.toUpperCase()}
            </span>
          </div>
          <p className="text-gray-400 text-sm">
            Last updated: {new Date().toLocaleTimeString()}
          </p>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        <StatCard
          title="Total P&L"
          value={`$${stats.totalPnL.toFixed(2)}`}
          change={12.5}
          icon={<DollarSign className="w-10 h-10" />}
          colorClass="text-emerald-400"
        />
        <StatCard
          title="Daily P&L"
          value={`$${stats.dailyPnL.toFixed(2)}`}
          change={5.2}
          icon={<BarChart3 className="w-10 h-10" />}
          colorClass="text-blue-400"
        />
        <StatCard
          title="Win Rate"
          value={`${stats.winRate}%`}
          icon={<TrendingUp className="w-10 h-10" />}
          colorClass="text-green-400"
        />
        <StatCard
          title="Total Trades"
          value={stats.totalTrades}
          icon={<BarChart3 className="w-10 h-10" />}
          colorClass="text-orange-400"
        />
        <StatCard
          title="Active Positions"
          value={stats.activePositions}
          icon={<DollarSign className="w-10 h-10" />}
          colorClass="text-purple-400"
        />
        <StatCard
          title="Portfolio Value"
          value={`$${stats.portfolioValue.toFixed(2)}`}
          change={8.7}
          icon={<TrendingUp className="w-10 h-10" />}
          colorClass="text-emerald-400"
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <div className="bg-gradient-to-br from-slate-800 to-slate-700 p-6 rounded-lg shadow-lg">
            <h3 className="text-xl font-semibold text-white mb-4">
              Portfolio Performance (24h)
            </h3>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={priceData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
                  <XAxis dataKey="time" stroke="#94a3b8" />
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
                    dataKey="price"
                    stroke="#10b981"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        <div>
          <div className="bg-gradient-to-br from-slate-800 to-slate-700 p-6 rounded-lg shadow-lg">
            <h3 className="text-xl font-semibold text-white mb-4">
              Strategy Performance
            </h3>
            <div className="space-y-4">
              {[
                { name: 'Momentum Strategy', performance: 85, color: 'bg-emerald-500' },
                { name: 'Liquidity Strategy', performance: 72, color: 'bg-blue-500' },
                { name: 'Pattern Strategy', performance: 68, color: 'bg-orange-500' },
                { name: 'Candlestick Strategy', performance: 91, color: 'bg-green-500' },
              ].map((strategy) => (
                <div key={strategy.name} className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-300 text-sm">{strategy.name}</span>
                    <span className="text-white font-medium">
                      {strategy.performance}%
                    </span>
                  </div>
                  <div className="w-full bg-slate-600 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full ${strategy.color}`}
                      style={{ width: `${strategy.performance}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;