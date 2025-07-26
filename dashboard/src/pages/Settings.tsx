import React, { useState } from 'react';
import { Save, RefreshCw, AlertTriangle, Key, Shield, Bell } from 'lucide-react';

interface Settings {
  api: {
    binanceApiKey: string;
    binanceApiSecret: string;
    testnet: boolean;
  };
  risk: {
    maxPositionSize: number;
    dailyLossLimit: number;
    maxDrawdown: number;
    stopLossPct: number;
    takeProfitPct: number;
  };
  trading: {
    tradingEnabled: boolean;
    dryRun: boolean;
    autoStart: boolean;
  };
  notifications: {
    emailEnabled: boolean;
    webhookEnabled: boolean;
    consoleEnabled: boolean;
    tradeNotifications: boolean;
    errorNotifications: boolean;
  };
}

const Settings: React.FC = () => {
  const [settings, setSettings] = useState<Settings>({
    api: {
      binanceApiKey: '',
      binanceApiSecret: '',
      testnet: true,
    },
    risk: {
      maxPositionSize: 2.0,
      dailyLossLimit: 5.0,
      maxDrawdown: 15.0,
      stopLossPct: 2.0,
      takeProfitPct: 4.0,
    },
    trading: {
      tradingEnabled: false,
      dryRun: true,
      autoStart: false,
    },
    notifications: {
      emailEnabled: false,
      webhookEnabled: false,
      consoleEnabled: true,
      tradeNotifications: true,
      errorNotifications: true,
    },
  });

  const [activeTab, setActiveTab] = useState<'api' | 'risk' | 'trading' | 'notifications'>('api');
  const [isSaving, setIsSaving] = useState(false);

  const handleSave = async () => {
    setIsSaving(true);
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000));
    setIsSaving(false);
    // Show success message
  };

  const handleReset = () => {
    // Reset to default values
    setSettings({
      api: {
        binanceApiKey: '',
        binanceApiSecret: '',
        testnet: true,
      },
      risk: {
        maxPositionSize: 2.0,
        dailyLossLimit: 5.0,
        maxDrawdown: 15.0,
        stopLossPct: 2.0,
        takeProfitPct: 4.0,
      },
      trading: {
        tradingEnabled: false,
        dryRun: true,
        autoStart: false,
      },
      notifications: {
        emailEnabled: false,
        webhookEnabled: false,
        consoleEnabled: true,
        tradeNotifications: true,
        errorNotifications: true,
      },
    });
  };

  const updateSettings = (section: keyof Settings, field: string, value: any) => {
    setSettings(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        [field]: value,
      },
    }));
  };

  const tabs = [
    { id: 'api', label: 'API Configuration', icon: Key },
    { id: 'risk', label: 'Risk Management', icon: Shield },
    { id: 'trading', label: 'Trading Settings', icon: RefreshCw },
    { id: 'notifications', label: 'Notifications', icon: Bell },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-gray-900">Settings</h1>
        <div className="flex space-x-3">
          <button onClick={handleReset} className="btn-secondary">
            <RefreshCw className="w-4 h-4 mr-2" />
            Reset
          </button>
          <button 
            onClick={handleSave} 
            disabled={isSaving}
            className="btn-primary flex items-center"
          >
            {isSaving ? (
              <div className="loading-spinner mr-2"></div>
            ) : (
              <Save className="w-4 h-4 mr-2" />
            )}
            Save Changes
          </button>
        </div>
      </div>

      {/* Warning Banner */}
      <div className="alert alert-warning">
        <AlertTriangle className="w-5 h-5 mr-2" />
        <div>
          <strong>Important:</strong> Changes to these settings will affect live trading. 
          Always test in paper trading mode before enabling live trading.
        </div>
      </div>

      <div className="flex space-x-6">
        {/* Sidebar Navigation */}
        <div className="w-64 space-y-1">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`w-full flex items-center px-4 py-3 text-left rounded-lg transition-colors ${
                  activeTab === tab.id
                    ? 'bg-primary-50 text-primary-700 border-l-4 border-primary-600'
                    : 'text-gray-600 hover:bg-gray-50'
                }`}
              >
                <Icon className="w-5 h-5 mr-3" />
                {tab.label}
              </button>
            );
          })}
        </div>

        {/* Settings Content */}
        <div className="flex-1">
          <div className="card">
            {/* API Configuration */}
            {activeTab === 'api' && (
              <div className="space-y-6">
                <h2 className="text-xl font-semibold">API Configuration</h2>
                
                <div className="space-y-4">
                  <div>
                    <label className="form-label">Binance API Key</label>
                    <input
                      type="password"
                      value={settings.api.binanceApiKey}
                      onChange={(e) => updateSettings('api', 'binanceApiKey', e.target.value)}
                      className="form-input"
                      placeholder="Enter your Binance API key"
                    />
                  </div>
                  
                  <div>
                    <label className="form-label">Binance API Secret</label>
                    <input
                      type="password"
                      value={settings.api.binanceApiSecret}
                      onChange={(e) => updateSettings('api', 'binanceApiSecret', e.target.value)}
                      className="form-input"
                      placeholder="Enter your Binance API secret"
                    />
                  </div>
                  
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      id="testnet"
                      checked={settings.api.testnet}
                      onChange={(e) => updateSettings('api', 'testnet', e.target.checked)}
                      className="mr-3"
                    />
                    <label htmlFor="testnet" className="form-label mb-0">
                      Use Testnet (Recommended for testing)
                    </label>
                  </div>
                </div>
              </div>
            )}

            {/* Risk Management */}
            {activeTab === 'risk' && (
              <div className="space-y-6">
                <h2 className="text-xl font-semibold">Risk Management</h2>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label className="form-label">Max Position Size (%)</label>
                    <input
                      type="number"
                      step="0.1"
                      value={settings.risk.maxPositionSize}
                      onChange={(e) => updateSettings('risk', 'maxPositionSize', parseFloat(e.target.value))}
                      className="form-input"
                    />
                    <p className="text-sm text-gray-500 mt-1">Maximum percentage of portfolio per trade</p>
                  </div>
                  
                  <div>
                    <label className="form-label">Daily Loss Limit (%)</label>
                    <input
                      type="number"
                      step="0.1"
                      value={settings.risk.dailyLossLimit}
                      onChange={(e) => updateSettings('risk', 'dailyLossLimit', parseFloat(e.target.value))}
                      className="form-input"
                    />
                    <p className="text-sm text-gray-500 mt-1">Maximum daily loss as percentage of portfolio</p>
                  </div>
                  
                  <div>
                    <label className="form-label">Max Drawdown (%)</label>
                    <input
                      type="number"
                      step="0.1"
                      value={settings.risk.maxDrawdown}
                      onChange={(e) => updateSettings('risk', 'maxDrawdown', parseFloat(e.target.value))}
                      className="form-input"
                    />
                    <p className="text-sm text-gray-500 mt-1">Maximum portfolio drawdown before stopping</p>
                  </div>
                  
                  <div>
                    <label className="form-label">Stop Loss (%)</label>
                    <input
                      type="number"
                      step="0.1"
                      value={settings.risk.stopLossPct}
                      onChange={(e) => updateSettings('risk', 'stopLossPct', parseFloat(e.target.value))}
                      className="form-input"
                    />
                    <p className="text-sm text-gray-500 mt-1">Default stop loss percentage</p>
                  </div>
                  
                  <div>
                    <label className="form-label">Take Profit (%)</label>
                    <input
                      type="number"
                      step="0.1"
                      value={settings.risk.takeProfitPct}
                      onChange={(e) => updateSettings('risk', 'takeProfitPct', parseFloat(e.target.value))}
                      className="form-input"
                    />
                    <p className="text-sm text-gray-500 mt-1">Default take profit percentage</p>
                  </div>
                </div>
              </div>
            )}

            {/* Trading Settings */}
            {activeTab === 'trading' && (
              <div className="space-y-6">
                <h2 className="text-xl font-semibold">Trading Settings</h2>
                
                <div className="space-y-4">
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      id="tradingEnabled"
                      checked={settings.trading.tradingEnabled}
                      onChange={(e) => updateSettings('trading', 'tradingEnabled', e.target.checked)}
                      className="mr-3"
                    />
                    <label htmlFor="tradingEnabled" className="form-label mb-0">
                      Enable Trading
                    </label>
                  </div>
                  
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      id="dryRun"
                      checked={settings.trading.dryRun}
                      onChange={(e) => updateSettings('trading', 'dryRun', e.target.checked)}
                      className="mr-3"
                    />
                    <label htmlFor="dryRun" className="form-label mb-0">
                      Dry Run Mode (Paper Trading)
                    </label>
                  </div>
                  
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      id="autoStart"
                      checked={settings.trading.autoStart}
                      onChange={(e) => updateSettings('trading', 'autoStart', e.target.checked)}
                      className="mr-3"
                    />
                    <label htmlFor="autoStart" className="form-label mb-0">
                      Auto-start trading on bot startup
                    </label>
                  </div>
                </div>
              </div>
            )}

            {/* Notifications */}
            {activeTab === 'notifications' && (
              <div className="space-y-6">
                <h2 className="text-xl font-semibold">Notification Settings</h2>
                
                <div className="space-y-4">
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      id="emailEnabled"
                      checked={settings.notifications.emailEnabled}
                      onChange={(e) => updateSettings('notifications', 'emailEnabled', e.target.checked)}
                      className="mr-3"
                    />
                    <label htmlFor="emailEnabled" className="form-label mb-0">
                      Email Notifications
                    </label>
                  </div>
                  
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      id="webhookEnabled"
                      checked={settings.notifications.webhookEnabled}
                      onChange={(e) => updateSettings('notifications', 'webhookEnabled', e.target.checked)}
                      className="mr-3"
                    />
                    <label htmlFor="webhookEnabled" className="form-label mb-0">
                      Webhook Notifications
                    </label>
                  </div>
                  
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      id="consoleEnabled"
                      checked={settings.notifications.consoleEnabled}
                      onChange={(e) => updateSettings('notifications', 'consoleEnabled', e.target.checked)}
                      className="mr-3"
                    />
                    <label htmlFor="consoleEnabled" className="form-label mb-0">
                      Console Notifications
                    </label>
                  </div>
                  
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      id="tradeNotifications"
                      checked={settings.notifications.tradeNotifications}
                      onChange={(e) => updateSettings('notifications', 'tradeNotifications', e.target.checked)}
                      className="mr-3"
                    />
                    <label htmlFor="tradeNotifications" className="form-label mb-0">
                      Trade Notifications
                    </label>
                  </div>
                  
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      id="errorNotifications"
                      checked={settings.notifications.errorNotifications}
                      onChange={(e) => updateSettings('notifications', 'errorNotifications', e.target.checked)}
                      className="mr-3"
                    />
                    <label htmlFor="errorNotifications" className="form-label mb-0">
                      Error Notifications
                    </label>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Settings;