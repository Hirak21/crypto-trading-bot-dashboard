#!/usr/bin/env python3
"""
FastAPI server for the crypto trading bot dashboard.

This server provides REST API endpoints for the React dashboard
to interact with the trading bot.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import json

# Import trading bot components
try:
    from crypto_trading_bot.main import TradingBotApplication
    from crypto_trading_bot.models.trading import TradingSignal, MarketData, Trade
    from crypto_trading_bot.utils.config import ConfigManager
    from tests.test_mock_data import MockDataGenerator
    TRADING_BOT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Trading bot components not available: {e}")
    TRADING_BOT_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(
    title="Crypto Trading Bot API",
    description="REST API for crypto trading bot dashboard",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
trading_bot: Optional[Any] = None
websocket_connections: List[WebSocket] = []

# Mock data generator (fallback if trading bot not available)
if TRADING_BOT_AVAILABLE:
    mock_generator = MockDataGenerator()

# Mock data for demonstration
mock_portfolio = {
    "total_value": 10250.75,
    "available_balance": 8500.25,
    "unrealized_pnl": 125.50,
    "daily_pnl": 75.25,
    "positions": [
        {
            "symbol": "BTCUSDT",
            "side": "LONG",
            "size": 0.025,
            "entry_price": 43180.50,
            "current_price": 43250.75,
            "unrealized_pnl": 1.76,
            "pnl_percentage": 0.16
        },
        {
            "symbol": "ETHUSDT", 
            "side": "LONG",
            "size": 1.5,
            "entry_price": 2655.25,
            "current_price": 2680.50,
            "unrealized_pnl": 37.88,
            "pnl_percentage": 0.95
        }
    ]
}

mock_strategies = [
    {
        "name": "Liquidity Strategy",
        "enabled": True,
        "performance": {
            "total_trades": 45,
            "win_rate": 68.9,
            "total_pnl": 245.50,
            "avg_trade": 5.46
        }
    },
    {
        "name": "Momentum Strategy", 
        "enabled": True,
        "performance": {
            "total_trades": 32,
            "win_rate": 71.9,
            "total_pnl": 189.25,
            "avg_trade": 5.91
        }
    },
    {
        "name": "Pattern Strategy",
        "enabled": False,
        "performance": {
            "total_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_trade": 0.0
        }
    }
]

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Crypto Trading Bot API", 
        "status": "running",
        "trading_bot_available": TRADING_BOT_AVAILABLE
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "bot_status": "running" if trading_bot else "stopped",
        "trading_bot_available": TRADING_BOT_AVAILABLE
    }

@app.get("/status")
async def get_bot_status():
    """Get trading bot status."""
    if trading_bot and TRADING_BOT_AVAILABLE:
        try:
            status = trading_bot.get_application_status()
            return status
        except Exception as e:
            return {"error": f"Failed to get bot status: {str(e)}"}
    else:
        return {
            "is_running": False,
            "startup_time": None,
            "uptime_seconds": 0,
            "component_status": {},
            "performance_metrics": {
                "total_trades": 0,
                "successful_trades": 0,
                "failed_trades": 0,
                "total_pnl": 0.0
            }
        }

@app.post("/start")
async def start_bot():
    """Start the trading bot."""
    global trading_bot
    
    if not TRADING_BOT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Trading bot components not available")
    
    try:
        if trading_bot and hasattr(trading_bot, 'is_running') and trading_bot.is_running:
            raise HTTPException(status_code=400, detail="Bot is already running")
        
        trading_bot = TradingBotApplication()
        # Start bot in background task
        asyncio.create_task(trading_bot.start())
        
        return {"message": "Trading bot started successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start bot: {str(e)}")

@app.post("/stop")
async def stop_bot():
    """Stop the trading bot."""
    global trading_bot
    
    try:
        if not trading_bot or not hasattr(trading_bot, 'is_running') or not trading_bot.is_running:
            raise HTTPException(status_code=400, detail="Bot is not running")
        
        await trading_bot.shutdown()
        trading_bot = None
        
        return {"message": "Trading bot stopped successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop bot: {str(e)}")

@app.get("/portfolio")
async def get_portfolio():
    """Get portfolio information."""
    if trading_bot and TRADING_BOT_AVAILABLE and hasattr(trading_bot, 'portfolio_manager'):
        try:
            return trading_bot.portfolio_manager.get_portfolio_summary()
        except Exception as e:
            return {"error": f"Failed to get portfolio: {str(e)}"}
    else:
        return mock_portfolio

@app.get("/strategies")
async def get_strategies():
    """Get strategy information."""
    if trading_bot and TRADING_BOT_AVAILABLE and hasattr(trading_bot, 'strategy_manager'):
        try:
            return trading_bot.strategy_manager.get_manager_performance()
        except Exception as e:
            return {"error": f"Failed to get strategies: {str(e)}"}
    else:
        return {"strategies": mock_strategies}

@app.get("/trades")
async def get_trades(limit: int = 50):
    """Get recent trades."""
    if TRADING_BOT_AVAILABLE:
        try:
            # Mock trade data using generator
            trades = []
            for i in range(min(limit, 10)):  # Limit to 10 for demo
                trade_data = {
                    "id": f"trade_{i+1}",
                    "symbol": ["BTCUSDT", "ETHUSDT", "ADAUSDT"][i % 3],
                    "side": ["BUY", "SELL"][i % 2],
                    "size": round(0.01 + (i * 0.005), 4),
                    "price": 43000 + (i * 100),
                    "timestamp": (datetime.now() - timedelta(minutes=i*10)).isoformat(),
                    "pnl": round((-50 + (i * 25)), 2),
                    "strategy": ["momentum", "liquidity", "pattern"][i % 3]
                }
                trades.append(trade_data)
            
            return {"trades": trades}
        except Exception as e:
            return {"error": f"Failed to get trades: {str(e)}"}
    else:
        # Fallback mock data
        trades = [
            {
                "id": "trade_1",
                "symbol": "BTCUSDT",
                "side": "BUY",
                "size": 0.025,
                "price": 43250.50,
                "timestamp": datetime.now().isoformat(),
                "pnl": 125.50,
                "strategy": "momentum"
            }
        ]
        return {"trades": trades}

@app.get("/market-data")
async def get_market_data():
    """Get current market data."""
    symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
    market_data = []
    
    for i, symbol in enumerate(symbols):
        # Generate mock market data
        base_prices = {"BTCUSDT": 43250, "ETHUSDT": 2680, "ADAUSDT": 0.485}
        data = {
            "symbol": symbol,
            "price": base_prices[symbol] + (i * 10),
            "change_24h": round((-2 + (i * 1.5)), 2),
            "volume": 1000000 + (i * 500000),
            "timestamp": datetime.now().isoformat()
        }
        market_data.append(data)
    
    return {"market_data": market_data}

@app.get("/logs")
async def get_logs(level: Optional[str] = None, component: Optional[str] = None, limit: int = 100):
    """Get system logs."""
    # Mock log data
    logs = [
        {
            "id": "1",
            "timestamp": datetime.now().isoformat(),
            "level": "INFO",
            "component": "TradingBot",
            "message": "Trading bot API server started successfully"
        },
        {
            "id": "2", 
            "timestamp": (datetime.now() - timedelta(minutes=5)).isoformat(),
            "level": "WARNING",
            "component": "RiskManager",
            "message": "Position size adjusted due to risk limits"
        },
        {
            "id": "3",
            "timestamp": (datetime.now() - timedelta(minutes=10)).isoformat(),
            "level": "ERROR",
            "component": "TradeManager", 
            "message": "Failed to execute trade: Insufficient balance"
        }
    ]
    
    # Apply filters
    if level:
        logs = [log for log in logs if log["level"] == level]
    if component:
        logs = [log for log in logs if log["component"] == component]
    
    return {"logs": logs[:limit]}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    websocket_connections.append(websocket)
    
    try:
        while True:
            # Send periodic updates
            await asyncio.sleep(5)
            
            # Send market data update
            market_update = {
                "type": "market_data",
                "data": {
                    "BTCUSDT": {"price": 43250.75, "change": 0.25},
                    "ETHUSDT": {"price": 2680.50, "change": -0.15},
                    "ADAUSDT": {"price": 0.485, "change": 1.20}
                },
                "timestamp": datetime.now().isoformat()
            }
            
            await websocket.send_text(json.dumps(market_update))
            
    except WebSocketDisconnect:
        if websocket in websocket_connections:
            websocket_connections.remove(websocket)

async def broadcast_update(message: Dict[str, Any]):
    """Broadcast update to all connected WebSocket clients."""
    if websocket_connections:
        message_str = json.dumps(message)
        for websocket in websocket_connections[:]:  # Copy list to avoid modification during iteration
            try:
                await websocket.send_text(message_str)
            except:
                # Remove disconnected websockets
                if websocket in websocket_connections:
                    websocket_connections.remove(websocket)

@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    print("🚀 Trading Bot API server starting up...")
    print(f"📊 Trading bot components available: {TRADING_BOT_AVAILABLE}")
    print("🌐 API will be available at: http://localhost:8000")
    print("📚 API documentation at: http://localhost:8000/docs")

@app.on_event("shutdown") 
async def shutdown_event():
    """Application shutdown event."""
    global trading_bot
    
    print("🛑 Trading Bot API server shutting down...")
    
    if trading_bot and hasattr(trading_bot, 'is_running') and trading_bot.is_running:
        await trading_bot.shutdown()

if __name__ == "__main__":
    print("🚀 Starting Crypto Trading Bot API Server...")
    print("📊 Dashboard will connect to: http://localhost:8000")
    print("🌐 CORS enabled for: http://localhost:3000")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )