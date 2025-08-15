import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
import datetime
from datetime import timedelta
import ta
from ta.utils import dropna
import requests
import json
from typing import Dict, List, Any
import warnings
import time
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="AI Stock Analysis Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f4037, #99f2c8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .live-indicator {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        padding: 0.5rem;
        border-radius: 5px;
        color: white;
        text-align: center;
        margin: 0.2rem 0;
        font-size: 0.8rem;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .warning-box {
        background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .success-box {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class EnhancedStockAnalyzer:
    def __init__(self):
        self.data = None
        self.ticker = None

    def fetch_stock_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Fetch stock data from Yahoo Finance with better error handling"""
        try:
            stock = yf.Ticker(symbol)

            # For live data, use shorter intervals when possible
            if period in ["1d", "5d"] and interval == "1d":
                interval = "1m" if period == "1d" else "5m"

            data = stock.history(period=period, interval=interval)

            if data.empty:
                # Try alternative periods if main request fails
                fallback_periods = ["3mo", "6mo", "1y", "2y"]
                for fallback_period in fallback_periods:
                    if fallback_period != period:
                        data = stock.history(period=fallback_period)
                        if not data.empty:
                            st.info(f"Using {fallback_period} data instead of {period}")
                            break

                if data.empty:
                    st.error(f"No data found for symbol {symbol}")
                    return None

            return data
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def get_stock_info(self, symbol: str) -> Dict:
        """Get comprehensive stock information"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            return info
        except Exception as e:
            st.error(f"Error getting stock info: {str(e)}")
            return {}

    def calculate_technical_indicators(self, data: pd.DataFrame, min_periods: int = 10) -> pd.DataFrame:
        """Calculate technical indicators with adaptive periods"""
        try:
            df = data.copy()
            data_length = len(df)

            if data_length < min_periods:
                st.warning(f"⚠️ Limited data points ({data_length}). Some indicators may be unavailable.")
                return df

            # Adaptive periods based on available data
            ma_short = min(10, data_length // 2)
            ma_medium = min(20, data_length // 1.5)
            ma_long = min(50, data_length)

            # Moving Averages with adaptive periods
            if data_length >= ma_short:
                df[f'MA{ma_short}'] = ta.trend.sma_indicator(df['Close'], window=ma_short)
            if data_length >= ma_medium:
                df[f'MA{ma_medium}'] = ta.trend.sma_indicator(df['Close'], window=ma_medium)
            if data_length >= ma_long:
                df[f'MA{ma_long}'] = ta.trend.sma_indicator(df['Close'], window=ma_long)

            # RSI with adaptive period
            try:
                rsi_period = min(14, data_length // 2)
                if rsi_period >= 2:
                    df['RSI'] = ta.momentum.rsi(df['Close'], window=rsi_period)
            except:
                pass

            # Bollinger Bands
            try:
                bb_period = min(20, data_length // 2)
                if bb_period >= 5:
                    bollinger = ta.volatility.BollingerBands(df['Close'], window=bb_period)
                    df['BB_Upper'] = bollinger.bollinger_hband()
                    df['BB_Middle'] = bollinger.bollinger_mavg()
                    df['BB_Lower'] = bollinger.bollinger_lband()
            except:
                pass

            # MACD
            try:
                if data_length >= 26:
                    macd = ta.trend.MACD(df['Close'])
                    df['MACD'] = macd.macd()
                    df['MACD_Signal'] = macd.macd_signal()
                    df['MACD_Histogram'] = macd.macd_diff()
                elif data_length >= 12:
                    # Shorter MACD for limited data
                    macd = ta.trend.MACD(df['Close'], window_fast=6, window_slow=12, window_sign=9)
                    df['MACD'] = macd.macd()
                    df['MACD_Signal'] = macd.macd_signal()
                    df['MACD_Histogram'] = macd.macd_diff()
            except:
                pass

            # Stochastic Oscillator
            try:
                stoch_period = min(14, data_length // 2)
                if stoch_period >= 5:
                    stoch = ta.momentum.StochasticOscillator(
                        df['High'], df['Low'], df['Close'],
                        window=stoch_period, smooth_window=3
                    )
                    df['Stoch_K'] = stoch.stoch()
                    df['Stoch_D'] = stoch.stoch_signal()
            except:
                pass

            # Volume indicators
            try:
                vol_period = min(20, data_length // 2)
                if vol_period >= 5:
                    df['Volume_MA'] = df['Volume'].rolling(window=vol_period).mean()
                    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
            except:
                pass

            # ATR (Average True Range)
            try:
                atr_period = min(14, data_length // 2)
                if atr_period >= 2:
                    df['ATR'] = ta.volatility.average_true_range(
                        df['High'], df['Low'], df['Close'], window=atr_period
                    )
            except:
                pass

            return df
        except Exception as e:
            st.error(f"Error calculating technical indicators: {str(e)}")
            return data

def create_live_stock_chart(data: pd.DataFrame, symbol: str, live_mode: bool = False):
    """Create interactive live stock chart"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=[
            f'{symbol} Live Stock Price' if live_mode else f'{symbol} Stock Price',
            'Volume',
            'RSI',
            'MACD'
        ]
    )

    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price',
            increasing_line_color='#00ff00',
            decreasing_line_color='#ff0000'
        ),
        row=1, col=1
    )

    # Add moving averages
    ma_columns = [col for col in data.columns if col.startswith('MA')]
    colors = ['orange', 'blue', 'purple', 'green']

    for i, ma_col in enumerate(ma_columns[:4]):
        if ma_col in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[ma_col],
                    name=ma_col,
                    line=dict(color=colors[i % len(colors)], width=1.5)
                ),
                row=1, col=1
            )

    # Add Bollinger Bands
    if all(col in data.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['BB_Upper'],
                name='BB Upper', line=dict(color='lightgray', width=1),
                showlegend=False
            ), row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['BB_Lower'],
                name='BB Lower', line=dict(color='lightgray', width=1),
                fill='tonexty', fillcolor='rgba(200,200,200,0.2)',
                showlegend=False
            ), row=1, col=1
        )

    # Volume chart
    colors_volume = ['green' if close >= open else 'red'
                    for close, open in zip(data['Close'], data['Open'])]

    fig.add_trace(
        go.Bar(
            x=data.index, y=data['Volume'],
            name='Volume',
            marker_color=colors_volume,
            showlegend=False
        ),
        row=2, col=1
    )

    if 'Volume_MA' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['Volume_MA'],
                name='Vol MA', line=dict(color='yellow', width=2)
            ),
            row=2, col=1
        )

    # RSI
    if 'RSI' in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['RSI'],
                name='RSI', line=dict(color='purple', width=2)
            ),
            row=3, col=1
        )
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1, opacity=0.7)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1, opacity=0.7)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1, opacity=0.5)

    # MACD
    if all(col in data.columns for col in ['MACD', 'MACD_Signal']):
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['MACD'],
                name='MACD', line=dict(color='blue', width=2)
            ),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['MACD_Signal'],
                name='MACD Signal', line=dict(color='red', width=2)
            ),
            row=4, col=1
        )

        if 'MACD_Histogram' in data.columns:
            colors_macd = ['green' if val >= 0 else 'red' for val in data['MACD_Histogram']]
            fig.add_trace(
                go.Bar(
                    x=data.index, y=data['MACD_Histogram'],
                    name='MACD Hist',
                    marker_color=colors_macd,
                    showlegend=False
                ),
                row=4, col=1
            )

    # Update layout
    fig.update_layout(
        title=f'{symbol} Technical Analysis - {"Live Data" if live_mode else "Historical Data"}',
        xaxis_rangeslider_visible=False,
        height=1000,
        showlegend=True,
        template='plotly_dark',
        font=dict(size=10)
    )

    # Update y-axis labels
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=4, col=1)

    return fig

def create_stock_tools():
    """Create tools for the LangChain agent"""
    analyzer = EnhancedStockAnalyzer()

    def get_stock_price(symbol: str) -> str:
        """Get current stock price and basic info"""
        try:
            data = analyzer.fetch_stock_data(symbol, period="5d")
            if data is not None and not data.empty:
                current_price = data['Close'][-1]
                prev_price = data['Close'][-2] if len(data) > 1 else current_price
                change = current_price - prev_price
                change_pct = (change / prev_price) * 100

                return f"Current price of {symbol}: ${current_price:.2f}, Change: ${change:.2f} ({change_pct:.2f}%)"
            return f"Could not fetch data for {symbol}"
        except Exception as e:
            return f"Error getting price for {symbol}: {str(e)}"

    def analyze_stock_technicals(symbol: str) -> str:
        """Analyze stock technical indicators"""
        try:
            data = analyzer.fetch_stock_data(symbol, period="3mo")
            if data is not None and not data.empty:
                df_with_indicators = analyzer.calculate_technical_indicators(data)

                current_price = df_with_indicators['Close'][-1]

                # Find available MA columns
                ma_columns = [col for col in df_with_indicators.columns if col.startswith('MA')]

                analysis = f"Technical Analysis for {symbol}:\n"
                analysis += f"- Current Price: ${current_price:.2f}\n"

                # RSI analysis
                if 'RSI' in df_with_indicators.columns and not pd.isna(df_with_indicators['RSI'][-1]):
                    rsi = df_with_indicators['RSI'][-1]
                    rsi_status = 'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral'
                    analysis += f"- RSI: {rsi:.2f} ({rsi_status})\n"
                else:
                    analysis += "- RSI: Not available\n"

                # Moving average analysis
                for ma_col in sorted(ma_columns):
                    if not pd.isna(df_with_indicators[ma_col][-1]):
                        ma_value = df_with_indicators[ma_col][-1]
                        position = 'Above' if current_price > ma_value else 'Below'
                        analysis += f"- Price vs {ma_col}: {position} (${ma_value:.2f})\n"

                # Trend analysis
                if len(ma_columns) >= 2:
                    short_ma = df_with_indicators[ma_columns[0]][-1]
                    long_ma = df_with_indicators[ma_columns[-1]][-1]
                    if not pd.isna(short_ma) and not pd.isna(long_ma):
                        if current_price > short_ma > long_ma:
                            trend = 'Strong Bullish'
                        elif current_price > short_ma and short_ma > long_ma:
                            trend = 'Bullish'
                        elif current_price < short_ma < long_ma:
                            trend = 'Strong Bearish'
                        elif current_price < short_ma and short_ma < long_ma:
                            trend = 'Bearish'
                        else:
                            trend = 'Mixed/Sideways'
                        analysis += f"- Trend: {trend}\n"

                return analysis
            return f"Could not analyze technicals for {symbol}"
        except Exception as e:
            return f"Error analyzing technicals for {symbol}: {str(e)}"

    def get_stock_fundamentals(symbol: str) -> str:
        """Get fundamental analysis data"""
        try:
            info = analyzer.get_stock_info(symbol)
            if info:
                pe_ratio = info.get('trailingPE', 'N/A')
                market_cap = info.get('marketCap', 'N/A')
                dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 'N/A'
                debt_to_equity = info.get('debtToEquity', 'N/A')

                return f"""Fundamental Analysis for {symbol}:
                - Market Cap: ${market_cap:,} if isinstance(market_cap, (int, float)) else market_cap
                - P/E Ratio: {pe_ratio}
                - Dividend Yield: {dividend_yield}% if dividend_yield != 'N/A' else 'N/A'
                - Debt to Equity: {debt_to_equity}
                - Sector: {info.get('sector', 'N/A')}
                - Industry: {info.get('industry', 'N/A')}
                """
            return f"Could not get fundamentals for {symbol}"
        except Exception as e:
            return f"Error getting fundamentals for {symbol}: {str(e)}"

    tools = [
        Tool(
            name="get_stock_price",
            func=get_stock_price,
            description="Get current stock price and recent change for a given symbol"
        ),
        Tool(
            name="analyze_technicals",
            func=analyze_stock_technicals,
            description="Perform technical analysis including RSI, moving averages, and trend analysis"
        ),
        Tool(
            name="get_fundamentals",
            func=get_stock_fundamentals,
            description="Get fundamental analysis data including P/E ratio, market cap, dividend yield"
        )
    ]

    return tools

def initialize_stock_agent(api_key):
    """Initialize the LangChain agent with Gemini"""
    if not api_key:
        return None

    try:
        # Initialize Gemini model
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.1,
            convert_system_message_to_human=True
        )

        # Create tools
        tools = create_stock_tools()

        # Initialize memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Initialize agent
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            verbose=True,
            max_iterations=3,
            handle_parsing_errors=True
        )

        return agent
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
        return None

def main():
    st.markdown('<h1 class="main-header">🤖 AI Stock Analysis Agent</h1>', unsafe_allow_html=True)

    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    if 'live_data' not in st.session_state:
        st.session_state.live_data = {}
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False

    # Sidebar
    st.sidebar.header("🔧 Configuration")

    # API Key input
    api_key = st.sidebar.text_input(
        "🔑 Enter Google API Key",
        type="password",
        value=st.session_state.api_key,
        help="Get your API key from: https://makersuite.google.com/app/apikey"
    )

    if api_key != st.session_state.api_key:
        st.session_state.api_key = api_key
        if api_key:
            st.session_state.agent = initialize_stock_agent(api_key)
        else:
            st.session_state.agent = None

    # Initialize agent if not exists and API key is provided
    if 'agent' not in st.session_state and api_key:
        st.session_state.agent = initialize_stock_agent(api_key)

    # Stock symbol input
    symbol = st.sidebar.text_input("📊 Enter Stock Symbol", value="AAPL").upper()
    period = st.sidebar.selectbox("📅 Time Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"])

    # Live mode toggle
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔴 Live Mode")
    live_mode = st.sidebar.checkbox("Enable Live Data", help="Updates data automatically every 30 seconds")
    auto_refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 10, 300, 30)

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "💬 AI Chat", "📈 Technical Analysis", "🔴 Live Chart", "📊 Fundamentals", "📋 Stock Screener", "🔮 Predictions"
    ])

    # Initialize stock analyzer
    analyzer = EnhancedStockAnalyzer()

    with tab1:
        st.header("🤖 AI Stock Assistant")

        if not api_key:
            st.warning("⚠️ Please enter your Google API key in the sidebar to use the AI assistant!")
            st.info("📝 Get your free API key from: https://makersuite.google.com/app/apikey")
        elif st.session_state.agent is None:
            st.error("❌ Failed to initialize AI agent. Please check your API key.")
        else:
            # Chat interface
            user_input = st.text_input("Ask me anything about stocks:",
                                     placeholder="e.g., 'Analyze AAPL stock' or 'Compare TSLA vs MSFT'")

            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("🚀 Send", use_container_width=True):
                    if user_input:
                        with st.spinner("🧠 AI is thinking..."):
                            try:
                                response = st.session_state.agent.run(user_input)
                                st.session_state.chat_history.append(("You", user_input))
                                st.session_state.chat_history.append(("AI", response))
                            except Exception as e:
                                st.error(f"Error: {str(e)}")

            with col2:
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()

            # Display chat history
            st.subheader("💬 Conversation History")
            for speaker, message in st.session_state.chat_history[-10:]:  # Show last 10 messages
                if speaker == "You":
                    st.write(f"**🙋‍♂️ You:** {message}")
                else:
                    st.write(f"**🤖 AI:** {message}")
                st.divider()

    with tab2:
        st.header("📈 Technical Analysis Dashboard")

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            if st.button("🔄 Refresh Data", key="tech_refresh"):
                with st.spinner("Fetching and analyzing data..."):
                    data = analyzer.fetch_stock_data(symbol, period)
                    if data is not None:
                        st.session_state.current_data = data

        with col2:
            show_extended = st.checkbox("📊 Extended Analysis", help="Show additional technical indicators")

        with col3:
            chart_theme = st.selectbox("🎨 Chart Theme", ["plotly_dark", "plotly", "plotly_white"])

        if 'current_data' in st.session_state and st.session_state.current_data is not None:
            data = st.session_state.current_data

            # Calculate technical indicators
            data_with_indicators = analyzer.calculate_technical_indicators(data)

            # Display key metrics
            col1, col2, col3, col4, col5 = st.columns(5)

            current_price = data_with_indicators['Close'][-1]

            # Safe calculation of previous price and change
            if len(data_with_indicators) > 1:
                prev_price = data_with_indicators['Close'][-2]
                change = current_price - prev_price
                change_pct = (change / prev_price) * 100
            else:
                change = 0.0
                change_pct = 0.0

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>${current_price:.2f}</h3>
                    <p>Current Price</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                color_style = 'linear-gradient(135deg, #4CAF50 0%, #45a049 100%)' if change >= 0 else 'linear-gradient(135deg, #f44336 0%, #da190b 100%)'
                st.markdown(f"""
                <div class="metric-card" style="background: {color_style};">
                    <h3>{change:+.2f} ({change_pct:+.2f}%)</h3>
                    <p>Change</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                volume = data_with_indicators['Volume'][-1]
                volume_display = f"{volume/1e6:.1f}M" if volume > 1e6 else f"{volume/1e3:.1f}K"
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{volume_display}</h3>
                    <p>Volume</p>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                if 'RSI' in data_with_indicators.columns and not pd.isna(data_with_indicators['RSI'][-1]):
                    rsi = data_with_indicators['RSI'][-1]
                    if rsi > 70:
                        rsi_color = "#f44336"
                        rsi_status = "Overbought"
                    elif rsi < 30:
                        rsi_color = "#4CAF50"
                        rsi_status = "Oversold"
                    else:
                        rsi_color = "#2196F3"
                        rsi_status = "Neutral"

                    st.markdown(f"""
                    <div class="metric-card" style="background: linear-gradient(135deg, {rsi_color} 0%, {rsi_color}dd 100%);">
                        <h3>{rsi:.1f}</h3>
                        <p>RSI ({rsi_status})</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>N/A</h3>
                        <p>RSI</p>
                    </div>
                    """, unsafe_allow_html=True)

            with col5:
                # Show trend based on available MAs
                ma_columns = [col for col in data_with_indicators.columns if col.startswith('MA')]
                if len(ma_columns) >= 2:
                    short_ma = data_with_indicators[ma_columns[0]][-1]
                    long_ma = data_with_indicators[ma_columns[-1]][-1]
                    if not pd.isna(short_ma) and not pd.isna(long_ma):
                        if current_price > short_ma > long_ma:
                            trend = "Strong Bull"
                            trend_color = "#4CAF50"
                        elif current_price > short_ma:
                            trend = "Bullish"
                            trend_color = "#8BC34A"
                        elif current_price < short_ma < long_ma:
                            trend = "Strong Bear"
                            trend_color = "#f44336"
                        elif current_price < short_ma:
                            trend = "Bearish"
                            trend_color = "#FF5722"
                        else:
                            trend = "Sideways"
                            trend_color = "#FF9800"
                    else:
                        trend = "Unknown"
                        trend_color = "#9E9E9E"
                else:
                    trend = "Insufficient Data"
                    trend_color = "#9E9E9E"

                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, {trend_color} 0%, {trend_color}dd 100%);">
                    <h3>{trend}</h3>
                    <p>Trend</p>
                </div>
                """, unsafe_allow_html=True)

            # Data quality indicator
            data_length = len(data_with_indicators)
            if data_length < 20:
                st.markdown(f"""
                <div class="warning-box">
                    ⚠️ <strong>Limited Data Warning:</strong> Only {data_length} data points available.
                    Some technical indicators may be less reliable. Consider using a longer time period.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="success-box">
                    ✅ <strong>Good Data Quality:</strong> {data_length} data points available for analysis.
                </div>
                """, unsafe_allow_html=True)

            # Create and display chart
            fig = create_live_stock_chart(data_with_indicators, symbol)
            fig.update_layout(template=chart_theme)
            st.plotly_chart(fig, use_container_width=True)

            # Technical indicators table
            if show_extended:
                st.subheader("📊 Current Technical Indicators")

                # Create comprehensive indicators table
                def safe_get_indicator(df, col_name, format_str="{:.2f}", prefix=""):
                    try:
                        if col_name in df.columns and not pd.isna(df[col_name].iloc[-1]):
                            value = df[col_name].iloc[-1]
                            if prefix == "$":
                                return f"${value:.2f}"
                            else:
                                return format_str.format(value)
                        return 'N/A'
                    except:
                        return 'N/A'

                # Get all available indicators
                available_indicators = []

                # Price-based indicators
                for col in data_with_indicators.columns:
                    if col.startswith('MA'):
                        available_indicators.append((col, safe_get_indicator(data_with_indicators, col, prefix="$")))

                # Technical indicators
                indicator_map = {
                    'RSI': ('RSI', "{:.2f}"),
                    'MACD': ('MACD', "{:.4f}"),
                    'MACD_Signal': ('MACD Signal', "{:.4f}"),
                    'BB_Upper': ('Bollinger Upper', "{:.2f}", "$"),
                    'BB_Lower': ('Bollinger Lower', "{:.2f}", "$"),
                    'ATR': ('ATR', "{:.2f}"),
                    'Stoch_K': ('Stochastic %K', "{:.2f}"),
                    'Stoch_D': ('Stochastic %D', "{:.2f}"),
                    'OBV': ('On Balance Volume', "{:,.0f}")
                }

                for col, (name, fmt, *prefix) in indicator_map.items():
                    pref = prefix[0] if prefix else ""
                    value = safe_get_indicator(data_with_indicators, col, fmt, pref)
                    if value != 'N/A':
                        available_indicators.append((name, value))

                # Create two columns for indicators
                if available_indicators:
                    col1, col2 = st.columns(2)
                    mid_point = len(available_indicators) // 2

                    with col1:
                        for indicator, value in available_indicators[:mid_point]:
                            st.write(f"**{indicator}:** {value}")

                    with col2:
                        for indicator, value in available_indicators[mid_point:]:
                            st.write(f"**{indicator}:** {value}")
        else:
            st.info("Click 'Refresh Data' to load technical analysis for the selected symbol.")

    with tab3:
        st.header("🔴 Live Stock Chart")

        # Live mode controls
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("▶️ Start Live Mode", key="start_live"):
                st.session_state.auto_refresh = True

        with col2:
            if st.button("⏸️ Pause Live Mode", key="pause_live"):
                st.session_state.auto_refresh = False

        with col3:
            if st.button("🔄 Manual Refresh", key="manual_refresh"):
                with st.spinner("Fetching live data..."):
                    # Use shorter period and interval for live data
                    live_data = analyzer.fetch_stock_data(symbol, period="1d", interval="1m")
                    if live_data is not None:
                        st.session_state.live_data[symbol] = live_data

        with col4:
            live_status = "🟢 LIVE" if st.session_state.auto_refresh else "🔴 PAUSED"
            st.markdown(f"""
            <div class="live-indicator">
                {live_status} - Last Update: {datetime.datetime.now().strftime('%H:%M:%S')}
            </div>
            """, unsafe_allow_html=True)

        # Auto-refresh mechanism
        if st.session_state.auto_refresh and live_mode:
            # Create placeholder for dynamic updates
            chart_placeholder = st.empty()
            metrics_placeholder = st.empty()

            # Auto-refresh loop (simplified for demo)
            time.sleep(1)  # Small delay
            with st.spinner("Updating live data..."):
                live_data = analyzer.fetch_stock_data(symbol, period="1d", interval="5m")
                if live_data is not None:
                    st.session_state.live_data[symbol] = live_data

                    # Calculate indicators for live data
                    live_data_with_indicators = analyzer.calculate_technical_indicators(live_data)

                    # Update metrics
                    with metrics_placeholder.container():
                        col1, col2, col3, col4 = st.columns(4)

                        current_price = live_data_with_indicators['Close'][-1]
                        if len(live_data_with_indicators) > 1:
                            prev_price = live_data_with_indicators['Close'][-2]
                            change = current_price - prev_price
                            change_pct = (change / prev_price) * 100
                        else:
                            change = 0.0
                            change_pct = 0.0

                        with col1:
                            st.metric("Price", f"${current_price:.2f}", f"{change:+.2f}")
                        with col2:
                            st.metric("Change %", f"{change_pct:+.2f}%")
                        with col3:
                            volume = live_data_with_indicators['Volume'][-1]
                            st.metric("Volume", f"{volume:,.0f}")
                        with col4:
                            if 'RSI' in live_data_with_indicators.columns:
                                rsi = live_data_with_indicators['RSI'][-1]
                                st.metric("RSI", f"{rsi:.1f}")

                    # Update chart
                    with chart_placeholder.container():
                        live_fig = create_live_stock_chart(live_data_with_indicators, symbol, live_mode=True)
                        st.plotly_chart(live_fig, use_container_width=True)

            # Auto-refresh timer
            if st.session_state.auto_refresh:
                time.sleep(auto_refresh_interval)
                st.rerun()

        # Display static live data if available
        elif symbol in st.session_state.live_data:
            live_data = st.session_state.live_data[symbol]
            live_data_with_indicators = analyzer.calculate_technical_indicators(live_data)

            # Live metrics
            col1, col2, col3, col4 = st.columns(4)
            current_price = live_data_with_indicators['Close'][-1]

            if len(live_data_with_indicators) > 1:
                prev_price = live_data_with_indicators['Close'][-2]
                change = current_price - prev_price
                change_pct = (change / prev_price) * 100
            else:
                change = 0.0
                change_pct = 0.0

            with col1:
                st.metric("Live Price", f"${current_price:.2f}", f"{change:+.2f}")
            with col2:
                st.metric("Change %", f"{change_pct:+.2f}%")
            with col3:
                volume = live_data_with_indicators['Volume'][-1]
                st.metric("Volume", f"{volume:,.0f}")
            with col4:
                if 'RSI' in live_data_with_indicators.columns and not pd.isna(live_data_with_indicators['RSI'][-1]):
                    rsi = live_data_with_indicators['RSI'][-1]
                    st.metric("RSI", f"{rsi:.1f}")

            # Live chart
            live_fig = create_live_stock_chart(live_data_with_indicators, symbol, live_mode=True)
            st.plotly_chart(live_fig, use_container_width=True)

            st.info("💡 Enable 'Live Data' in the sidebar and click 'Start Live Mode' for automatic updates.")
        else:
            st.info("Click 'Manual Refresh' to load live data for the selected symbol.")

    with tab4:
        st.header("📊 Fundamental Analysis")

        if st.button("📈 Get Fundamentals", key="fund_refresh"):
            info = analyzer.get_stock_info(symbol)
            if info:
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("💰 Valuation Metrics")
                    metrics = {
                        'Market Cap': f"${info.get('marketCap', 'N/A'):,}" if info.get('marketCap') else 'N/A',
                        'Enterprise Value': f"${info.get('enterpriseValue', 'N/A'):,}" if info.get('enterpriseValue') else 'N/A',
                        'P/E Ratio': info.get('trailingPE', 'N/A'),
                        'Forward P/E': info.get('forwardPE', 'N/A'),
                        'PEG Ratio': info.get('pegRatio', 'N/A'),
                        'Price to Book': info.get('priceToBook', 'N/A'),
                        'Price to Sales': info.get('priceToSalesTrailing12Months', 'N/A')
                    }

                    for metric, value in metrics.items():
                        st.write(f"**{metric}:** {value}")

                with col2:
                    st.subheader("💼 Financial Health")
                    financial = {
                        'Revenue Growth': f"{info.get('revenueGrowth', 0)*100:.2f}%" if info.get('revenueGrowth') else 'N/A',
                        'Profit Margin': f"{info.get('profitMargins', 0)*100:.2f}%" if info.get('profitMargins') else 'N/A',
                        'Return on Assets': f"{info.get('returnOnAssets', 0)*100:.2f}%" if info.get('returnOnAssets') else 'N/A',
                        'Return on Equity': f"{info.get('returnOnEquity', 0)*100:.2f}%" if info.get('returnOnEquity') else 'N/A',
                        'Debt to Equity': info.get('debtToEquity', 'N/A'),
                        'Current Ratio': info.get('currentRatio', 'N/A'),
                        'Quick Ratio': info.get('quickRatio', 'N/A')
                    }

                    for metric, value in financial.items():
                        st.write(f"**{metric}:** {value}")

                st.subheader("ℹ️ Company Information")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Company:** {info.get('longName', symbol)}")
                    st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                    st.write(f"**Industry:** {info.get('industry', 'N/A')}")
                    st.write(f"**Employees:** {info.get('fullTimeEmployees', 'N/A'):,}" if info.get('fullTimeEmployees') else "**Employees:** N/A")

                with col2:
                    st.write(f"**Exchange:** {info.get('exchange', 'N/A')}")
                    st.write(f"**Currency:** {info.get('currency', 'N/A')}")
                    st.write(f"**Country:** {info.get('country', 'N/A')}")
                    st.write(f"**Website:** {info.get('website', 'N/A')}")

                if info.get('longBusinessSummary'):
                    st.subheader("📝 Business Summary")
                    st.write(info['longBusinessSummary'])

    with tab5:
        st.header("📋 Stock Screener")

        st.subheader("🔍 Popular Stock Tickers")

        # Popular tickers by category
        categories = {
            "Tech Giants": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX"],
            "Financial": ["JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC"],
            "Healthcare": ["JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "DHR", "BMY"],
            "Consumer": ["KO", "PEP", "WMT", "HD", "MCD", "NKE", "SBUX", "TGT"],
            "Energy": ["XOM", "CVX", "COP", "EOG", "SLB", "PXD", "KMI", "WMB"]
        }

        selected_category = st.selectbox("Select Category:", list(categories.keys()))
        tickers = categories[selected_category]

        if st.button("🔄 Screen Selected Category"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []

            for i, ticker in enumerate(tickers):
                status_text.text(f"Processing {ticker}... ({i+1}/{len(tickers)})")
                try:
                    data = analyzer.fetch_stock_data(ticker, period="5d")
                    info = analyzer.get_stock_info(ticker)

                    if data is not None and not data.empty:
                        current_price = data['Close'][-1]
                        prev_price = data['Close'][-2] if len(data) > 1 else current_price
                        change_pct = ((current_price - prev_price) / prev_price) * 100

                        # Calculate simple RSI if possible
                        rsi = 'N/A'
                        if len(data) >= 14:
                            try:
                                data_with_ind = analyzer.calculate_technical_indicators(data)
                                if 'RSI' in data_with_ind.columns and not pd.isna(data_with_ind['RSI'][-1]):
                                    rsi = f"{data_with_ind['RSI'][-1]:.1f}"
                            except:
                                pass

                        results.append({
                            'Symbol': ticker,
                            'Company': info.get('longName', ticker)[:30] + '...' if info.get('longName') and len(info.get('longName')) > 30 else info.get('longName', ticker),
                            'Price': f"${current_price:.2f}",
                            'Change %': f"{change_pct:.2f}%",
                            'Market Cap': f"${info.get('marketCap', 0)/1e9:.1f}B" if info.get('marketCap') else 'N/A',
                            'P/E': f"{info.get('trailingPE', 'N/A'):.1f}" if isinstance(info.get('trailingPE'), (int, float)) else 'N/A',
                            'RSI': rsi,
                            'Volume': f"{data['Volume'][-1]/1e6:.1f}M"
                        })

                    progress_bar.progress((i + 1) / len(tickers))
                except:
                    continue

            status_text.text("Screening complete!")

            if results:
                df_results = pd.DataFrame(results)

                # Add color coding for change %
                def color_negative_red(val):
                    if isinstance(val, str) and val.endswith('%'):
                        try:
                            num_val = float(val.replace('%', ''))
                            color = 'red' if num_val < 0 else 'green' if num_val > 0 else 'black'
                            return f'color: {color}'
                        except:
                            return ''
                    return ''

                styled_df = df_results.style.applymap(color_negative_red, subset=['Change %'])
                st.dataframe(styled_df, use_container_width=True)

                # Summary statistics
                st.subheader("📊 Screening Summary")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    gainers = sum(1 for result in results if float(result['Change %'].replace('%', '')) > 0)
                    st.metric("Gainers", gainers)

                with col2:
                    losers = sum(1 for result in results if float(result['Change %'].replace('%', '')) < 0)
                    st.metric("Losers", losers)

                with col3:
                    avg_change = sum(float(result['Change %'].replace('%', '')) for result in results) / len(results)
                    st.metric("Avg Change", f"{avg_change:.2f}%")

                with col4:
                    st.metric("Total Screened", len(results))

    with tab6:
        st.header("🔮 Stock Predictions & Analysis")

        st.subheader("📈 Enhanced Price Analysis")

        if st.button("🤖 Generate Enhanced Analysis"):
            data = analyzer.fetch_stock_data(symbol, period="1y")
            if data is not None:
                # Enhanced analysis with technical indicators
                data_with_indicators = analyzer.calculate_technical_indicators(data)

                # Recent trend analysis (30 days)
                recent_data = data_with_indicators.tail(30) if len(data_with_indicators) >= 30 else data_with_indicators

                # Price analysis
                current_price = data_with_indicators['Close'][-1]
                price_trend = "Upward" if recent_data['Close'].iloc[-1] > recent_data['Close'].iloc[0] else "Downward"
                volatility = recent_data['Close'].pct_change().std() * 100

                # Volume analysis
                avg_volume = data_with_indicators['Volume'].tail(30).mean()
                recent_volume = data_with_indicators['Volume'][-1]
                volume_ratio = recent_volume / avg_volume

                # Technical signals
                signals = []

                # RSI signal
                if 'RSI' in data_with_indicators.columns and not pd.isna(data_with_indicators['RSI'][-1]):
                    rsi = data_with_indicators['RSI'][-1]
                    if rsi > 70:
                        signals.append("🔴 RSI indicates overbought condition")
                    elif rsi < 30:
                        signals.append("🟢 RSI indicates oversold condition")
                    else:
                        signals.append("🟡 RSI in neutral territory")

                # Moving average signal
                ma_columns = [col for col in data_with_indicators.columns if col.startswith('MA')]
                if len(ma_columns) >= 2:
                    short_ma = data_with_indicators[ma_columns[0]][-1]
                    long_ma = data_with_indicators[ma_columns[-1]][-1]
                    if not pd.isna(short_ma) and not pd.isna(long_ma):
                        if current_price > short_ma > long_ma:
                            signals.append("🟢 Strong bullish trend - price above all moving averages")
                        elif current_price < short_ma < long_ma:
                            signals.append("🔴 Strong bearish trend - price below all moving averages")
                        else:
                            signals.append("🟡 Mixed signals from moving averages")

                # Volume signal
                if volume_ratio > 1.5:
                    signals.append("🟢 High volume confirms price movement")
                elif volume_ratio < 0.7:
                    signals.append("🟡 Low volume - weak conviction in price movement")

                # Display analysis
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"""
                    ### 📊 Technical Analysis Summary
                    - **Current Price:** ${current_price:.2f}
                    - **30-Day Trend:** {price_trend}
                    - **Volatility:** {volatility:.2f}%
                    - **Support Level:** ${recent_data['Low'].min():.2f}
                    - **Resistance Level:** ${recent_data['High'].max():.2f}
                    - **Volume Ratio:** {volume_ratio:.2f}x average
                    """)

                with col2:
                    st.markdown(f"""
                    ### 🎯 Key Levels & Ranges
                    - **52W High:** ${data_with_indicators['High'].max():.2f}
                    - **52W Low:** ${data_with_indicators['Low'].min():.2f}
                    - **Average Volume:** {avg_volume/1e6:.1f}M
                    - **Price from 52W High:** {((current_price - data_with_indicators['High'].max()) / data_with_indicators['High'].max() * 100):+.1f}%
                    - **Price from 52W Low:** {((current_price - data_with_indicators['Low'].min()) / data_with_indicators['Low'].min() * 100):+.1f}%
                    """)

                # Technical signals
                st.subheader("🚦 Technical Signals")
                for signal in signals:
                    st.write(signal)

                # Risk assessment
                st.subheader("⚠️ Risk Assessment")
                risk_factors = []

                if volatility > 5:
                    risk_factors.append("High volatility indicates increased risk")
                if volume_ratio < 0.5:
                    risk_factors.append("Low volume may indicate lack of interest")

                # Check for extreme RSI
                if 'RSI' in data_with_indicators.columns and not pd.isna(data_with_indicators['RSI'][-1]):
                    rsi = data_with_indicators['RSI'][-1]
                    if rsi > 80 or rsi < 20:
                        risk_factors.append("Extreme RSI levels suggest potential reversal")

                if risk_factors:
                    for factor in risk_factors:
                        st.write(f"⚠️ {factor}")
                else:
                    st.write("✅ No major technical risk factors identified")

                # Prediction disclaimer
                st.markdown("""
                ---
                ⚠️ **Important Disclaimer:**

                This analysis is for educational and informational purposes only. Stock predictions are inherently uncertain and should never be used as the sole basis for investment decisions. Markets can be unpredictable, and past performance does not guarantee future results.

                **Always:**
                - Consult with qualified financial professionals
                - Conduct your own thorough research
                - Consider your risk tolerance and investment goals
                - Diversify your investment portfolio
                - Never invest more than you can afford to lose
                """)

if __name__ == "__main__":
    main()