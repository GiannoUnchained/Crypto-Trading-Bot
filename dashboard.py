import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import requests

# Try to import config
try:
    import config
except ImportError:
    import config_template as config

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.trade_history = pd.DataFrame()
    st.session_state.portfolio_value = 0
    st.session_state.coin_balances = {}
    st.session_state.last_update = datetime.now()

# Load trade history
def load_trade_history():
    try:
        df = pd.read_csv('trade_history.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        print(f"Error loading trade history: {e}")
        return pd.DataFrame()

# Calculate portfolio value
def calculate_portfolio_value(df):
    if df.empty:
        return 1000.0, {}  # Start with initial balance if no trades yet
    
    # Get latest balances
    balances = df.groupby('coin_id')['amount'].sum()
    
    # Get current prices
    prices = {}
    total_value = 0
    
    # Start with USD balance
    total_value = df['usd_balance'].iloc[-1]
    
    for coin_id in balances.index:
        try:
            response = requests.get(
                'https://api.coingecko.com/api/v3/simple/price',
                params={'ids': coin_id, 'vs_currencies': 'usd'}
            )
            response.raise_for_status()
            price = response.json()[coin_id]['usd']
            prices[coin_id] = price
            total_value += balances[coin_id] * price
        except Exception as e:
            print(f"Error fetching price for {coin_id}: {e}")
            prices[coin_id] = 0
    
    return total_value, prices

# Send push notification for significant changes
def send_notification_if_needed(old_value, new_value):
    """
    Send push notification if portfolio value changes significantly
    """
    if old_value == 0:
        return
    
    change_pct = ((new_value - old_value) / old_value) * 100
    
    # Check if change exceeds threshold
    if abs(change_pct) >= config.PORTFOLIO_CHANGE_THRESHOLD:
        message = f"Portfolio value changed by {change_pct:.2f}%\n"
        message += f"Current value: ${new_value:.2f}\n"
        message += f"Previous value: ${old_value:.2f}"
        
        try:
            if config.PUSHBULLET_API_KEY:
                pb = Pushbullet(config.PUSHBULLET_API_KEY)
                pb.push_note("Portfolio Change Alert", message)
        except Exception as e:
            print(f"Error sending notification: {e}")

def main():
    st.title("Crypto Trading Bot Dashboard")
    
    # Load trade history
    if not st.session_state.data_loaded:
        st.session_state.trade_history = load_trade_history()
        st.session_state.data_loaded = True
    
    # Calculate portfolio value
    old_value = st.session_state.portfolio_value
    portfolio_value, prices = calculate_portfolio_value(st.session_state.trade_history)
    
    # Send notification if needed
    send_notification_if_needed(old_value, portfolio_value)
    
    # Update session state
    st.session_state.portfolio_value = portfolio_value
    st.session_state.last_update = datetime.now()
    
    # Show portfolio overview
    col1, col2, col3 = st.columns(3)
    col1.metric("Portfolio Value", f"${portfolio_value:,.2f}")
    col2.metric("Total Trades", len(st.session_state.trade_history))
    col3.metric("Last Update", st.session_state.last_update.strftime("%Y-%m-%d %H:%M:%S"))
    
    # Show coin balances
    st.subheader("Coin Balances")
    balances = st.session_state.trade_history.groupby('coin_id')['amount'].sum()
    for coin_id, balance in balances.items():
        st.metric(coin_id.title(), f"{balance:,.4f}")
    
    # Show trade history
    st.subheader("Trade History")
    if not st.session_state.trade_history.empty:
        trade_history = st.session_state.trade_history.groupby('coin_id').tail(10)
        st.dataframe(trade_history[['timestamp', 'coin_id', 'action', 'price', 'amount', 'usd_balance', 'coin_balance']])
    
    # Show portfolio value chart
    st.subheader("Portfolio Value Over Time")
    portfolio_df = st.session_state.trade_history.copy()
    portfolio_df['portfolio_value'] = portfolio_df.groupby('coin_id')['amount'].transform('sum') * portfolio_df['price']
    fig = px.line(portfolio_df, x='timestamp', y='portfolio_value', color='coin_id')
    st.plotly_chart(fig)
    
    # Show price chart
    st.subheader("Price History")
    price_df = st.session_state.trade_history.copy()
    fig = px.line(price_df, x='timestamp', y='price', color='coin_id')
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
