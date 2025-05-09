import requests
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import numpy as np
import time

class HistoricalAnalysis:
    def __init__(self, coin_id, initial_balance=1000.0, start_date=None):
        """
        Initialize the historical analysis with advanced parameters
        Args:
            coin_id (str): CoinGecko ID of the coin
            initial_balance (float): Initial portfolio value
            start_date (str): Start date for analysis (YYYY-MM-DD)
        """
        self.coin_id = coin_id
        self.initial_balance = initial_balance
        self.coin_balance = 0
        self.last_price = None
        self.trades = []
        self.portfolio_values = []
        self.start_date = start_date
        
        # Parameters for advanced strategy
        self.prices = []  # Store historical prices for trend analysis
        self.volatility_window = 20  # Number of periods for volatility calculation
        self.trend_window = 10  # Number of periods for trend analysis
        self.volatility_threshold = 0.02  # 2% volatility threshold
        self.risk_factor = 0.01  # 1% risk per trade
        self.max_position_size = 0.5  # Maximum 50% of balance per trade
        self.stop_loss_pct = 0.1  # 10% stop loss
        self.take_profit_pct = 0.2  # 20% take profit
        
        # Position management
        self.position_size = 0
        self.entry_price = 0
        self.take_profit_price = 0
        self.stop_loss_price = 0

    def get_historical_prices(self):
        """
        Fetch historical coin prices from CoinGecko API
        Returns:
            pd.DataFrame: DataFrame with historical prices
        """
        try:
            # Use the daily market chart endpoint which is free
            url = f'https://api.coingecko.com/api/v3/coins/{self.coin_id}/market_chart'
            params = {
                'vs_currency': 'usd',
                'days': '365',  # Get data for the last year
                'interval': 'daily'  # Daily data points
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Convert prices to DataFrame
            prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
            prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')
            prices.set_index('timestamp', inplace=True)
            
            return prices
        except requests.RequestException as e:
            print(f"Error fetching historical prices: {e}")
            return None

    def simulate_trading(self, prices):
        """
        Simulate trading based on historical prices with advanced strategy
        Args:
            prices (pd.DataFrame): DataFrame with historical prices
        """
        for timestamp, row in prices.iterrows():
            current_price = row['price']
            
            # Check stop loss
            if self.coin_balance > 0 and current_price <= self.stop_loss_price:
                print(f"Executing stop loss at ${current_price:.2f} on {timestamp}")
                self.execute_trade('sell', current_price, timestamp)
                continue
            
            # Check take profit
            if self.coin_balance > 0 and current_price >= self.take_profit_price:
                print(f"Executing take profit at ${current_price:.2f} on {timestamp}")
                self.execute_trade('sell', current_price, timestamp)
                continue
            
            # Check if we should trade
            action = self.should_trade(current_price)
            if action:
                print(f"Executing {action} at ${current_price:.2f} on {timestamp}")
                self.execute_trade(action, current_price, timestamp)
            
            # Update last price
            self.last_price = current_price

    def should_trade(self, current_price):
        """
        Determine if we should buy or sell based on multiple factors
        Args:
            current_price (float): Current coin price
        Returns:
            str: 'buy', 'sell', or None
        """
        if self.last_price is None:
            return None

        # Calculate price change
        price_change = ((current_price - self.last_price) / self.last_price) * 100

        # Calculate volatility
        self.prices.append(current_price)
        if len(self.prices) > self.volatility_window:
            self.prices.pop(0)
        
        if len(self.prices) >= self.volatility_window:
            returns = np.diff(self.prices) / self.prices[:-1]
            volatility = np.std(returns)
        else:
            volatility = 0

        # Calculate trend
        if len(self.prices) >= self.trend_window:
            trend = np.mean(np.diff(self.prices[-self.trend_window:]))
        else:
            trend = 0

        # Dynamic threshold based on volatility and trend
        base_threshold = 1.0
        volatility_factor = min(volatility / self.volatility_threshold, 1)
        trend_factor = np.sign(trend) * min(abs(trend) / self.volatility_threshold, 1)
        
        threshold = base_threshold * (1 + volatility_factor + trend_factor)
        
        # Apply dynamic threshold with trend consideration
        if price_change <= -threshold and trend < 0:  # Buy if price drops and trend is downward
            return 'buy'
        elif price_change >= threshold and trend > 0:  # Sell if price rises and trend is upward
            return 'sell'
        
        return None

    def execute_trade(self, action, price, timestamp):
        """
        Execute a trade with advanced position sizing and risk management
        Args:
            action (str): 'buy' or 'sell'
            price (float): Current coin price
            timestamp (datetime): Time of trade
        """
        if action == 'buy':
            # Calculate position size based on volatility and risk
            if len(self.prices) >= self.volatility_window:
                returns = np.diff(self.prices[-self.volatility_window:]) / self.prices[-self.volatility_window:-1]
                volatility = np.std(returns)
                position_size = self.initial_balance * self.risk_factor / volatility
            else:
                position_size = self.initial_balance * self.risk_factor
            
            # Apply maximum position size limit
            position_size = min(position_size, self.initial_balance * self.max_position_size)
            
            # Calculate coin amount to buy
            coin_amount = position_size / price
            self.coin_balance += coin_amount
            self.initial_balance -= position_size
            
            # Set entry price and calculate stop loss/take profit
            self.entry_price = price
            self.take_profit_price = price * (1 + self.take_profit_pct)
            self.stop_loss_price = price * (1 - self.stop_loss_pct)
            
        elif action == 'sell':
            # Sell all coin
            self.initial_balance += self.coin_balance * price
            self.coin_balance = 0
            
            # Reset entry price and levels
            self.entry_price = 0
            self.take_profit_price = 0
            self.stop_loss_price = 0

        # Log the trade
        self.trades.append({
            'timestamp': timestamp,
            'action': action,
            'price': price,
            'amount': coin_amount if action == 'buy' else 0,
            'balance': self.initial_balance,
            'coin_balance': self.coin_balance
        })

        # Calculate portfolio value
        portfolio_value = self.initial_balance + (price * self.coin_balance)
        self.portfolio_values.append({
            'timestamp': timestamp,
            'value': portfolio_value
        })

    def create_analysis_report(self, coin_name):
        """
        Create and display analysis report
        Args:
            coin_name (str): Name of the coin for display
        Returns:
            tuple: (total_return, final_value)
        """
        if not self.portfolio_values:
            return None, None

        initial_value = self.initial_balance
        final_value = self.portfolio_values[-1]['value']
        total_return = ((final_value - initial_value) / initial_value) * 100

        # Print detailed report
        print(f"\n=== {coin_name} Analysis Report ===")
        print(f"Start Date: {self.portfolio_values[0]['timestamp'].strftime('%Y-%m-%d')}")
        print(f"End Date: {self.portfolio_values[-1]['timestamp'].strftime('%Y-%m-%d')}")
        print(f"Total Trades: {len(self.trades)}")
        print(f"Initial Value: ${initial_value:.2f}")
        print(f"Final Value: ${final_value:.2f}")
        print(f"Total Return: {total_return:.2f}%")

        # Create portfolio value chart
        if self.portfolio_values:
            values_df = pd.DataFrame(self.portfolio_values)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=values_df['timestamp'],
                y=values_df['value'],
                mode='lines',
                name=f'{coin_name} Portfolio'
            ))
            
            fig.update_layout(
                title=f'{coin_name} Portfolio Value Over Time',
                xaxis_title='Date',
                yaxis_title='Value (USD)',
                template='plotly_dark'
            )
            
            fig.show()

            return total_return, final_value

    def run_analysis(self, coin_name):
        """
        Run complete historical analysis
        Args:
            coin_name (str): Name of the coin for display
        """
        print("Fetching historical data...")
        prices = self.get_historical_prices()
        
        if prices is not None:
            print("Simulating trading...")
            self.simulate_trading(prices)
            print("Creating analysis report...")
            return self.create_analysis_report(coin_name)
        return None, None

    def get_historical_prices(self):
        """
        Fetch historical coin prices from CoinGecko API
        Returns:
            pd.DataFrame: DataFrame with historical prices
        """
        try:
            # Use the daily market chart endpoint which is free
            url = f'https://api.coingecko.com/api/v3/coins/{self.coin_id}/market_chart'
            params = {
                'vs_currency': 'usd',
                'days': '365',  # Get data for the last year
                'interval': 'daily'  # Daily data points
            }
            
            # Add delay to respect rate limits
            time.sleep(5)  # Wait 5 seconds between requests
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Convert prices to DataFrame
            prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
            prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')
            prices.set_index('timestamp', inplace=True)
            
            return prices
        except requests.RequestException as e:
            print(f"Error fetching historical prices: {e}")
            return None

if __name__ == "__main__":
    # Define coins to analyze
    coins = {
        'ethereum': 'Ethereum',
        'solana': 'Solana',
        'matic-network': 'Polygon',
        'avalanche-2': 'Avalanche',
        'pepe': 'Pepe'
    }

    # Store results
    results = {}
    
    # Analyze each coin
    for coin_id, coin_name in coins.items():
        print(f"\nAnalyzing {coin_name}...")
        analyzer = HistoricalAnalysis(coin_id)
        total_return, final_value = analyzer.run_analysis(coin_name)
        if total_return is not None and final_value is not None:
            results[coin_name] = {'return': total_return, 'final_value': final_value}
        
        # Add delay between coin analyses
        time.sleep(10)  # Wait 10 seconds between coins

    # Print comparison
    print("\n=== Coin Performance Comparison ===")
    print("\nCoin\t\tReturn (%)\tFinal Value")
    for coin_name, data in results.items():
        print(f"{coin_name}\t\t{data['return']:.2f}\t\t${data['final_value']:.2f}")
