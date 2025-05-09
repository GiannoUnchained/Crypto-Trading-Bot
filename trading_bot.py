import requests
import csv
import time
from datetime import datetime
import numpy as np
import pandas as pd
from datetime import timedelta
from pushbullet import Pushbullet
import config

# Initialize Pushbullet
try:
    pb = Pushbullet(config.PUSHBULLET_API_KEY)
    if config.PUSHBULLET_DEVICE_ID:
        device = pb.get_device(config.PUSHBULLET_DEVICE_ID)
except Exception as e:
    print(f"Warning: Could not initialize Pushbullet: {e}")
    pb = None

class DataCache:
    def __init__(self):
        self.cache = {}
        self.ttl = timedelta(hours=1)  # Cache timeout
        
    def get(self, key):
        """
        Get cached data if it exists and is not expired
        """
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.ttl:
                return data
        return None
    
    def set(self, key, value):
        """
        Store data in cache with current timestamp
        """
        self.cache[key] = (value, datetime.now())

class ExternalData:
    def __init__(self):
        self.cache = DataCache()
        self.fallback_data = {
            'sentiment': 0,
            'social_volume': 1000,
            'exchange_volume': 1000000,
            'network_activity': {
                'active_addresses': 1000,
                'transactions': 1000
            }
        }
        
    def get_news_sentiment(self, coin_id):
        """
        Fetch news sentiment for a specific coin with caching and fallback
        Returns:
            float: Sentiment score (-1 to 1)
        """
        cache_key = f'sentiment_{coin_id}'
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
            
        try:
            # Fallback to multiple APIs
            apis = [
                'https://api.coingecko.com/api/v3/coins/news',
                'https://api.coinmarketcap.com/v1/news',
                'https://api.cryptocompare.com/news'
            ]
            
            for api in apis:
                try:
                    response = requests.get(
                        api,
                        params={'coin_id': coin_id},
                        timeout=5
                    )
                    response.raise_for_status()
                    
                    # Calculate sentiment score
                    news = response.json()
                    sentiment = 0
                    count = 0
                    
                    for article in news:
                        # Simple sentiment scoring based on keywords
                        positive_words = ['positive', 'bullish', 'growth', 'increase', 'up']
                        negative_words = ['negative', 'bearish', 'decrease', 'down', 'loss']
                        
                        text = article['title'].lower() + ' ' + article['description'].lower()
                        
                        # Count positive and negative words
                        pos_count = sum(1 for word in positive_words if word in text)
                        neg_count = sum(1 for word in negative_words if word in text)
                        
                        # Calculate sentiment
                        if pos_count + neg_count > 0:
                            sentiment += (pos_count - neg_count) / (pos_count + neg_count)
                            count += 1
                    
                    score = sentiment / count if count > 0 else 0
                    self.cache.set(cache_key, score)
                    return score
                except Exception as e:
                    print(f"Error with {api}: {e}")
                    continue
            
            # Use fallback data if all APIs fail
            print(f"All APIs failed for sentiment, using fallback")
            return self.fallback_data['sentiment']
            
        except Exception as e:
            print(f"Error fetching news sentiment: {e}")
            return self.fallback_data['sentiment']
    
    def get_social_volume(self, coin_id):
        """
        Fetch social media volume for a specific coin with caching and fallback
        Returns:
            int: Number of mentions
        """
        cache_key = f'social_volume_{coin_id}'
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
            
        try:
            # Fallback to multiple APIs
            apis = [
                f'https://api.coingecko.com/api/v3/coins/{coin_id}/social',
                f'https://api.coinmarketcap.com/v1/coins/{coin_id}/social',
                f'https://api.cryptocompare.com/social/{coin_id}'
            ]
            
            for api in apis:
                try:
                    response = requests.get(api, timeout=5)
                    response.raise_for_status()
                    
                    data = response.json()
                    volume = data.get('twitter_followers', 0) + data.get('reddit_subscribers', 0)
                    self.cache.set(cache_key, volume)
                    return volume
                except Exception as e:
                    print(f"Error with {api}: {e}")
                    continue
            
            # Use fallback data if all APIs fail
            print(f"All APIs failed for social volume, using fallback")
            return self.fallback_data['social_volume']
            
        except Exception as e:
            print(f"Error fetching social volume: {e}")
            return self.fallback_data['social_volume']
    
    def get_exchange_volume(self, coin_id):
        """
        Fetch exchange trading volume for a specific coin with caching and fallback
        Returns:
            float: Trading volume in USD
        """
        cache_key = f'exchange_volume_{coin_id}'
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
            
        try:
            # Fallback to multiple APIs
            apis = [
                f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart',
                f'https://api.coinmarketcap.com/v1/coins/{coin_id}/volume',
                f'https://api.cryptocompare.com/volume/{coin_id}'
            ]
            
            for api in apis:
                try:
                    response = requests.get(
                        api,
                        params={'vs_currency': 'usd', 'days': 1},
                        timeout=5
                    )
                    response.raise_for_status()
                    
                    data = response.json()
                    volume = data['total_volume'][-1][1] if 'total_volume' in data else 0
                    self.cache.set(cache_key, volume)
                    return volume
                except Exception as e:
                    print(f"Error with {api}: {e}")
                    continue
            
            # Use fallback data if all APIs fail
            print(f"All APIs failed for exchange volume, using fallback")
            return self.fallback_data['exchange_volume']
            
        except Exception as e:
            print(f"Error fetching exchange volume: {e}")
            return self.fallback_data['exchange_volume']
    
    def get_network_activity(self, coin_id):
        """
        Fetch network activity metrics for a specific coin with caching and fallback
        Returns:
            dict: Network activity metrics
        """
        cache_key = f'network_activity_{coin_id}'
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
            
        try:
            # Fallback to multiple APIs
            apis = [
                f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart',
                f'https://api.coinmarketcap.com/v1/coins/{coin_id}/network',
                f'https://api.cryptocompare.com/network/{coin_id}'
            ]
            
            for api in apis:
                try:
                    response = requests.get(
                        api,
                        params={'vs_currency': 'usd', 'days': 1},
                        timeout=5
                    )
                    response.raise_for_status()
                    
                    data = response.json()
                    activity = {
                        'active_addresses': data.get('active_addresses', [])[0][1] if 'active_addresses' in data else 0,
                        'transactions': data.get('transactions', [])[0][1] if 'transactions' in data else 0
                    }
                    self.cache.set(cache_key, activity)
                    return activity
                except Exception as e:
                    print(f"Error with {api}: {e}")
                    continue
            
            # Use fallback data if all APIs fail
            print(f"All APIs failed for network activity, using fallback")
            return self.fallback_data['network_activity']
            
        except Exception as e:
            print(f"Error fetching network activity: {e}")
            return self.fallback_data['network_activity']
class HistoricalData:
    def __init__(self):
        self.data = {}
        
    def load_data(self, coin_id, start_date, end_date):
        """
        Load historical price data for a specific coin
        Args:
            coin_id (str): ID of the coin
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
        Returns:
            pd.DataFrame: Historical price data
        """
        if coin_id not in self.data:
            try:
                # Load data from CoinGecko API
                response = requests.get(
                    f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range',
                    params={
                        'vs_currency': 'usd',
                        'from': int(datetime.strptime(start_date, '%Y-%m-%d').timestamp()),
                        'to': int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
                    }
                )
                response.raise_for_status()
                
                data = response.json()
                df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Add additional metrics
                df['volume'] = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])[1]
                df['active_addresses'] = pd.DataFrame(data['active_addresses'], columns=['timestamp', 'active_addresses'])[1]
                df['transactions'] = pd.DataFrame(data['transactions'], columns=['timestamp', 'transactions'])[1]
                
                self.data[coin_id] = df
                
            except Exception as e:
                print(f"Error loading historical data for {coin_id}: {e}")
                return pd.DataFrame()
        
        return self.data[coin_id]

class Backtester:
    def __init__(self, bot, historical_data):
        self.bot = bot
        self.historical_data = historical_data
        self.results = []
        
    def run_backtest(self, coin_id, start_date, end_date):
        """
        Run backtest for a specific coin
        Args:
            coin_id (str): ID of the coin to test
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
        """
        # Load historical data
        df = self.historical_data.load_data(coin_id, start_date, end_date)
        if df.empty:
            return
            
        # Initialize bot with starting balance
        self.bot.initialize_backtest()
        
        # Simulate trading
        for timestamp, row in df.iterrows():
            # Update bot with current price
            self.bot.update_price(coin_id, row['price'])
            
            # Simulate trading
            action = self.bot.should_trade(coin_id, row['price'])
            if action:
                self.bot.execute_trade(coin_id, action, row['price'])
                
            # Update external data
            self.bot.update_external_data(coin_id, {
                'volume': row['volume'],
                'active_addresses': row['active_addresses'],
                'transactions': row['transactions']
            })
            
            # Store results
            self.results.append({
                'timestamp': timestamp,
                'price': row['price'],
                'action': action,
                'balance': self.bot.get_balance(),
                'coin_balance': self.bot.get_coin_balance(coin_id)
            })
        
    def get_results(self):
        """
        Get backtest results
        Returns:
            pd.DataFrame: Backtest results
        """
        return pd.DataFrame(self.results)

class TradingBot:
    def __init__(self, starting_balance=1000.0, mode='live'):
        """
        Initialize the trading bot with advanced parameters
        Args:
            starting_balance (float): Initial USD balance
            mode (str): 'live' or 'backtest'
        """
        self.starting_balance = starting_balance
        self.coin_balances = {}  # Dictionary to store balances for each coin
        self.last_prices = {}    # Dictionary to store last prices for each coin
        self.csv_file = 'trade_history.csv'
        self.initialize_csv()
        self.external_data = ExternalData()
        self.mode = mode
        self.prices = {}  # Store historical prices for backtesting
        self.last_notification = {}  # Store last notification times
        self.last_portfolio_value = self.starting_balance  # Store last portfolio value
        
        # Configure coins to trade
        self.coins = {
            'ethereum': {
                'id': 'ethereum',
                'volatility_window': 20,
                'trend_window': 10,
                'volatility_threshold': 0.02,  # 2% for stable coins
                'risk_factor': 0.01,
                'max_position_size': 0.5,
                'stop_loss_pct': 0.1,
                'take_profit_pct': 0.2,
                'sentiment_weight': 0.2,
                'volume_weight': 0.3,
                'network_weight': 0.1
            },
            'solana': {
                'id': 'solana',
                'volatility_window': 15,
                'trend_window': 7,
                'volatility_threshold': 0.03,  # 3% for volatile coins
                'risk_factor': 0.02,
                'max_position_size': 0.3,
                'stop_loss_pct': 0.15,
                'take_profit_pct': 0.25,
                'sentiment_weight': 0.3,
                'volume_weight': 0.4,
                'network_weight': 0.2
            },
            'polygon': {
                'id': 'polygon',
                'volatility_window': 10,
                'trend_window': 5,
                'volatility_threshold': 0.04,  # 4% for highly volatile coins
                'risk_factor': 0.03,
                'max_position_size': 0.2,
                'stop_loss_pct': 0.2,
                'take_profit_pct': 0.3,
                'sentiment_weight': 0.4,
                'volume_weight': 0.5,
                'network_weight': 0.3
            }
        }
        
        # Initialize CSV file for trade logging
        self.initialize_csv()
    
    def send_push_notification(self, message, title="Trading Bot Update"):
        """
        Send a push notification
        Args:
            message (str): Notification message
            title (str): Notification title
        """
        try:
            if pb:
                if config.PUSHBULLET_DEVICE_ID:
                    device.push_note(title, message)
                else:
                    pb.push_note(title, message)
        except Exception as e:
            print(f"Error sending push notification: {e}")
    
    def check_portfolio_changes(self):
        """
        Check for significant portfolio changes and send notifications
        """
        current_value = self.get_portfolio_value()
        change_pct = ((current_value - self.last_portfolio_value) / self.last_portfolio_value) * 100
        
        # Check if change exceeds threshold
        if abs(change_pct) >= config.PORTFOLIO_CHANGE_THRESHOLD:
            # Check if we've sent too many notifications recently
            now = datetime.now()
            if self.last_notification.get('portfolio', now - timedelta(hours=1)) < now - timedelta(hours=1):
                message = f"Portfolio value changed by {change_pct:.2f}%\n"
                message += f"Current value: ${current_value:.2f}\n"
                message += f"Previous value: ${self.last_portfolio_value:.2f}"
                
                self.send_push_notification(message, "Portfolio Change Alert")
                self.last_notification['portfolio'] = now
        
        self.last_portfolio_value = current_value
        
    def initialize_backtest(self):
        """
        Initialize bot for backtesting
        """
        self.coin_balances = {}
        self.last_prices = {}
        self.prices = {}
        self.initialize_csv()
        
    def update_price(self, coin_id, price):
        """
        Update price for backtesting
        Args:
            coin_id (str): ID of the coin
            price (float): Current price
        """
        self.last_prices[coin_id] = price
        if coin_id not in self.prices:
            self.prices[coin_id] = []
        self.prices[coin_id].append(price)
        
    def update_external_data(self, coin_id, data):
        """
        Update external data for backtesting
        Args:
            coin_id (str): ID of the coin
            data (dict): External data
        """
        if self.mode == 'backtest':
            # Store historical external data
            if coin_id not in self.external_data.cache.cache:
                self.external_data.cache.cache[coin_id] = ([], datetime.now())
            self.external_data.cache.cache[coin_id][0].append(data)

    def initialize_csv(self):
        """
        Create and initialize the CSV file with headers
        """
        with open(self.csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['timestamp', 'coin_id', 'action', 'price', 'amount', 'usd_balance', 'coin_balance'])

    def get_coin_price(self, coin_id):
        """
        Fetch current price for a specific coin from CoinGecko API
        Args:
            coin_id (str): ID of the coin to fetch price for
        Returns:
            float: Current coin price in USD
        """
        try:
            response = requests.get('https://api.coingecko.com/api/v3/simple/price',
                                 params={'ids': coin_id, 'vs_currencies': 'usd'})
            response.raise_for_status()
            return response.json()[coin_id]['usd']
        except requests.RequestException as e:
            print(f"Error fetching price for {coin_id}: {e}")
            return None

    def should_trade(self, coin_id, current_price):
        """
        Determine if we should buy or sell a specific coin based on multiple factors
        Args:
            coin_id (str): ID of the coin to analyze
            current_price (float): Current coin price
        Returns:
            str: 'buy', 'sell', or None
        """
        if self.last_prices.get(coin_id) is None:
            return None

        # Get coin configuration
        config = self.coins[coin_id]

        # Calculate price change
        price_change = (current_price - self.last_prices[coin_id]) / self.last_prices[coin_id]
        
        # Calculate volatility
        if coin_id not in self.prices:
            self.prices[coin_id] = []
        
        self.prices[coin_id].append(current_price)
        if len(self.prices[coin_id]) > config['volatility_window']:
            self.prices[coin_id].pop(0)
        
        if len(self.prices[coin_id]) >= 2:
            returns = pd.Series(self.prices[coin_id]).pct_change().dropna()
            volatility = returns.std()
        else:
            volatility = 0

        # Calculate trend
        if len(self.prices[coin_id]) >= config['trend_window']:
            trend = np.polyfit(range(config['trend_window']), self.prices[coin_id][-config['trend_window']:], 1)[0]
        else:
            trend = 0

        # Get external data
        sentiment = self.external_data.get_news_sentiment(config['id'])
        social_volume = self.external_data.get_social_volume(config['id'])
        exchange_volume = self.external_data.get_exchange_volume(config['id'])
        network_activity = self.external_data.get_network_activity(config['id'])

        # Calculate external factors score
        external_score = (
            sentiment * config['sentiment_weight'] +
            (np.log1p(social_volume) / 10) * config['volume_weight'] +
            (np.log1p(exchange_volume) / 1000000) * config['volume_weight'] +
            (network_activity['active_addresses'] / 100000) * config['network_weight'] +
            (network_activity['transactions'] / 10000) * config['network_weight']
        )

        # Adjust threshold based on volatility, coin type, and external factors
        threshold = config['volatility_threshold'] * (1 + volatility) * (1 + external_score)

        # Check if we should trade
        if price_change <= -threshold and trend < 0:
            return 'buy'
        elif price_change >= threshold and trend > 0:
            return 'sell'

        return None

    def execute_trade(self, coin_id, action, price):
        """
        Execute a trade for a specific coin
        Args:
            coin_id (str): ID of the coin to trade
            action (str): 'buy' or 'sell'
            price (float): Trade price
        """
        config = self.coins[coin_id]
        
        if action == 'buy':
            # Calculate position size based on volatility
            volatility = 0
            if coin_id in self.prices and len(self.prices[coin_id]) >= 2:
                returns = pd.Series(self.prices[coin_id]).pct_change().dropna()
                volatility = returns.std()
            
            # Calculate risk-adjusted position size
            position_size = self.starting_balance * config['max_position_size'] * (1 - volatility)
            position_size = min(position_size, self.starting_balance * config['max_position_size'])
            
            # Calculate amount to buy
            amount = position_size / price
            
            # Update balances
            self.coin_balances[coin_id] = self.coin_balances.get(coin_id, 0) + amount
            self.starting_balance -= position_size
            
            print(f"Bought {amount:.6f} {coin_id} at ${price:.2f}")
            
        elif action == 'sell':
            # Calculate amount to sell
            amount = self.coin_balances.get(coin_id, 0)
            position_size = amount * price
            
            # Update balances
            self.coin_balances[coin_id] = 0
            self.starting_balance += position_size
            
            print(f"Sold {amount:.6f} {coin_id} at ${price:.2f}")
            
        # Log the trade
        self.log_trade(coin_id, action, price)

    def check_stop_loss(self, coin_id, current_price):
        """
        Check and execute stop loss if needed
        Args:
            coin_id (str): ID of the coin to check
            current_price (float): Current coin price
        """
        if self.coin_balances.get(coin_id, 0) > 0 and current_price <= self.coins[coin_id]['stop_loss_pct'] * self.last_prices[coin_id]:
            print(f"Stop loss triggered for {coin_id} at ${current_price:.2f}")
            self.execute_trade(coin_id, 'sell', current_price)

    def check_take_profit(self, coin_id, current_price):
        """
        Check and execute take profit if needed
        Args:
            coin_id (str): ID of the coin to check
            current_price (float): Current coin price
        """
        if self.coin_balances.get(coin_id, 0) > 0 and current_price >= self.coins[coin_id]['take_profit_pct'] * self.last_prices[coin_id]:
            print(f"Take profit triggered for {coin_id} at ${current_price:.2f}")
            self.execute_trade(coin_id, 'sell', current_price)

    def log_trade(self, coin_id, action, price):
        """
        Log a trade to CSV file
        Args:
            coin_id (str): ID of the coin traded
            action (str): 'buy' or 'sell'
            price (float): Trade price
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                timestamp,
                coin_id,
                action,
                price,
                self.coin_balances.get(coin_id, 0) if action == 'buy' else 0,
                self.starting_balance,
                self.coin_balances.get(coin_id, 0)
            ])

    def run(self):
        """
        Run the trading bot with improved risk management
        """
        print(f"Starting trading bot with ${self.starting_balance} initial balance...")
        
        while True:
            try:
                # Check for killswitch
                if os.path.exists('killswitch.txt'):
                    print("\nKillswitch activated! Stopping trading...")
                    print("Closing all positions...")
                    
                    # Close all positions
                    for coin_id in self.coin_balances:
                        if self.coin_balances[coin_id] > 0:
                            current_price = self.get_coin_price(coin_id)
                            if current_price:
                                self.execute_trade(coin_id, 'sell', current_price)
                    
                    print("Trading stopped by killswitch")
                    break
                    
                # Process each coin
                for coin_id in self.coins:
                    # Get current price
                    current_price = self.get_coin_price(self.coins[coin_id]['id'])
                    if current_price is None:
                        continue
                    
                    # Check if we should trade
                    action = self.should_trade(coin_id, current_price)
                    if action:
                        self.execute_trade(coin_id, action, current_price)
                        bot.execute_trade(coin_id, action, current_price)
                        
                # Check for portfolio changes
                self.check_portfolio_changes()
                
                # Sleep for 5 minutes
                time.sleep(300)
                
            except KeyboardInterrupt:
                print("\nTrading stopped by user")
                break
            except Exception as e:
                print(f"Error in trading loop: {e}")
                time.sleep(60)

if __name__ == "__main__":
    bot = TradingBot()
    bot.run()
