import logging
import asyncio
import pandas as pd
from binance.client import AsyncClient
from telegram import Bot
import config  # Import the configuration settings
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define intervals and their display names
INTERVALS = {
    '1h': '1 hour',
    '4h': '4 hours',
    '1d': '1 day',
    '1w': '1 week'
}

# Define periods to check for swing lows
SWING_PERIODS = [25, 50, 150, 300, 500]

# Dictionary to store the last swing low prices
last_swing_lows = {interval: {period: {} for period in SWING_PERIODS} for interval in INTERVALS.keys()}

# Load portfolio from CSV
def load_portfolio(file_path):
    df = pd.read_csv(file_path)
    return {row['symbol']: row['initial_price'] for _, row in df.iterrows()}

# Load portfolio at runtime
PORTFOLIO = load_portfolio('portfolio.csv')

# Percentage thresholds for notifications
PERCENT_THRESHOLDS = [0.5, 1.0]  # 50% and 100%

async def get_symbols_from_binance(client):
    try:
        exchange_info = await client.get_exchange_info()
        symbols = [s['symbol'] for s in exchange_info['symbols'] if s['status'] == 'TRADING' and s['symbol'].endswith('USDT')]
        return symbols
    except Exception as e:
        logging.error(f"Error fetching symbols from Binance: {e}")
        return []

async def get_new_swing_lows(client, interval):
    new_swing_lows = []  # Initialize with an empty list
    try:
        symbols = await get_symbols_from_binance(client)

        tasks = []
        for symbol in symbols:
            tasks.append(fetch_klines(client, symbol, interval))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                logging.error(f"Error fetching data: {result}")
                continue
            symbol, df = result
            try:
                df['open'] = df['open'].astype(float)
                df['high'] = df['high'].astype(float)
                df['low'] = df['low'].astype(float)
                df['close'] = df['close'].astype(float)
                df['rsi'] = calculate_rsi(df['close'])
                df['momentum'] = calculate_momentum(df['close'])
                df['atr'] = calculate_atr(df)  # Updated to use pandas-based ATR calculation
                df['volatility'] = calculate_volatility(df)
                df['MA20'] = df['close'].rolling(window=20).mean()
                df['MA50'] = df['close'].rolling(window=50).mean()
                df['MA100'] = df['close'].rolling(window=100).mean()
                df['MA200'] = df['close'].rolling(window=200).mean()
                df['stochastic_k'], df['stochastic_d'] = calculate_stochastic(df)

                for period in SWING_PERIODS:
                    if len(df) >= period:
                        recent_df = df[-period:]
                        lowest_price_last_period = recent_df['low'].min()
                        swing_lows = recent_df[(recent_df['low'] < recent_df['low'].shift(1)) & (recent_df['low'] < recent_df['low'].shift(-1))]
                        
                        if not swing_lows.empty:
                            latest_swing_low = swing_lows['low'].iloc[-1]
                            latest_rsi = swing_lows['rsi'].iloc[-1]
                            latest_momentum = swing_lows['momentum'].iloc[-1]
                            latest_atr = swing_lows['atr'].iloc[-1]
                            latest_volatility = swing_lows['volatility'].iloc[-1]
                            latest_ma20 = swing_lows['MA20'].iloc[-1]
                            latest_ma50 = swing_lows['MA50'].iloc[-1]
                            latest_ma100 = swing_lows['MA100'].iloc[-1]
                            latest_ma200 = swing_lows['MA200'].iloc[-1]
                            latest_stochastic_k = swing_lows['stochastic_k'].iloc[-1]
                            latest_stochastic_d = swing_lows['stochastic_d'].iloc[-1]

                            # Compare with the last stored swing low price
                            if symbol not in last_swing_lows[interval][period]:
                                last_swing_lows[interval][period][symbol] = latest_swing_low
                                new_swing_lows.append((symbol, latest_swing_low, lowest_price_last_period, latest_rsi, latest_momentum, latest_atr, latest_volatility, interval, period, latest_ma20, latest_ma50, latest_ma100, latest_ma200, latest_stochastic_k, latest_stochastic_d))
                            elif latest_swing_low <= last_swing_lows[interval][period][symbol]:
                                new_swing_lows.append((symbol, latest_swing_low, lowest_price_last_period, latest_rsi, latest_momentum, latest_atr, latest_volatility, interval, period, latest_ma20, latest_ma50, latest_ma100, latest_ma200, latest_stochastic_k, latest_stochastic_d))
                                last_swing_lows[interval][period][symbol] = latest_swing_low

            except Exception as e:
                logging.error(f"Error processing data for {symbol}: {e}")
    except Exception as e:
        logging.error(f"Error fetching exchange info: {e}")

    return new_swing_lows

async def fetch_klines(client, symbol, interval):
    try:
        klines = await client.get_klines(symbol=symbol, interval=interval, limit=max(SWING_PERIODS))
        df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume',
                                           'close_time', 'quote_asset_volume', 'number_of_trades',
                                           'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        return symbol, df
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")
        return e

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_momentum(series, period=14):
    return series.diff(period)

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

def calculate_volatility(df):
    close = df['close'].values
    volatility = close.std()
    return volatility

def classify_momentum(momentum):
    if abs(momentum) < 0.1:
        return "Very Little"
    elif abs(momentum) < 1:
        return "Medium"
    else:
        return "High"

def calculate_stochastic(df, period=14, smooth_k=3, smooth_d=3):
    lowest_low = df['low'].rolling(window=period).min()
    highest_high = df['high'].rolling(window=period).max()
    stochastic_k = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
    stochastic_d = stochastic_k.rolling(window=smooth_d).mean()
    return stochastic_k, stochastic_d

async def send_telegram_message(bot, message, chat_id, pin_message=False, max_retries=5):
    for attempt in range(max_retries):
        try:
            sent_message = await bot.send_message(chat_id=chat_id, text=message, parse_mode='Markdown')
            if pin_message:
                await bot.pin_chat_message(chat_id=chat_id, message_id=sent_message.message_id)
            return
        except Exception as e:
            wait_time = 2 ** attempt  # Exponential backoff
            logging.error(f"Error sending Telegram message: {e}. Retrying in {wait_time} seconds...")
            await asyncio.sleep(wait_time)
    logging.error("Failed to send Telegram message after multiple attempts.")

def determine_entry_amount(symbol, current_price, dca_budget, dca_intervals):
    """Determine the entry amount based on DCA strategy."""
    return dca_budget / dca_intervals

def determine_exit_price(current_price, atr):
    """Determine the exit price based on ATR (Average True Range)."""
    return current_price + 2 * atr  # Example exit strategy based on ATR

def analyze_coin_type(symbol):
    """Analyze coin type based on its symbol."""
    if "BTC" in symbol or "ETH" in symbol:
        return "Stable"  # Assuming BTC and ETH as stable
    else:
        return "Altcoin"

async def main():
    try:
        # Retrieve TELEGRAM_TOKEN from config
        telegram_token = config.TELEGRAM_TOKEN

        # Initialize Binance Client
        client = await AsyncClient.create(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)

        # Initialize Telegram Bot
        bot = Bot(token=telegram_token)

        dca_budget = 1000  # Total budget for DCA
        dca_intervals = 5  # Number of intervals for DCA

        while True:
            all_conditions_met = True

            # Monitor Bitcoin movements
            btc_price = await get_current_price(client, 'BTCUSDT')
            if not btc_price:
                logging.error("Failed to fetch Bitcoin price.")
                continue

            # Check portfolio for price changes
            for symbol, initial_price in PORTFOLIO.items():
                current_price = await get_current_price(client, symbol)
                if not current_price:
                    logging.error(f"Failed to fetch price for {symbol}.")
                    continue
                
                current_price = float(current_price)
                for threshold in PERCENT_THRESHOLDS:
                    change = (current_price - initial_price) / initial_price
                    if abs(change) >= threshold:
                        change_percent = change * 100
                        message = (f"🔔 *Price Alert for {symbol}*\n"
                                   f"Initial Price: {initial_price}\n"
                                   f"Current Price: {current_price}\n"
                                   f"Change: {change_percent:.2f}%")
                        await send_telegram_message(bot, message, config.TELEGRAM_CHAT_ID)
                        # Update the initial price to avoid repetitive notifications
                        PORTFOLIO[symbol] = current_price

            for interval in INTERVALS.keys():
                try:
                    # Fetch new swing lows asynchronously
                    new_swing_lows = await get_new_swing_lows(client, interval=interval)
                    
                    # Check if new_swing_lows is None or empty before iterating
                    if new_swing_lows:  # Ensure it's not empty
                        for symbol, new_swing_low, lowest_price_last_period, latest_rsi, latest_momentum, latest_atr, latest_volatility, interval, period, latest_ma20, latest_ma50, latest_ma100, latest_ma200, latest_stochastic_k, latest_stochastic_d in new_swing_lows:
                            current_price = await get_current_price(client, symbol)
                            
                            if current_price and float(current_price) < lowest_price_last_period:
                                fundamentals_good = True  # Placeholder for real fundamental analysis
                                
                                # Add checks for technical analysis including Stochastic Oscillator
                                technical_good = (latest_rsi < 30 and latest_momentum < 0.1 and latest_atr > 0 and 
                                                  current_price > latest_ma20 > latest_ma50 > latest_ma100 > latest_ma200 and
                                                  latest_stochastic_k < 20 and latest_stochastic_k > latest_stochastic_d)
                                
                                momentum_classification = classify_momentum(latest_momentum)
                                
                                # Determine DCA entry amount
                                entry_amount = determine_entry_amount(symbol, current_price, dca_budget, dca_intervals)
                                
                                # Determine exit price based on ATR
                                exit_price = determine_exit_price(current_price, latest_atr)
                                
                                # Analyze coin type
                                coin_type = analyze_coin_type(symbol)

                                if fundamentals_good and technical_good:
                                    message = (f"📊 *New Swing Low Detected!*\n"
                                               f"Symbol: `{symbol}` ({coin_type})\n"
                                               f"💰 *Current Price:* `{current_price}`\n"
                                               f"🟢 *NEW SWING LOW PRICE NOW ({period} CANDLES):* `{lowest_price_last_period}`\n"
                                               f"📈 *RSI:* `{latest_rsi:.2f}`\n"
                                               f"📉 *Momentum:* `{momentum_classification}`\n"
                                               f"🔶 *ATR (14):* `{latest_atr:.2f}`\n"
                                               f"📉 *Volatility:* `{latest_volatility:.2f}`\n"
                                               f"📊 *MA20:* `{latest_ma20:.2f}`\n"
                                               f"📊 *MA50:* `{latest_ma50:.2f}`\n"
                                               f"📊 *MA100:* `{latest_ma100:.2f}`\n"
                                               f"📊 *MA200:* `{latest_ma200:.2f}`\n"
                                               f"📊 *Stochastic K:* `{latest_stochastic_k:.2f}`\n"
                                               f"📊 *Stochastic D:* `{latest_stochastic_d:.2f}`\n"
                                               f"💵 *Recommended Entry (DCA):* `{entry_amount}` USDT\n"
                                               f"🏷️ *Suggested Exit Price:* `{exit_price}` USDT\n"
                                               f"⏰ *Timeframe:* {INTERVALS[interval]}\n"
                                               f"📈 *BTC Price:* `{btc_price}`\n"
                                               f"✅ *Conditions Met: Perfect for Buy!*")
                                else:
                                    message = (f"📊 *New Swing Low Detected!*\n"
                                               f"Symbol: `{symbol}` ({coin_type})\n"
                                               f"💰 *Current Price:* `{current_price}`\n"
                                               f"🟢 *NEW SWING LOW PRICE NOW ({period} CANDLES):* `{lowest_price_last_period}`\n"
                                               f"📈 *RSI:* `{latest_rsi:.2f}`\n"
                                               f"📉 *Momentum:* `{momentum_classification}`\n"
                                               f"🔶 *ATR (14):* `{latest_atr:.2f}`\n"
                                               f"📉 *Volatility:* `{latest_volatility:.2f}`\n"
                                               f"📊 *MA20:* `{latest_ma20:.2f}`\n"
                                               f"📊 *MA50:* `{latest_ma50:.2f}`\n"
                                               f"📊 *MA100:* `{latest_ma100:.2f}`\n"
                                               f"📊 *MA200:* `{latest_ma200:.2f}`\n"
                                               f"📊 *Stochastic K:* `{latest_stochastic_k:.2f}`\n"
                                               f"📊 *Stochastic D:* `{latest_stochastic_d:.2f}`\n"
                                               f"⏰ *Timeframe:* {INTERVALS[interval]}\n"
                                               f"📈 *BTC Price:* `{btc_price}`\n"
                                               f"⚠️ *Conditions Not Met: Not Ideal for Buy*")
                                
                                await send_telegram_message(bot, message, config.TELEGRAM_CHAT_ID)
                            else:
                                logging.info(f"No new swing low for {symbol} in {interval} interval with period {period}. Current price: {current_price}, Lowest price in last {period} candles: {lowest_price_last_period}")
                    else:
                        logging.info(f"No new swing lows found in {interval} interval.")

                except asyncio.TimeoutError as te:
                    logging.error(f"Timeout error occurred: {te}")
                except Exception as e:
                    logging.error(f"An error occurred in {interval} interval: {e}")

            await asyncio.sleep(10)  # Adjust sleep time as necessary

    except Exception as e:
        logging.error(f"Main loop error: {e}")

    finally:
        await client.close_connection()

async def get_current_price(client, symbol):
    try:
        ticker = await client.get_symbol_ticker(symbol=symbol)
        return ticker['price']
    except Exception as e:
        logging.error(f"Error fetching current price for {symbol}: {e}")
        return None

if __name__ == "__main__":
    asyncio.run(main())
