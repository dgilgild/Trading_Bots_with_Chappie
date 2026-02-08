import ccxt
import os
from dotenv import load_dotenv

load_dotenv()

def get_exchange():
    return ccxt.binance({
        "apiKey": os.getenv("BINANCE_API_KEY"),
        "secret": os.getenv("BINANCE_API_SECRET"),
        "enableRateLimit": True
    })
