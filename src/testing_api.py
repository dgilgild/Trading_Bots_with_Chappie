from dotenv import load_dotenv
import os

load_dotenv()

print(os.getenv("BINANCE_API_KEY"))