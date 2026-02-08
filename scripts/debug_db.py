import sqlite3

conn = sqlite3.connect("data/market_data.db")
cur = conn.cursor()

cur.execute("SELECT COUNT(*) FROM ohlcv")
print("TOTAL OHLCV ROWS:", cur.fetchone()[0])

cur.execute("SELECT DISTINCT symbol FROM ohlcv LIMIT 20")
print("SYMBOLS:", cur.fetchall())

cur.execute("SELECT DISTINCT timeframe FROM ohlcv LIMIT 20")
print("TIMEFRAMES:", cur.fetchall())

cur.execute("SELECT * FROM ohlcv LIMIT 5")
print("SAMPLE ROWS:", cur.fetchall())

conn.close()
