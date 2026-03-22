try:
    from src.strategies.ema_cross.runner import run
except ModuleNotFoundError:
    from strategies.ema_cross.runner import run

if __name__ == "__main__":
    run()
