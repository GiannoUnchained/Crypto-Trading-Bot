import os
import sys
import time
from datetime import datetime

def create_killswitch_file():
    """
    Create a killswitch file to stop the trading bot
    """
    try:
        with open('killswitch.txt', 'w') as f:
            f.write(f"Killswitch activated at {datetime.now()}")
        print("Killswitch activated! Trading bot will stop on next check.")
    except Exception as e:
        print(f"Error creating killswitch: {e}")

def check_killswitch():
    """
    Check if killswitch is active
    Returns:
        bool: True if killswitch is active, False otherwise
    """
    return os.path.exists('killswitch.txt')

def remove_killswitch():
    """
    Remove the killswitch file
    """
    try:
        if os.path.exists('killswitch.txt'):
            os.remove('killswitch.txt')
            print("Killswitch removed!")
    except Exception as e:
        print(f"Error removing killswitch: {e}")

if __name__ == "__main__":
    # Create killswitch if called directly
    create_killswitch_file()
