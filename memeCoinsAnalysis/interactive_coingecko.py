#!/usr/bin/env python3
"""
Interactive CoinGecko Analyzer

Simple interactive script to analyze any cryptocurrency using CoinGecko data
and our trained LSTM model.

Usage:
    python3 interactive_coingecko.py
"""

import os
import sys
import warnings
import subprocess
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():
    """Interactive main function"""
    
    print("INTERACTIVE COINGECKO ANALYZER")
    print("=" * 50)
    print("Analyze any cryptocurrency using real-time CoinGecko data!")
    print("=" * 50)
    
    while True:
        try:
            # Get coin name from user
            print("\nEnter cryptocurrency name (e.g., bitcoin, ethereum, dogecoin):")
            coin_name = input("Coin: ").strip()
            
            if not coin_name:
                print("Please enter a coin name")
                continue
            
            # Get days parameter
            print("\nHow many days of data to analyze? (1-7 recommended)")
            days_input = input("Days (default: 1): ").strip()
            
            try:
                days = int(days_input) if days_input else 1
                if days < 1 or days > 30:
                    print("Using 1 day (valid range: 1-30)")
                    days = 1
            except ValueError:
                days = 1
            
            # Run the analysis
            print(f"\nAnalyzing {coin_name} with {days} day(s) of data...")
            print("-" * 50)
            
            # Call the coingecko_analyzer.py script
            cmd = [
                sys.executable, 
                "coingecko_analyzer.py", 
                "--coin", coin_name, 
                "--days", str(days)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
            
            if result.returncode == 0:
                print(result.stdout)
            else:
                print("Analysis failed:")
                print(result.stderr)
            
            # Ask if user wants to continue
            print("\n" + "=" * 50)
            continue_analysis = input("Analyze another coin? (y/n): ").strip().lower()
            
            if continue_analysis not in ['y', 'yes']:
                break
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            continue
    
    print("\nThanks for using Interactive CoinGecko Analyzer!")

if __name__ == "__main__":
    main()
