import os
import requests
import json
import time
from typing import List, Dict, Optional
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

class CoinGeckoAPI:
    """
    CoinGecko Pro API client for fetching coins by categories
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://pro-api.coingecko.com/api/v3"
        self.headers = {
            "accept": "application/json",
            "x-cg-pro-api-key": api_key
        }

    def get_categories(self) -> List[Dict]:
        """
        Fetch all available categories from CoinGecko

        Returns:
            List of category dictionaries with id, name, and market data
        """
        url = f"{self.base_url}/coins/categories"

        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching categories: {e}")
            return []

    def get_coins_by_category(self, category_id: str, order: str = "market_cap_desc",
                              per_page: int = 100, page: int = 1) -> List[Dict]:
        """
        Fetch coins belonging to a specific category

        Args:
            category_id: The category ID (e.g., "decentralized-finance-defi")
            order: Sort order ("market_cap_desc", "market_cap_asc", "name_asc", "name_desc")
            per_page: Number of results per page (max 250)
            page: Page number

        Returns:
            List of coin dictionaries
        """
        url = f"{self.base_url}/coins/markets"
        params = {
            "vs_currency": "usd",
            "category": category_id,
            "order": order,
            "per_page": per_page,
            "page": page,
            "sparkline": "false",
            "price_change_percentage": "1h,24h,7d"
        }

        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching coins for category {category_id}: {e}")
            return []

    def get_all_coins_by_category(self, category_id: str, order: str = "market_cap_desc") -> List[Dict]:
        """
        Fetch ALL coins from a specific category using pagination

        Args:
            category_id: The category ID (e.g., "decentralized-finance-defi")
            order: Sort order ("market_cap_desc", "market_cap_asc", "name_asc", "name_desc")

        Returns:
            List of ALL coin dictionaries from the category
        """
        all_coins = []
        page = 1
        per_page = 250  # Maximum allowed by API
        
        while True:
            print(f"Fetching page {page} for category: {category_id}")
            coins = self.get_coins_by_category(category_id, order=order, per_page=per_page, page=page)
            
            if not coins:  # No more coins available
                break
                
            all_coins.extend(coins)
            print(f"  Found {len(coins)} coins on page {page} (total so far: {len(all_coins)})")
            
            # If we got fewer coins than per_page, we've reached the end
            if len(coins) < per_page:
                break
                
            page += 1
            
            # Add delay to respect rate limits
            time.sleep(0.5)
        
        print(f"Total coins fetched for {category_id}: {len(all_coins)}")
        return all_coins

    def get_multiple_categories_coins(self, category_ids: List[str],
                                      fetch_all: bool = True) -> Dict[str, List[Dict]]:
        """
        Fetch coins from multiple categories

        Args:
            category_ids: List of category IDs
            fetch_all: If True, fetch ALL coins from each category. If False, fetch only top 50.

        Returns:
            Dictionary with category_id as key and list of coins as value
        """
        results = {}

        for category_id in category_ids:
            print(f"\n{'='*50}")
            print(f"Fetching coins for category: {category_id}")
            print(f"{'='*50}")
            
            if fetch_all:
                coins = self.get_all_coins_by_category(category_id)
            else:
                coins = self.get_coins_by_category(category_id, per_page=50)
                
            results[category_id] = coins

            # Add delay between categories to respect rate limits
            time.sleep(1)

        return results

    def search_categories(self, search_term: str) -> List[Dict]:
        """
        Search for categories containing a specific term

        Args:
            search_term: Term to search for in category names

        Returns:
            List of matching categories
        """
        all_categories = self.get_categories()
        matching_categories = []

        for category in all_categories:
            if search_term.lower() in category.get('name', '').lower():
                matching_categories.append(category)

        return matching_categories


def main():
    """
    Example usage of the CoinGecko API client
    """
    # Replace with your actual CoinGecko Pro API key
    API_KEY = os.getenv('COINGECKO_API_KEY')

    # Initialize the API client
    cg = CoinGeckoAPI(API_KEY)

    # Example 1: Get all categories
    print("Fetching all categories...")
    categories = cg.get_categories()
    print(f"Found {len(categories)} categories")

    # Display first 10 categories
    print("\nTop 10 categories by market cap:")
    for i, cat in enumerate(categories[:10]):
        print(f"{i + 1}. {cat.get('name', 'N/A')} (ID: {cat.get('id', 'N/A')})")
        print(f"   Market Cap: ${cat.get('market_cap', 0):,.2f}")
        print(f"   Volume 24h: ${cat.get('volume_24h', 0):,.2f}")
        print()

    # Example 2: Get coins from specific categories
    target_categories = [
        "letsbonk-fun-ecosystem",
        "pump-fun",
        "moonshot-ecosystem"
    ]

    print("Fetching ALL coins from specific categories...")
    category_coins = cg.get_multiple_categories_coins(target_categories, fetch_all=True)

    # Display results
    for category_id, coins in category_coins.items():
        print(f"\n--- {category_id.upper()} ---")
        print(f"Found {len(coins)} coins")

        for i, coin in enumerate(coins[:5]):  # Show top 5 coins
            print(f"{i + 1}. {coin.get('name', 'N/A')} ({coin.get('symbol', 'N/A').upper()})")
            print(f"   Price: ${coin.get('current_price', 0):.6f}")
            print(f"   Market Cap: ${coin.get('market_cap', 0):,.0f}")
            print(f"   24h Change: {coin.get('price_change_percentage_24h', 0):.2f}%")

    # Example 3: Search for specific categories
    print("\nSearching for 'gaming' categories...")
    gaming_categories = cg.search_categories("gaming")
    for cat in gaming_categories:
        print(f"- {cat.get('name', 'N/A')} (ID: {cat.get('id', 'N/A')})")

    # Example 4: Export to CSV
    if category_coins:
        print("\nExporting data to CSV...")
        all_coins_data = []

        for category_id, coins in category_coins.items():
            for coin in coins:
                coin_data = {
                    'category': category_id,
                    'name': coin.get('name', ''),
                    'symbol': coin.get('symbol', ''),
                    'current_price': coin.get('current_price', 0),
                    'market_cap': coin.get('market_cap', 0),
                    'market_cap_rank': coin.get('market_cap_rank', 0),
                    'volume_24h': coin.get('total_volume', 0),
                    'price_change_24h': coin.get('price_change_percentage_24h', 0),
                    'price_change_7d': coin.get('price_change_percentage_7d', 0)
                }
                all_coins_data.append(coin_data)

        df = pd.DataFrame(all_coins_data)
        df.to_csv('coingecko_category_coins.csv', index=False)
        print("Data exported to 'coingecko_category_coins.csv'")


def get_top_coins_by_category(api_key: str, category_id: str, top_n: int = 10) -> List[Dict]:
    """
    Convenience function to get top N coins from a specific category

    Args:
        api_key: CoinGecko Pro API key
        category_id: Category ID to fetch coins from
        top_n: Number of top coins to return

    Returns:
        List of top coins from the category
    """
    cg = CoinGeckoAPI(api_key)
    return cg.get_coins_by_category(category_id, per_page=top_n)


if __name__ == "__main__":
    # Check if required packages are installed
    try:
        import pandas as pd
    except ImportError:
        print("pandas not installed. Install it with: pip install pandas")
        print("Running without CSV export functionality...")

    main()
