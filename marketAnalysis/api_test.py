"""
CoinGecko API Fix Script - Handles both Free and Pro API properly
"""

import requests
import os
import time
from dotenv import load_dotenv

load_dotenv()


def test_api_key_type():
    """Determine if API key is Pro or Demo"""
    api_key = os.getenv('COINGECKO_API_KEY', '')

    print("=" * 60)
    print("DETECTING API KEY TYPE")
    print("=" * 60)

    if not api_key or api_key == 'your_api_key_here':
        print("❌ No API key found - will use FREE tier")
        return 'free', None, "https://api.coingecko.com/api/v3"

    print(f"Testing API key: {api_key[:10]}...")

    # Test 1: Try Pro API endpoint with x-cg-pro-api-key header
    print("\n1. Testing as Pro API key...")
    headers = {'x-cg-pro-api-key': api_key}
    try:
        response = requests.get(
            "https://pro-api.coingecko.com/api/v3/ping",
            headers=headers,
            timeout=10
        )
        if response.status_code == 200:
            print("✅ Valid PRO API key detected!")
            return 'pro', api_key, "https://pro-api.coingecko.com/api/v3"
        else:
            print(f"   Not a Pro key: {response.status_code}")
    except Exception as e:
        print(f"   Pro test failed: {e}")

    # Test 2: Try Demo API key with x-cg-demo-api-key header
    print("\n2. Testing as Demo API key...")
    headers = {'x-cg-demo-api-key': api_key}
    try:
        response = requests.get(
            "https://api.coingecko.com/api/v3/ping",
            headers=headers,
            timeout=10
        )
        if response.status_code == 200:
            print("✅ Valid DEMO API key detected!")
            return 'demo', api_key, "https://api.coingecko.com/api/v3"
        else:
            print(f"   Not a Demo key: {response.status_code}")
    except Exception as e:
        print(f"   Demo test failed: {e}")

    # Test 3: Try without any API key (free tier)
    print("\n3. Testing free tier (no API key)...")
    try:
        response = requests.get(
            "https://api.coingecko.com/api/v3/ping",
            timeout=10
        )
        if response.status_code == 200:
            print("✅ Free tier access works!")
            return 'free', None, "https://api.coingecko.com/api/v3"
    except Exception as e:
        print(f"   Free tier test failed: {e}")

    print("\n❌ Could not determine API key type")
    return 'unknown', None, "https://api.coingecko.com/api/v3"


def get_correct_headers(api_type, api_key):
    """Get correct headers based on API type"""
    headers = {'accept': 'application/json'}

    if api_type == 'pro' and api_key:
        headers['x-cg-pro-api-key'] = api_key
    elif api_type == 'demo' and api_key:
        headers['x-cg-demo-api-key'] = api_key
    # For free tier, no API key header needed

    return headers


def test_working_endpoints(api_type, api_key, base_url):
    """Test which endpoints work with your setup"""
    print("\n" + "=" * 60)
    print("TESTING WORKING ENDPOINTS")
    print("=" * 60)

    headers = get_correct_headers(api_type, api_key)
    working_endpoints = []

    # List of endpoints to test
    endpoints = [
        ('Ping', '/ping', {}),
        ('Simple Price', '/simple/price', {'ids': 'bitcoin,dogecoin', 'vs_currencies': 'usd'}),
        ('Coin List', '/coins/list', {}),
        ('Specific Coin', '/coins/dogecoin', {'localization': 'false', 'market_data': 'true'}),
        ('Market Data', '/coins/markets', {
            'vs_currency': 'usd',
            'ids': 'dogecoin,shiba-inu',
            'order': 'market_cap_desc',
            'per_page': 10,
            'page': 1
        }),
        ('OHLC Data', '/coins/dogecoin/ohlc', {'vs_currency': 'usd', 'days': '1'}),
    ]

    for name, endpoint, params in endpoints:
        print(f"\nTesting {name}...")
        try:
            url = base_url + endpoint
            response = requests.get(url, headers=headers, params=params, timeout=10)

            if response.status_code == 200:
                print(f"  ✅ {name}: SUCCESS")
                working_endpoints.append((name, endpoint))

                # Show sample data for some endpoints
                if 'simple/price' in endpoint:
                    data = response.json()
                    if 'bitcoin' in data:
                        print(f"     BTC: ${data['bitcoin']['usd']:,.0f}")
                    if 'dogecoin' in data:
                        print(f"     DOGE: ${data['dogecoin']['usd']:.4f}")

            elif response.status_code == 429:
                print(f"  ⚠️  {name}: RATE LIMITED")
            elif response.status_code == 401:
                print(f"  ❌ {name}: UNAUTHORIZED")
            else:
                print(f"  ❌ {name}: FAILED ({response.status_code})")

        except Exception as e:
            print(f"  ❌ {name}: ERROR - {str(e)[:50]}")

        time.sleep(0.5)  # Rate limiting

    return working_endpoints


def generate_fixed_config(api_type, api_key, base_url):
    """Generate fixed configuration files"""
    print("\n" + "=" * 60)
    print("GENERATING FIXED CONFIGURATION")
    print("=" * 60)

    # Fixed .env content
    env_content = f"""# Fixed CoinGecko Configuration
COINGECKO_BASE_URL={base_url}
COINGECKO_API_KEY={api_key if api_key else ''}
API_TYPE={api_type.upper()}
API_RATE_LIMIT_SECONDS=2

# Target Coins
TARGET_COINS=dogecoin,shiba-inu,pepe,floki,bonk

# Model Settings
HISTORICAL_DAYS=30
SEQUENCE_LENGTH=20
PREDICTION_HORIZON=1
TRENDING_THRESHOLD=0.001

# Training
BATCH_SIZE=32
EPOCHS=30
VALIDATION_SPLIT=0.2
MODEL_SAVE_PATH=./models/meme_coin_market_model

# Production
USE_COOKING=false
LOG_FILE=logs/meme_analysis.log
LOG_LEVEL=INFO
"""

    print("📝 Fixed .env configuration:")
    print("-" * 40)
    print(env_content)
    print("-" * 40)

    # Save to file
    with open('.env.fixed', 'w') as f:
        f.write(env_content)
    print("\n✅ Saved to .env.fixed")

    # Generate fixed data_collector snippet
    collector_fix = f"""
# Add this to your DataCollector class in data_collector.py

def _get_headers(self):
    '''Get correct headers based on API type'''
    headers = {{'User-Agent': 'Trading-Assistant/1.0', 'Accept': 'application/json'}}

    api_type = os.getenv('API_TYPE', 'free').lower()
    api_key = self.api_key

    if api_type == 'pro' and api_key:
        headers['x-cg-pro-api-key'] = api_key
    elif api_type == 'demo' and api_key:
        headers['x-cg-demo-api-key'] = api_key
    # No key header for free tier

    return headers

def test_connection(self):
    '''Test connection with proper endpoint'''
    print(f"Testing connection...")
    try:
        url = f"{{self.base_url}}/simple/price"
        params = {{'ids': 'bitcoin', 'vs_currencies': 'usd'}}
        response = self._make_request(url, params)

        if response and 'bitcoin' in response:
            print(f"✅ Connection successful! BTC: ${{response['bitcoin']['usd']:,.0f}}")
            return True
        return False
    except Exception as e:
        print(f"❌ Connection failed: {{e}}")
        return False
"""

    print("\n📝 Fix for data_collector.py:")
    print("-" * 40)
    print(collector_fix)
    print("-" * 40)


def test_meme_coins_fetch(api_type, api_key, base_url):
    """Test fetching meme coins with the working method"""
    print("\n" + "=" * 60)
    print("TESTING MEME COINS FETCH")
    print("=" * 60)

    headers = get_correct_headers(api_type, api_key)

    # Method 1: Fetch specific meme coins
    print("\nMethod 1: Fetching specific meme coins...")
    meme_coin_ids = 'dogecoin,shiba-inu,pepe,floki,bonk,dogwifhat'

    try:
        response = requests.get(
            f"{base_url}/coins/markets",
            headers=headers,
            params={
                'vs_currency': 'usd',
                'ids': meme_coin_ids,
                'order': 'market_cap_desc',
                'per_page': 20,
                'page': 1,
                'sparkline': False
            },
            timeout=10
        )

        if response.status_code == 200:
            coins = response.json()
            print(f"✅ Successfully fetched {len(coins)} meme coins:")
            for coin in coins[:5]:
                print(f"   - {coin['symbol'].upper()}: {coin['name']} (${coin['market_cap'] / 1e6:.1f}M)")
            return True
        else:
            print(f"❌ Failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    print("🔧 COINGECKO API DIAGNOSTIC & FIX TOOL")
    print("=" * 60)

    # Step 1: Detect API key type
    api_type, api_key, base_url = test_api_key_type()

    # Step 2: Test working endpoints
    working = test_working_endpoints(api_type, api_key, base_url)

    # Step 3: Test meme coins fetch
    meme_coins_work = test_meme_coins_fetch(api_type, api_key, base_url)

    # Step 4: Generate fixed configuration
    generate_fixed_config(api_type, api_key, base_url)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✓ API Type: {api_type.upper()}")
    print(f"✓ Base URL: {base_url}")
    print(f"✓ Working Endpoints: {len(working)}")
    print(f"✓ Meme Coins Fetch: {'✅ Working' if meme_coins_work else '❌ Not Working'}")

    print("\n📌 NEXT STEPS:")
    print("1. Replace your .env with .env.fixed:")
    print("   mv .env .env.backup && mv .env.fixed .env")
    print("2. Update data_collector.py with the fixes shown above")
    print("3. Run: python quickstart.py")

    if api_type == 'free':
        print("\n⚠️  NOTE: You're using the FREE tier which has:")
        print("   - Rate limit: 10-50 calls/minute")
        print("   - No category filtering")
        print("   - Limited endpoints")
        print("   Consider getting a Demo or Pro API key for better access")


if __name__ == "__main__":
    main()
