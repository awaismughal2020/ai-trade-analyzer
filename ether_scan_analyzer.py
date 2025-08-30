import requests
import json
import os
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional
import csv
import time
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EtherscanAnalyzer:
    """
    A comprehensive class for analyzing Ethereum transactions using Etherscan API with smart pagination
    """

    def __init__(self, api_key: str = None):
        """
        Initialize the analyzer with Etherscan API key

        Args:
            api_key (str, optional): Your Etherscan API key. If not provided,
                                   will try to get from ETHERSCAN_API_KEY environment variable
        """
        self.api_key = api_key or os.getenv('ETHERSCAN_API_KEY')
        if not self.api_key:
            raise ValueError(
                "API key is required. Provide it as parameter or set ETHERSCAN_API_KEY environment variable")

        self.base_url = "https://api.etherscan.io/api"
        self.session = requests.Session()

        # Popular DEX contract addresses for identifying trades
        self.dex_contracts = {
            '0x7a250d5630b4cf539739df2c5dacb4c659f2488d': 'Uniswap V2',
            '0xe592427a0aece92de3edee1f18e0157c05861564': 'Uniswap V3',
            '0xd9e1ce17f2641f24ae83637ab66a2cca9c378b9f': 'SushiSwap',
            '0x881d40237659c251811cec9c364ef91dc08d300c': 'Metamask Swap',
            '0x1111111254fb6c44bac0bed2854e76f90643097d': '1inch Router',
            '0xdef1c0ded9bec7f1a1670819833240f027b25eff': '0x Protocol',
            '0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45': 'Uniswap Universal Router'
        }

        logger.info(f"EtherscanAnalyzer initialized with API key: {self.api_key[:8]}...")

    def _make_request(self, params: Dict) -> Optional[Dict]:
        """
        Make API request to Etherscan with error handling

        Args:
            params (Dict): API parameters

        Returns:
            Optional[Dict]: API response data or None if error
        """
        params['apikey'] = self.api_key

        try:
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()

            data = response.json()

            if data['status'] == '1':
                return data['result']
            else:
                logger.error(f"API Error: {data.get('message', 'Unknown error')}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Request Error: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON Decode Error: {e}")
            return None

    def _get_normal_transactions(self, address: str, limit: int = 100, page: int = 1) -> List[Dict]:
        """
        Get normal ETH transactions for an address with pagination

        Args:
            address (str): Ethereum address
            limit (int): Maximum number of transactions to fetch per page
            page (int): Page number for pagination

        Returns:
            List[Dict]: List of transaction data
        """
        params = {
            'module': 'account',
            'action': 'txlist',
            'address': address,
            'startblock': 0,
            'endblock': 99999999,
            'page': page,
            'offset': limit,
            'sort': 'desc'
        }

        result = self._make_request(params)
        return result if result else []

    def _get_token_transactions(self, address: str, limit: int = 100, page: int = 1) -> List[Dict]:
        """
        Get ERC-20 token transactions for an address with pagination

        Args:
            address (str): Ethereum address
            limit (int): Maximum number of transactions to fetch per page
            page (int): Page number for pagination

        Returns:
            List[Dict]: List of token transaction data
        """
        params = {
            'module': 'account',
            'action': 'tokentx',
            'address': address,
            'page': page,
            'offset': limit,
            'sort': 'desc'
        }

        result = self._make_request(params)
        return result if result else []

    def _get_internal_transactions(self, address: str, limit: int = 100, page: int = 1) -> List[Dict]:
        """
        Get internal transactions for an address with pagination

        Args:
            address (str): Ethereum address
            limit (int): Maximum number of transactions to fetch per page
            page (int): Page number for pagination

        Returns:
            List[Dict]: List of internal transaction data
        """
        params = {
            'module': 'account',
            'action': 'txlistinternal',
            'address': address,
            'page': page,
            'offset': limit,
            'sort': 'desc'
        }

        result = self._make_request(params)
        return result if result else []

    def _determine_transaction_type(self, tx: Dict, user_address: str, tx_type: str = 'normal') -> str:
        """
        Determine if transaction is a purchase or sale

        Args:
            tx (Dict): Transaction data
            user_address (str): User's address
            tx_type (str): Type of transaction (normal, token, internal)

        Returns:
            str: 'Purchase', 'Sale', 'Trade', or 'Transfer'
        """
        user_address = user_address.lower()

        if tx_type == 'token':
            # For token transactions, check if it's part of a DEX interaction
            if tx['from'].lower() == user_address:
                # Check if this was sent to a known DEX contract
                if tx.get('to', '').lower() in [k.lower() for k in self.dex_contracts.keys()]:
                    return 'Trade'
                else:
                    return 'Sale'
            elif tx['to'].lower() == user_address:
                # Check if this came from a known DEX contract
                if tx.get('from', '').lower() in [k.lower() for k in self.dex_contracts.keys()]:
                    return 'Trade'
                else:
                    return 'Purchase'
            else:
                return 'Transfer'
        else:
            # For normal ETH transactions
            if tx['to'] and tx['to'].lower() == user_address:
                return 'Purchase'
            elif tx['from'].lower() == user_address:
                # Check if it's a DEX interaction
                if tx['to'] and tx['to'].lower() in [k.lower() for k in self.dex_contracts.keys()]:
                    return 'Trade'
                else:
                    return 'Sale'
            else:
                return 'Transfer'

    def _format_transaction_data(self, tx: Dict, user_address: str, tx_type: str = 'normal') -> Dict:
        """
        Format transaction data into standardized structure

        Args:
            tx (Dict): Raw transaction data
            user_address (str): User's address
            tx_type (str): Type of transaction

        Returns:
            Dict: Formatted transaction data
        """
        # Basic transaction info
        timestamp = datetime.fromtimestamp(int(tx['timeStamp']), tz=timezone.utc)

        # Determine transaction type and amount
        transaction_type = self._determine_transaction_type(tx, user_address, tx_type)

        if tx_type == 'token':
            # Token transaction
            decimals = int(tx.get('tokenDecimal', 18))
            amount = float(tx['value']) / (10 ** decimals)
            coin = tx.get('tokenSymbol', 'UNKNOWN')
            coin_name = tx.get('tokenName', coin)
        else:
            # ETH transaction
            amount = float(tx['value']) / (10 ** 18)
            coin = 'ETH'
            coin_name = 'Ethereum'

        # Transaction status
        status = 'Success' if tx.get('txreceipt_status') == '1' or tx.get('isError') == '0' else 'Failed'

        # Gas information (for normal transactions)
        gas_used = 0
        gas_price = 0
        if 'gasUsed' in tx and 'gasPrice' in tx:
            gas_used = int(tx['gasUsed'])
            gas_price = int(tx['gasPrice'])

        # Additional context
        contract_address = tx.get('contractAddress', '')
        dex_name = None
        if tx.get('to'):
            dex_name = self.dex_contracts.get(tx['to'].lower())

        return {
            'transaction_type': transaction_type,
            'transaction_status': status,
            'transaction_amount': amount,
            'transaction_hash': tx['hash'],
            'transaction_date': timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'),
            'transaction_timestamp': timestamp,
            'transaction_coin': coin,
            'coin_name': coin_name,
            'contract_address': contract_address,
            'dex_name': dex_name,
            'gas_used': gas_used,
            'gas_price': gas_price,
            'gas_cost_eth': (gas_used * gas_price) / (10 ** 18),
            'from_address': tx.get('from', ''),
            'to_address': tx.get('to', ''),
            'block_number': int(tx.get('blockNumber', 0)),
            'confirmation_count': int(tx.get('confirmations', 0)),
            'raw_data': tx  # Keep original data for reference
        }

    def get_transactions(self, address: str, limit: int = 50, include_tokens: bool = True,
                         include_internal: bool = False, transaction_type: Optional[str] = None,
                         transaction_status: Optional[str] = None) -> List[Dict]:
        """
        Get comprehensive transaction analysis for an address with smart pagination

        This method will keep fetching transactions until it finds the requested number
        of filtered results or exhausts all available transactions.

        Args:
            address (str): Ethereum address to analyze
            limit (int): Target number of filtered transactions to return
            include_tokens (bool): Include ERC-20 token transactions
            include_internal (bool): Include internal transactions
            transaction_type (str, optional): Filter by type ('Sale', 'Purchase', 'Trade', 'Transfer', 'All' or None)
            transaction_status (str, optional): Filter by status ('Success', 'Failed', 'All' or None)

        Returns:
            List[Dict]: List of filtered transaction data
        """
        if not self._is_valid_address(address):
            raise ValueError("Invalid Ethereum address format")

        logger.info(f"Fetching transactions for address: {address}")
        logger.info(f"Target: {limit} transactions with type='{transaction_type}' status='{transaction_status}'")

        # If no filters applied, use original simple method
        if not transaction_type or transaction_type.lower() == 'all':
            if not transaction_status or transaction_status.lower() == 'all':
                return self._get_transactions_simple(address, limit, include_tokens, include_internal)

        # Use smart pagination when filters are applied
        return self._get_filtered_transactions_with_pagination(
            address, limit, include_tokens, include_internal, transaction_type, transaction_status
        )

    def _get_transactions_simple(self, address: str, limit: int, include_tokens: bool, include_internal: bool) -> List[
        Dict]:
        """Simple transaction fetching without filters (original method)"""
        all_transactions = []

        # Get normal ETH transactions
        logger.info("Fetching normal transactions...")
        normal_txs = self._get_normal_transactions(address, limit)
        for tx in normal_txs:
            formatted_tx = self._format_transaction_data(tx, address, 'normal')
            all_transactions.append(formatted_tx)

        # Get token transactions
        if include_tokens:
            logger.info("Fetching token transactions...")
            time.sleep(0.2)
            token_txs = self._get_token_transactions(address, limit)
            for tx in token_txs:
                formatted_tx = self._format_transaction_data(tx, address, 'token')
                all_transactions.append(formatted_tx)

        # Get internal transactions
        if include_internal:
            logger.info("Fetching internal transactions...")
            time.sleep(0.2)
            internal_txs = self._get_internal_transactions(address, limit)
            for tx in internal_txs:
                formatted_tx = self._format_transaction_data(tx, address, 'internal')
                all_transactions.append(formatted_tx)

        # Sort by timestamp
        all_transactions.sort(key=lambda x: x['transaction_timestamp'], reverse=True)
        logger.info(f"Retrieved {len(all_transactions)} total transactions")

        return all_transactions[:limit]

    def _get_filtered_transactions_with_pagination(self, address: str, target_count: int, include_tokens: bool,
                                                   include_internal: bool, transaction_type: str,
                                                   transaction_status: str) -> List[Dict]:
        """
        Fetch transactions with smart pagination until we get enough filtered results
        """
        all_matching_transactions = []
        page = 1
        batch_size = 100  # Fetch larger batches for efficiency
        max_pages = 200  # Safety limit to avoid infinite loops
        total_fetched = 0

        logger.info("Starting smart pagination to find filtered transactions...")

        while len(all_matching_transactions) < target_count and page <= max_pages:
            batch_transactions = []
            current_batch_size = 0

            # Get normal ETH transactions for this page
            logger.info(f"Fetching normal transactions - Page {page}")
            time.sleep(0.5)
            normal_txs = self._get_normal_transactions(address, batch_size, page)

            if not normal_txs:
                logger.info("No more normal transactions available")
                break

            for tx in normal_txs:
                formatted_tx = self._format_transaction_data(tx, address, 'normal')
                batch_transactions.append(formatted_tx)
            current_batch_size += len(normal_txs)

            # Get token transactions for this page
            if include_tokens:
                logger.info(f"Fetching token transactions - Page {page}")
                time.sleep(0.5)
                token_txs = self._get_token_transactions(address, batch_size, page)

                for tx in token_txs:
                    formatted_tx = self._format_transaction_data(tx, address, 'token')
                    batch_transactions.append(formatted_tx)
                current_batch_size += len(token_txs)

            # Get internal transactions for this page
            if include_internal:
                logger.info(f"Fetching internal transactions - Page {page}")
                time.sleep(0.5)
                internal_txs = self._get_internal_transactions(address, batch_size, page)

                for tx in internal_txs:
                    formatted_tx = self._format_transaction_data(tx, address, 'internal')
                    batch_transactions.append(formatted_tx)
                current_batch_size += len(internal_txs)

            total_fetched += current_batch_size

            # If no transactions in this batch, we've reached the end
            if current_batch_size == 0:
                logger.info("No more transactions available")
                break

            # Apply filters to this batch
            filtered_batch = self._apply_filters(batch_transactions, transaction_type, transaction_status)
            all_matching_transactions.extend(filtered_batch)

            logger.info(f"Page {page}: Fetched {current_batch_size} transactions, "
                        f"found {len(filtered_batch)} matching filters. "
                        f"Total matches so far: {len(all_matching_transactions)}")

            # If we have enough results, break
            if len(all_matching_transactions) >= target_count:
                logger.info(f"Target reached! Found {len(all_matching_transactions)} matching transactions")
                break

            page += 1
            time.sleep(0.2)  # Rate limiting

        # Sort all transactions by timestamp and limit to target count
        all_matching_transactions.sort(key=lambda x: x['transaction_timestamp'], reverse=True)
        result = all_matching_transactions[:target_count]

        logger.info(f"Pagination complete: Retrieved {total_fetched} total transactions, "
                    f"returning {len(result)} filtered results")

        return result

    def _apply_filters(self, transactions: List[Dict], transaction_type: str, transaction_status: str) -> List[Dict]:
        """Apply transaction type and status filters to a list of transactions"""
        filtered = transactions.copy()

        # Filter by transaction type
        if transaction_type and transaction_type.lower() != 'all':
            filtered = [
                tx for tx in filtered
                if tx['transaction_type'].lower() == transaction_type.lower()
            ]

        # Filter by transaction status
        if transaction_status and transaction_status.lower() != 'all':
            status_filter = 'Success' if transaction_status.lower() in ['success', 'successful'] else 'Failed'
            filtered = [
                tx for tx in filtered
                if tx['transaction_status'] == status_filter
            ]

        return filtered

    def _is_valid_address(self, address: str) -> bool:
        """
        Validate Ethereum address format

        Args:
            address (str): Address to validate

        Returns:
            bool: True if valid, False otherwise
        """
        if not address or not isinstance(address, str):
            return False
        if not address.startswith('0x'):
            return False
        if len(address) != 42:
            return False
        try:
            int(address, 16)
            return True
        except ValueError:
            return False

    def get_transaction_summary(self, address: str, limit: int = 100) -> Dict:
        """
        Get summary statistics for an address

        Args:
            address (str): Ethereum address
            limit (int): Number of transactions to analyze

        Returns:
            Dict: Summary statistics
        """
        transactions = self.get_transactions(address, limit, include_tokens=True)

        if not transactions:
            return {}

        # Calculate statistics
        total_transactions = len(transactions)
        successful_transactions = len([tx for tx in transactions if tx['transaction_status'] == 'Success'])
        failed_transactions = total_transactions - successful_transactions

        # Transaction types
        purchases = len([tx for tx in transactions if tx['transaction_type'] == 'Purchase'])
        sales = len([tx for tx in transactions if tx['transaction_type'] == 'Sale'])
        trades = len([tx for tx in transactions if tx['transaction_type'] == 'Trade'])

        # Tokens traded
        tokens_traded = set([tx['transaction_coin'] for tx in transactions])

        # DEX usage
        dex_usage = {}
        for tx in transactions:
            if tx['dex_name']:
                dex_usage[tx['dex_name']] = dex_usage.get(tx['dex_name'], 0) + 1

        # Gas usage
        total_gas_cost = sum([tx['gas_cost_eth'] for tx in transactions])

        return {
            'address': address,
            'total_transactions': total_transactions,
            'successful_transactions': successful_transactions,
            'failed_transactions': failed_transactions,
            'success_rate': (successful_transactions / total_transactions * 100) if total_transactions > 0 else 0,
            'purchases': purchases,
            'sales': sales,
            'trades': trades,
            'unique_tokens': len(tokens_traded),
            'tokens_list': list(tokens_traded),
            'dex_usage': dex_usage,
            'total_gas_cost_eth': total_gas_cost,
            'avg_gas_cost_per_tx': total_gas_cost / total_transactions if total_transactions > 0 else 0,
            'analysis_timestamp': datetime.now(timezone.utc).isoformat()
        }

    def filter_transactions(self, transactions: List[Dict],
                            transaction_type: Optional[str] = None,
                            coin: Optional[str] = None,
                            min_amount: Optional[float] = None,
                            status: Optional[str] = None) -> List[Dict]:
        """
        Filter transactions based on criteria

        Args:
            transactions (List[Dict]): List of transactions to filter
            transaction_type (str, optional): Filter by type (Purchase/Sale/Trade)
            coin (str, optional): Filter by specific coin
            min_amount (float, optional): Minimum transaction amount
            status (str, optional): Filter by status (Success/Failed)

        Returns:
            List[Dict]: Filtered transactions
        """
        filtered = transactions.copy()

        if transaction_type:
            filtered = [tx for tx in filtered if tx['transaction_type'] == transaction_type]

        if coin:
            filtered = [tx for tx in filtered if tx['transaction_coin'].upper() == coin.upper()]

        if min_amount is not None:
            filtered = [tx for tx in filtered if tx['transaction_amount'] >= min_amount]

        if status:
            filtered = [tx for tx in filtered if tx['transaction_status'] == status]

        return filtered


def save_transactions_to_csv(transactions: List[Dict], filename: str = None,
                             directory: str = None, append_mode: bool = False) -> Optional[str]:
    """
    Save transactions to CSV file for future use

    Args:
        transactions (List[Dict]): List of transaction data from get_transactions()
        filename (str, optional): Custom filename. If not provided, auto-generates with timestamp
        directory (str, optional): Directory to save file in. Defaults to current directory
        append_mode (bool): If True, append to existing file. If False, overwrite

    Returns:
        str: Full path to the saved CSV file, or None if error occurred

    Raises:
        ValueError: If transactions list is empty
        OSError: If directory doesn't exist or file cannot be written
    """

    # Validate inputs
    if not transactions:
        logger.warning("No transactions to save - transactions list is empty")
        raise ValueError("Transactions list cannot be empty")

    if not isinstance(transactions, list):
        logger.error("Transactions must be a list")
        raise TypeError("Transactions must be a list of dictionaries")

    # Generate filename if not provided
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"transactions_{timestamp}.csv"

    # Ensure .csv extension
    if not filename.lower().endswith('.csv'):
        filename += '.csv'

    # Handle directory
    if directory:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Created directory: {directory}")
            except OSError as e:
                logger.error(f"Cannot create directory {directory}: {e}")
                raise OSError(f"Cannot create directory {directory}: {e}")

        filepath = os.path.join(directory, filename)
    else:
        filepath = filename

    # Define CSV columns in exact order requested
    csv_columns = [
        'id',
        'transaction_type',
        'transaction_status',
        'transaction_amount',
        'transaction_coin',
        'transaction_hash',
        'transaction_date'
    ]

    # Determine write mode
    write_mode = 'a' if append_mode else 'w'
    write_header = not (append_mode and os.path.exists(filepath))

    try:
        # Get starting ID if appending
        start_id = 1
        if append_mode and os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    rows = list(reader)
                    if rows:
                        last_id = int(rows[-1].get('id', 0))
                        start_id = last_id + 1
            except (ValueError, KeyError, IndexError):
                logger.warning("Could not determine last ID from existing file, starting from 1")
                start_id = 1

        # Write transactions to CSV
        with open(filepath, write_mode, newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)

            # Write header if needed
            if write_header:
                writer.writeheader()

            # Write transaction data
            for i, tx in enumerate(transactions, start_id):
                # Validate transaction data
                if not isinstance(tx, dict):
                    logger.warning(f"Skipping invalid transaction at index {i - start_id}: not a dictionary")
                    continue

                # Extract and validate transaction data
                try:
                    amount = tx.get('transaction_amount', 0)
                    if isinstance(amount, str):
                        amount = float(amount) if amount else 0
                    elif not isinstance(amount, (int, float)):
                        amount = 0

                    csv_row = {
                        'id': i,
                        'transaction_type': str(tx.get('transaction_type', '')).strip(),
                        'transaction_status': str(tx.get('transaction_status', '')).strip(),
                        'transaction_amount': amount,
                        'transaction_coin': str(tx.get('transaction_coin', '')).strip(),
                        'transaction_hash': str(tx.get('transaction_hash', '')).strip(),
                        'transaction_date': str(tx.get('transaction_date', '')).strip()
                    }

                    writer.writerow(csv_row)

                except (ValueError, TypeError) as e:
                    logger.warning(f"Error processing transaction {i}: {e}")
                    continue

        # Get absolute path for return
        abs_filepath = os.path.abspath(filepath)

        # Verify file was created and get file size
        if os.path.exists(abs_filepath):
            file_size = os.path.getsize(abs_filepath)
            mode_text = "Appended to" if append_mode else "Created"
            logger.info(f"{mode_text} CSV file: {abs_filepath}")
            logger.info(f"File size: {file_size:,} bytes")
            logger.info(f"Transactions saved: {len(transactions)}")

            # Count total rows in file
            try:
                with open(abs_filepath, 'r', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    total_rows = sum(1 for _ in reader)
                    logger.info(f"Total rows in file: {total_rows}")
            except Exception:
                pass  # Not critical if we can't count rows

        return abs_filepath

    except PermissionError as e:
        logger.error(f"Permission denied writing to {filepath}: {e}")
        raise OSError(f"Permission denied: Cannot write to {filepath}")

    except OSError as e:
        logger.error(f"OS error writing CSV file {filepath}: {e}")
        raise OSError(f"Cannot write to {filepath}: {e}")

    except Exception as e:
        logger.error(f"Unexpected error saving transactions to CSV: {e}")
        raise RuntimeError(f"Unexpected error saving CSV: {e}")

# Example usage and testing
if __name__ == "__main__":
    analyzer = EtherscanAnalyzer()
    test_address = "0xefC662Fe5c73E58BdDfD97015a21726D6423b088"

    try:
        # Test smart pagination with filters
        logger.info("=== Getting 20 Successful Sales with Smart Pagination ===")
        successful_sales = analyzer.get_transactions(
            test_address,
            limit=500,
            transaction_type="All",
            transaction_status="Success"
        )

        save_transactions_to_csv(successful_sales, "data/sample_transactions.csv")


    except Exception as e:
        logger.error(f"Error: {e}")
