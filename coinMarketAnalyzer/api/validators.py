"""
Shared input-validation helpers.

All public functions raise ``ValueError`` with a user-facing message
when the input is invalid.  Pydantic's ``@model_validator`` catches
these and converts them into 422 responses automatically.
"""

import re
from datetime import date, datetime, timedelta

# Base58 characters (Solana addresses)
SOLANA_ADDRESS_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")

# Perps tickers: 2-10 uppercase letters, optional "-USD" suffix
PERPS_TICKER_RE = re.compile(r"^[A-Z]{2,10}(-USD)?$")

# Perps user addresses: 0x-prefixed 40-hex-char string (42 total)
PERPS_USER_ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")

# ISO 8601 date (YYYY-MM-DD only)
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# ISO 8601 datetime (YYYY-MM-DDTHH:MM:SS with optional fractional seconds and Z/offset)
ISO_DATETIME_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?$"
)

DATE_MIN = date(2020, 1, 1)
DATE_MAX_FUTURE_DAYS = 1


def validate_token_address(value: str, token_type: str) -> str:
    """Validate and return a canonical token address / ticker."""
    token_type = token_type.lower().strip() if token_type else token_type
    if token_type == "perps":
        upper = value.upper()
        if not PERPS_TICKER_RE.match(upper):
            raise ValueError(
                f"Invalid perps ticker '{value}'. "
                "Expected 2-10 uppercase letters, optionally followed by '-USD'."
            )
        return upper
    # meme: Solana base58 address
    if not SOLANA_ADDRESS_RE.match(value):
        raise ValueError(
            f"Invalid Solana token address '{value}'. "
            "Expected 32-44 base58 characters."
        )
    return value


def validate_user_address(value: str, token_type: str) -> str:
    """Validate a user wallet address based on token type."""
    token_type = token_type.lower().strip() if token_type else token_type
    if token_type == "perps":
        if not PERPS_USER_ADDRESS_RE.match(value):
            raise ValueError(
                f"Invalid perps user address '{value}'. "
                "Expected 0x-prefixed 40-character hex string (42 chars total)."
            )
        return value
    # meme: Solana base58 address
    if not SOLANA_ADDRESS_RE.match(value):
        raise ValueError(
            f"Invalid Solana wallet address '{value}'. "
            "Expected 32-44 base58 characters."
        )
    return value


def validate_date_field(value: str) -> str:
    """Validate an ISO 8601 date string (YYYY-MM-DD) within allowed range."""
    if not ISO_DATE_RE.match(value):
        raise ValueError(
            f"Invalid date '{value}'. Expected ISO 8601 format: YYYY-MM-DD."
        )
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(f"Invalid date '{value}'. Could not parse as a real date.")

    max_date = date.today() + timedelta(days=DATE_MAX_FUTURE_DAYS)
    if parsed < DATE_MIN:
        raise ValueError(
            f"Date '{value}' is before the minimum allowed date ({DATE_MIN.isoformat()})."
        )
    if parsed > max_date:
        raise ValueError(
            f"Date '{value}' is too far in the future (max: {max_date.isoformat()})."
        )
    return value


def validate_iso_datetime(value: str) -> str:
    """Validate an ISO 8601 datetime string."""
    if not ISO_DATETIME_RE.match(value):
        raise ValueError(
            f"Invalid datetime '{value}'. "
            "Expected ISO 8601 format: YYYY-MM-DDTHH:MM:SS[.fff][Z|+HH:MM]."
        )
    return value
