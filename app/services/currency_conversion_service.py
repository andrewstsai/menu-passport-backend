"""
Currency Conversion Service
"""

import httpx
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class CurrencyConversionService:
    """
    Currency conversion service using ExchangeRate-API

    API: https://www.exchangerate-api.com/
    """

    # Mapping of supported ISO 639 language codes to ISO 4217 currency codes
    LANGUAGE_TO_CURRENCY = {
        'ar': 'SAR',
        'bg': 'BGN',
        'cs': 'CZK',
        'da': 'DKK',
        'de': 'EUR',
        'el': 'EUR',
        'en': 'USD',
        'es': 'EUR',
        'et': 'EUR',
        'fi': 'EUR',
        'fr': 'EUR',
        'hu': 'HUF',
        'id': 'IDR',
        'it': 'EUR',
        'ja': 'JPY',
        'ko': 'KRW',
        'lt': 'EUR',
        'lv': 'EUR',
        'nb': 'NOK',
        'nl': 'EUR',
        'pl': 'PLN',
        'pt': 'BRL',
        'ro': 'RON',
        'ru': 'RUB',
        'sk': 'EUR',
        'sl': 'EUR',
        'sv': 'SEK',
        'tr': 'TRY',
        'uk': 'UAH',
        'zh': 'CNY',
    }

    API_BASE_URL = "https://api.exchangerate-api.com/v4/latest"

    def __init__(self):
        """Initializes currency conversion service"""
        self.from_currency = None
        self.to_currency = None
        self.conversion_rate: Optional[float] = None

    def set_currency_conversion_rate(self, from_currency: str, to_currency: str):
        """
        Sets currency conversion rate

        Args:
            from_currency: Source currency (ISO 639 language code or ISO 4217 currency code)
            to_currency: Target currency (ISO 4217 currency code)
        """
        self.from_currency = self.language_to_currency(from_currency)
        self.to_currency = to_currency

        logger.info(f"Currency service initialized")
        logger.info(f"From: {self.from_currency}")
        logger.info(f"To: {self.to_currency or 'None'}")

        if self.to_currency and self.from_currency != self.to_currency:
            self.fetch_conversion_rate()
        elif self.from_currency == self.to_currency:
            self.conversion_rate = 1.0

    def language_to_currency(self, code: str) -> str:
        """
        Converts ISO 639 language code to ISO 4217 currency code

        Args:
            code: Language code (e.g., 'en', 'es', 'ja') or currency code (e.g., 'USD')

        Returns:
            ISO 4217 currency code
        """
        if len(code) == 3 and code.isupper():
            return code

        currency = self.LANGUAGE_TO_CURRENCY.get(code.lower())

        if currency:
            logger.info(f"Converted language '{code}' → currency '{currency}'")
            return currency

        logger.warning(f"Unknown language code '{code}', defaulting to USD")
        return 'USD'

    def fetch_conversion_rate(self):
        """
        Fetches live conversion rate from ExchangeRate-API

        Raises:
            Exception if API call fails
        """
        try:
            url = f"{self.API_BASE_URL}/{self.from_currency}"
            logger.info(f"Fetching conversion rate: {self.from_currency} → {self.to_currency}")

            with httpx.Client(timeout=5.0) as client:
                response = client.get(url)
                response.raise_for_status()
                data = response.json()

            if self.to_currency not in data['rates']:
                raise ValueError(f"Currency '{self.to_currency}' not found")

            self.conversion_rate = data['rates'][self.to_currency]
            logger.info(
                f"Conversion rate fetched: 1 {self.from_currency} = {self.conversion_rate} {self.to_currency}")

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching conversion rate: {e.response.status_code}")
            raise
        except httpx.TimeoutException:
            logger.error(f"Timeout fetching conversion rate")
            raise
        except Exception as e:
            logger.error(f"Failed to fetch conversion rate: {e}")
            raise

    def convert(self, price: float) -> Optional[float]:
        """
        Converts price from source currency to target currency

        Args:
            price: Original price in source currency

        Returns:
            Converted price in target currency, or None if no conversion configured
        """
        if not self.to_currency:
            logger.debug("No target currency")
            return None

        if self.from_currency == self.to_currency:
            logger.debug("Same currency")
            return price

        if self.conversion_rate is None:
            logger.warning("No conversion rate available")
            return None

        converted_price = price * self.conversion_rate
        logger.debug(f"Converted: {price} {self.from_currency} → {converted_price:.2f} {self.to_currency}")

        return round(converted_price, 2)

    def convert_batch(self, prices: list[float]) -> Dict[float, Optional[float]]:
        """
        Converts multiple prices at once

        Args:
            prices: List of prices in source currency

        Returns:
            Dict mapping original price to converted price
            Example: {10.50: 11.55, 25.00: 27.50}
        """
        if not self.to_currency:
            return {price: None for price in prices}

        return {price: self.convert(price) for price in prices}