"""
DeepL Translation Service
"""

import httpx
import logging
from typing import List, Dict
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class TranslationService:
    """DeepL API translation service"""

    DEEPL_API_URL = "https://api-free.deepl.com/v2/translate"

    SUPPORTED_LANGUAGES = {
        'ar': 'AR',
        'bg': 'BG',
        'cs': 'CS',
        'da': 'DA',
        'de': 'DE',
        'el': 'EL',
        'en': 'EN',
        'es': 'ES',
        'et': 'ET',
        'fi': 'FI',
        'fr': 'FR',
        'hu': 'HU',
        'id': 'ID',
        'it': 'IT',
        'ja': 'JA',
        'ko': 'KO',
        'lt': 'LT',
        'lv': 'LV',
        'nb': 'NB',
        'nl': 'NL',
        'pl': 'PL',
        'pt': 'PT-BR',
        'ro': 'RO',
        'ru': 'RU',
        'sk': 'SK',
        'sl': 'SL',
        'sv': 'SV',
        'tr': 'TR',
        'uk': 'UK',
        'zh': 'ZH',
    }

    def __init__(self, api_key: str):
        """
        Initializes DeepL translator

        Args:
            api_key: DeepL API key
        """
        self.api_key = api_key
        self.base_url = self.DEEPL_API_URL
        logger.info(f"Translation service initialized")

    def _normalize_language_code(self, lang: str) -> str:
        """
        Normalizes language code to DeepL format

        Args:
            lang: Language code (e.g., 'en', 'es', 'zh')

        Returns:
            Normalized language code

        Raises:
            ValueError: If language not supported
        """
        lang_lower = lang.lower().strip()

        if lang_lower in self.SUPPORTED_LANGUAGES:
            return self.SUPPORTED_LANGUAGES[lang_lower]

        lang_upper = lang.upper()
        if lang_upper in self.SUPPORTED_LANGUAGES.values():
            return lang_upper

        raise ValueError(
            f"Language '{lang}' not supported. "
            f"Supported: {list(self.SUPPORTED_LANGUAGES.keys())}"
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def translate_batch(
            self,
            texts: List[str],
            target_language: str,
            source_language: str = "auto"
    ) -> Dict[str, str]:
        """
        Translates multiple texts to target language

        Args:
            texts: List of texts to translate
            target_language: Target language code (ISO 639)
            source_language: Source language (default: auto-detect)

        Returns:
            Dictionary mapping original text to translated text
        """
        if not texts:
            return {}

        logger.info(f"Translating {len(texts)} texts to {target_language}")

        try:
            import httpx

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.base_url,
                    data={
                        "auth_key": self.api_key,
                        "text": texts,
                        "target_lang": target_language.upper(),
                        "source_lang": source_language.upper() if source_language != "auto" else None
                    }
                )

                response.raise_for_status()
                result = response.json()
                translation_map = {}
                for i, translation in enumerate(result['translations']):
                    original_text = texts[i]
                    translated_text = translation['text']
                    translation_map[original_text] = translated_text

                logger.info(f"Translated {len(translation_map)} texts")
                return translation_map

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return {text: text for text in texts}

    async def get_usage(self) -> Dict[str, any]:
        """
        Gets current API usage statistics

        Returns:
            Dict with usage information
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/usage",
                    data={'auth_key': self.api_key},
                    timeout=10.0
                )

                response.raise_for_status()
                usage = response.json()

                character_count = usage.get('character_count', 0)
                character_limit = usage.get('character_limit', 0)

                percentage = (
                    round((character_count / character_limit) * 100, 2)
                    if character_limit > 0
                    else 0.0
                )

                return {
                    "character_count": character_count,
                    "character_limit": character_limit,
                    "percentage_used": percentage,
                    "remaining": character_limit - character_count,
                    "limit_reached": character_count >= character_limit
                }

        except Exception as e:
            logger.error(f"Failed to get usage stats: {e}")
            return {
                "error": str(e),
                "character_count": 0,
                "character_limit": 0
            }

    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes"""
        return list(self.SUPPORTED_LANGUAGES.keys())

    def is_language_supported(self, lang: str) -> bool:
        """Check if language is supported"""
        return lang.lower() in self.SUPPORTED_LANGUAGES