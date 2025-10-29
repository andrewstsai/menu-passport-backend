"""
Menu Orchestration Agent
Coordinates services for complete menu extraction
"""

import logging
from typing import Dict, Optional
from datetime import datetime
from sqlalchemy.orm import Session
import asyncio
from langchain_openai import ChatOpenAI
from app.models.menu import MenuImage, MenuItem
from app.config import settings
from app.services.cache_service import CacheService
from app.services.image_search_service import ImageSearchService
from app.services.translation_service import TranslationService
from app.services.ocr_service import OCRService
from app.services.filtration_service import FiltrationService
from app.services.currency_conversion_service import CurrencyConversionService

logger = logging.getLogger(__name__)


class MenuProcessingService:
    """
    Orchestration Agent for menu extraction and enrichment
    """

    def __init__(
        self,
        ocr_service: OCRService,
        translation_service: TranslationService,
        image_search_service: ImageSearchService,
        currency_service: CurrencyConversionService,
        filtration_service: FiltrationService,
        cache_service: CacheService,
    ):
        """
        Initializes Orchestration Agent with services

        Args:
            ocr_service: OCR service instance
            translation_service: Translation service instance
            image_search_service: Image search service instance
            currency_service: Currency conversion service instance
            filtration_service: Filtration service instance
            cache_service: Cache service instance
        """
        self.ocr_service = ocr_service
        self.translation_service = translation_service
        self.image_search_service = image_search_service
        self.currency_service = currency_service
        self.filtration_service = filtration_service
        self.cache_service = cache_service
        self.llm = ChatOpenAI(
            model="gpt-5-nano",
            api_key=settings.openai_api_key,
            temperature=0.2,
            max_tokens=8192
        )
        logger.info("Orchestration Agent initialized")

    async def process_menu(
            self,
            image_bytes: bytes,
            image_hash: str,
            db: Session,
            target_language: str,
            target_currency: Optional[str] = None
    ) -> Dict:
        """
        Complete menu processing pipeline
        """
        try:
            logger.info("Starting menu processing pipeline")
            logger.info(
                f"Hash: {image_hash}, Language: {target_language}, Currency: {target_currency or 'none'}")

            self.cache_service.set_session_details(db, target_language, target_currency, image_hash)
            complete_cached = self.cache_service.get_complete_cached_result()

            if complete_cached:
                logger.info("Returning complete cached result")
                return complete_cached

            ocr_result = self._get_ocr_result(image_bytes, image_hash, db)
            detected_language = ocr_result.get('language', 'en')

            logger.info(f"OCR complete: {ocr_result['total_blocks']} blocks, language: {detected_language}")

            logger.info("Filtering menu items with AI agent")
            filtered_items = self.filtration_service.filter_ocr(
                ocr_blocks=ocr_result['blocks'],
                extract_prices=target_currency is not None
            )
            logger.info(f"Extracted {len(filtered_items)} menu items")

            if not filtered_items:
                logger.warning("No menu items found")
                return {
                    "status": "success",
                    "cached": False,
                    "data": {
                        "menu_items": [],
                        "total_items": 0,
                        "metadata": {
                            "original_language": detected_language,
                            "translated_to": target_language,
                            "processed_at": datetime.now().isoformat(),
                            "menu_id": None
                        }
                    }
                }

            item_names = [item['name'] for item in filtered_items]

            cached_translations, items_needing_translation = self.cache_service.get_cached_translations(item_names=item_names)
            cached_images, items_needing_images = self.cache_service.get_cached_images(item_names=item_names)

            async def get_translations():
                """Async task for translations"""
                if items_needing_translation:
                    logger.info(f"Translating {len(items_needing_translation)} items to {target_language}")
                    translations = await self.translation_service.translate_batch(
                        items_needing_translation,
                        target_language
                    )
                    logger.info(f"Translation complete")
                    return translations
                else:
                    logger.info(f"All {len(item_names)} translations cached")
                    return {}

            async def get_images():
                """Async task for image search"""
                if items_needing_images:
                    logger.info(f"Searching images for {len(items_needing_images)} items")
                    images = await self.image_search_service.search_multiple_dishes(
                        items_needing_images
                    )
                    logger.info(f"Image search complete")
                    return images
                else:
                    logger.info(f"All {len(item_names)} images cached")
                    return {}

            fresh_translations, fresh_images = await asyncio.gather(
                get_translations(),
                get_images()
            )

            translation_map = {**cached_translations, **fresh_translations}
            image_map = {**cached_images, **fresh_images}

            price_map = None
            if target_currency:
                logger.info(f"Converting prices to {target_currency}")
                prices = [
                    item['original_price']
                    for item in filtered_items
                    if item.get('original_price')
                ]

                if prices:
                    self.currency_service.set_currency_conversion_rate(
                        from_currency=detected_language,
                        to_currency=target_currency
                    )
                    price_map = self.currency_service.convert_batch(prices)
                    logger.info(f"Converted {len(prices)} prices")
                else:
                    logger.info("No prices to convert")

            for item in filtered_items:
                item['translated_name'] = translation_map.get(item['name'], item['name'])
                item['image_url'] = image_map.get(item['name'])
                if price_map and item.get('original_price'):
                    converted = price_map.get(item['original_price'])
                    if converted is not None:
                        item['converted_price'] = converted
                        item['currency'] = target_currency

            logger.info("Saving to database")
            menu_id = self.cache_service.save_to_database(
                filename=f"menu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                file_size=len(image_bytes),
                ocr_result=ocr_result,
                filtered_items=filtered_items,
            )

            result = {
                "status": "success",
                "cached": False,
                "data": {
                    "menu_items": filtered_items,
                    "total_items": len(filtered_items),
                    "metadata": {
                        "original_language": detected_language,
                        "translated_to": target_language,
                        "target_currency": target_currency,
                        "processed_at": datetime.now().isoformat(),
                        "menu_id": menu_id
                    }
                }
            }

            logger.info(f"Processing complete for menu ID {menu_id}")
            return result

        except Exception as e:
            logger.error(f"Menu processing failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _get_ocr_result(self, image_bytes: bytes, image_hash: str, db: Session) -> Dict:
        """
        Gets OCR result from database if cached or run fresh OCR

        Args:
            image_bytes: Image data as bytes
            image_hash: SHA-256 hash for cache lookup
            db: Database session

        Returns:
            OCR result with blocks and metadata
        """
        file_size = len(image_bytes)
        logger.info(f"Checking database for cached OCR (size: {file_size} bytes)...")

        cached_menu = (
            db.query(MenuImage)
            .filter(
                MenuImage.image_hash == image_hash,
                MenuImage.ocr_completed == True,
                MenuImage.ocr_blocks.isnot(None)
            )
            .first()
        )

        if cached_menu:
            logger.info(f"Cache hit - using menu ID {cached_menu.id}")
            return {
                'blocks': cached_menu.ocr_blocks,
                'total_blocks': cached_menu.total_blocks,
                'language': cached_menu.language,
                'language_confidence': cached_menu.language_confidence,
                'cached': True
            }

        logger.info(f"Cache miss - running OCR service")
        ocr_result = self.ocr_service.extract_text(image_bytes)
        ocr_result['cached'] = False

        return ocr_result