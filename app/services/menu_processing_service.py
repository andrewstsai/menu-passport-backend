"""
Menu Orchestration Agent
Coordinates services for complete menu extraction
"""

import logging
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
from sqlalchemy.orm import Session
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.agents.prompts import SYSTEM_PROMPT, FILTER_MENU_ITEMS_PROMPT
from app.models.menu import MenuImage, MenuItem
from app.config import settings

logger = logging.getLogger(__name__)


class MenuProcessingService:
    """
    Orchestration Agent for menu extraction and enrichment
    """

    def __init__(
        self,
        ocr_service,
        translation_service,
        image_search_service,
        currency_service_class
    ):
        """
        Initializes Orchestration Agent with services

        Args:
            ocr_service: OCR service instance
            translation_service: Translation service instance
            image_search_service: Image search service instance
            currency_service_class: CurrencyConversionService class
        """
        self.ocr_service = ocr_service
        self.translation_service = translation_service
        self.image_search_service = image_search_service
        self.currency_service_class = currency_service_class
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-flash-lite-latest",
            google_api_key=settings.gemini_api_key,
            temperature=0.1,
            max_output_tokens=8192,
            convert_system_message_to_human=True
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

            complete_cached = self._get_complete_cached_result(
                image_hash=image_hash,
                target_language=target_language,
                target_currency=target_currency,
                db=db
            )

            if complete_cached:
                logger.info("Returning complete cached result")
                return complete_cached

            ocr_result = self._get_ocr_result(image_bytes, image_hash, db)
            detected_language = ocr_result.get('language', 'en')

            logger.info(f"OCR complete: {ocr_result['total_blocks']} blocks, language: {detected_language}")

            logger.info("Filtering menu items with AI agent")
            filtered_items = await self._filter_menu_items(
                ocr_blocks=ocr_result['blocks'],
                extract_prices=target_currency is not None,
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

            cached_translations, items_needing_translation = self._get_cached_translations(
                item_names=item_names,
                target_language=target_language,
                image_hash=image_hash,
                db=db
            )

            cached_images, items_needing_images = self._get_cached_images(
                item_names=item_names,
                image_hash=image_hash,
                db=db
            )

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
                    currency_service = self.currency_service_class(
                        from_currency=detected_language,
                        to_currency=target_currency
                    )
                    price_map = currency_service.convert_batch(prices)
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
            menu_id = self._save_to_database(
                db=db,
                filename=f"menu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                file_size=len(image_bytes),
                hash_value=image_hash,
                ocr_result=ocr_result,
                filtered_items=filtered_items,
                target_language=target_language
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

    async def _filter_menu_items(
        self,
        ocr_blocks: List[Dict],
        extract_prices: bool
    ) -> List[Dict]:
        """
        Uses an LLM to filter through OCR blocks and extract menu items

        Args:
            ocr_blocks: Raw OCR text blocks
            extract_prices: Whether to extract and associate prices

        Returns:
            List of filtered and structured menu items
        """
        sorted_blocks = sorted(
            ocr_blocks,
            key=lambda b: (b['bounding_box']['top'], b['bounding_box']['left'])
        )

        input_data = {
            "total_blocks": len(sorted_blocks),
            "blocks": [
                {
                    "position": i + 1,
                    "text": block['text'],
                    "top": round(block['bounding_box']['top'], 4),
                    "left": round(block['bounding_box']['left'], 4),
                    "bounding_box": block['bounding_box']
                }
                for i, block in enumerate(sorted_blocks)
            ]
        }

        prompt = FILTER_MENU_ITEMS_PROMPT.format(
            input_data=json.dumps(input_data, indent=2),
            extract_prices=extract_prices
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]

        response = self.llm.invoke(messages)
        result_text = response.content.strip()

        if result_text.startswith("```"):
            lines = result_text.split("\n")
            result_text = "\n".join(lines[1:-1])
            if result_text.startswith("json"):
                result_text = result_text[4:].strip()

        try:
            parsed = json.loads(result_text)
            items = parsed.get('items', [])
            logger.info(f"Filtered {len(ocr_blocks)} blocks → {len(items)} menu items")
            return items

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response: {e}")
            logger.error(f"Response was: {result_text}")
            raise Exception(f"AI returned invalid JSON: {e}")

    @staticmethod
    def _get_complete_cached_result(
            image_hash: str,
            target_language: str,
            target_currency: Optional[str],
            db: Session
    ) -> Optional[Dict]:
        """
        Gets complete cached result including OCR, translations, and images

        Args:
            image_hash: SHA-256 hash of the image
            target_language: Target language code (ISO 639)
            target_currency: Target currency code (ISO 4217) - optional
            db: Database session

        Returns:
            Complete result dict if fully cached, None if any part missing
        """
        logger.info(
            f"Checking complete cache (hash: {image_hash}, lang: {target_language}, currency: {target_currency or 'none'})")

        cached_menu = (
            db.query(MenuImage)
            .filter(
                MenuImage.image_hash == image_hash,
                MenuImage.ocr_completed == True
            )
            .first()
        )

        if not cached_menu:
            logger.info("Complete cache miss")
            return None

        cached_items = (
            db.query(MenuItem)
            .filter(
                MenuItem.menu_image_id == cached_menu.id,
                MenuItem.target_language == target_language,
                MenuItem.translated_text.isnot(None)
            )
            .order_by(MenuItem.position)
            .all()
        )

        if not cached_items:
            logger.info(f"Complete cache miss'")
            return None

        if target_currency:
            items_with_prices = [item for item in cached_items if item.original_price is not None]

            if items_with_prices:
                currency_matches = all(
                    item.currency == target_currency
                    for item in items_with_prices
                )

                if not currency_matches:
                    logger.info(f"Complete cache miss - currency mismatch")
                    return None
            else:
                return None

        items = []
        for item in cached_items:
            item_dict = {
                'name': item.text,
                'translated_name': item.translated_text,
                'image_url': item.image_url,
                'bounding_box': item.bounding_box
            }

            if target_currency:
                item_dict['original_price'] = item.original_price
                item_dict['converted_price'] = item.converted_price
                item_dict['currency'] = item.currency

            items.append(item_dict)

        result = {
            "status": "success",
            "cached": True,
            "data": {
                "menu_items": items,
                "total_items": len(items),
                "metadata": {
                    "original_language": cached_menu.language,
                    "translated_to": target_language,
                    "target_currency": target_currency,
                    "processed_at": cached_menu.created_at.isoformat(),
                    "menu_id": cached_menu.id
                }
            }
        }

        logger.info(f"Complete cache HIT - returning {len(items)} items from menu ID {cached_menu.id}")
        return result

    @staticmethod
    def _get_cached_translations(
            item_names: List[str],
            target_language: str,
            image_hash: str,
            db: Session
    ) -> Tuple[Dict[str, str], List[str]]:
        """Gets cached translations with language filtering"""

        logger.info(f"Smart translation cache for {len(item_names)} items (lang: {target_language})")

        exact_match_menu = (
            db.query(MenuImage)
            .filter(MenuImage.image_hash == image_hash)
            .first()
        )

        exact_matches = {}
        if exact_match_menu:
            exact_items = (
                db.query(MenuItem.text, MenuItem.translated_text)
                .filter(
                    MenuItem.menu_image_id == exact_match_menu.id,
                    MenuItem.text.in_(item_names),
                    MenuItem.target_language == target_language,
                    MenuItem.translated_text.isnot(None)
                )
                .all()
            )
            exact_matches = {item.text: item.translated_text for item in exact_items}

            if exact_matches:
                logger.info(f"Exact match: {len(exact_matches)} translations")

        items_still_needed = [name for name in item_names if name not in exact_matches]

        global_matches = {}
        if items_still_needed:
            global_items = (
                db.query(MenuItem.text, MenuItem.translated_text)
                .filter(
                    MenuItem.text.in_(items_still_needed),
                    MenuItem.target_language == target_language,
                    MenuItem.translated_text.isnot(None)
                )
                .order_by(MenuItem.created_at.desc())
                .distinct(MenuItem.text)
                .all()
            )
            global_matches = {item.text: item.translated_text for item in global_items}

            if global_matches:
                logger.info(f"Global match: {len(global_matches)} translations")

        translation_map = {**global_matches, **exact_matches}

        items_needing_translation = [
            name for name in item_names
            if name not in translation_map
        ]

        logger.info(
            f"Cache: {len(exact_matches)} exact, "
            f"{len(global_matches)} global, "
            f"{len(items_needing_translation)} fresh"
        )

        return translation_map, items_needing_translation

    @staticmethod
    def _get_cached_images(
            item_names: List[str],
            image_hash: str,
            db: Session
    ) -> Tuple[Dict[str, Optional[str]], List[str]]:
        """
        Gets cached image URLs

        Args:
            item_names: List of menu item names
            image_hash: Current menu's hash
            db: Database session

        Returns:
            Tuple of (image_map, items_needing_search)
        """
        logger.info(f"Image cache check for {len(item_names)} items")

        exact_match_menu = (
            db.query(MenuImage)
            .filter(MenuImage.image_hash == image_hash)
            .first()
        )

        exact_matches = {}
        if exact_match_menu:
            exact_items = (
                db.query(MenuItem.text, MenuItem.image_url)
                .filter(
                    MenuItem.menu_image_id == exact_match_menu.id,
                    MenuItem.text.in_(item_names),
                    MenuItem.image_url.isnot(None)
                )
                .all()
            )
            exact_matches = {item.text: item.image_url for item in exact_items}

            if exact_matches:
                logger.info(f"Exact match: {len(exact_matches)} images from same menu")

        items_still_needed = [name for name in item_names if name not in exact_matches]

        global_matches = {}
        if items_still_needed:
            global_items = (
                db.query(MenuItem.text, MenuItem.image_url)
                .filter(
                    MenuItem.text.in_(items_still_needed),
                    MenuItem.image_url.isnot(None)
                )
                .order_by(MenuItem.created_at.desc())
                .all()
            )
            seen = set()
            for item in global_items:
                if item.text not in seen:
                    global_matches[item.text] = item.image_url
                    seen.add(item.text)

            if global_matches:
                logger.info(f"Global match: {len(global_matches)} images from other menus")

        image_map = {**global_matches, **exact_matches}

        items_needing_search = [
            name for name in item_names
            if name not in image_map
        ]

        logger.info(
            f"Image cache: {len(exact_matches)} exact, "
            f"{len(global_matches)} global, "
            f"{len(items_needing_search)} need search"
        )

        return image_map, items_needing_search

    @staticmethod
    def _save_to_database(
            db: Session,
            filename: str,
            file_size: int,
            hash_value: str,
            ocr_result: Dict,
            filtered_items: List[Dict],
            target_language: str
    ) -> int:
        """Saves menu and items"""
        try:
            existing_menu = (
                db.query(MenuImage)
                .filter(MenuImage.image_hash == hash_value)
                .first()
            )

            if existing_menu:
                logger.info(f"Reusing existing menu ID {existing_menu.id}")
                menu = existing_menu

                if not menu.ocr_completed:
                    menu.ocr_completed = True
                    menu.ocr_blocks = ocr_result['blocks']
                    menu.total_blocks = ocr_result['total_blocks']
                    menu.language = ocr_result.get('language')
                    menu.language_confidence = ocr_result.get('language_confidence')
                    menu.status = "completed"
                    db.commit()
                    logger.info(f"Updated OCR data for menu ID {menu.id}")
            else:
                menu = MenuImage(
                    filename=filename,
                    file_size=file_size,
                    image_hash=hash_value,
                    ocr_completed=True,
                    ocr_blocks=ocr_result['blocks'],
                    total_blocks=ocr_result['total_blocks'],
                    language=ocr_result.get('language'),
                    language_confidence=ocr_result.get('language_confidence'),
                    status="completed"
                )

                db.add(menu)
                db.commit()
                db.refresh(menu)

                currencies_in_items = {
                    item.get('currency')
                    for item in filtered_items
                    if item.get('currency')
                }
                target_currency = currencies_in_items.pop() if currencies_in_items else None

                existing_items = (
                    db.query(MenuItem)
                    .filter(
                        MenuItem.menu_image_id == menu.id,
                        MenuItem.target_language == target_language,
                        MenuItem.currency == target_currency
                    )
                    .all()
                )

                if existing_items:
                    for item in existing_items:
                        db.delete(item)
                    db.commit()

            for i, item_data in enumerate(filtered_items):
                item = MenuItem(
                    menu_image_id=menu.id,
                    text=item_data['name'],
                    translated_text=item_data.get('translated_name'),
                    target_language=target_language,
                    original_price=item_data.get('original_price'),
                    converted_price=item_data.get('converted_price'),
                    currency=item_data.get('currency'),
                    image_url=item_data.get('image_url'),
                    bounding_box=item_data.get('bounding_box'),
                    position=i + 1
                )
                db.add(item)

            db.commit()
            logger.info(f"💾 Saved {len(filtered_items)} items")
            return menu.id

        except Exception as e:
            db.rollback()
            logger.error(f"❌ Save failed: {e}")
            raise