from typing import Dict, List, Optional, Tuple
import logging
from sqlalchemy.orm import Session
from app.models.menu import MenuImage, MenuItem
logger = logging.getLogger(__name__)

class CacheService:

    def __init__(self):
        self.db = None
        self.target_language = None
        self.target_currency = None
        self.image_hash = None
        self.target_currency = None

    def set_session_details(self, db: Session, target_language: str, target_currency: str, image_hash: str):
        """
        Sets session details for current menu

        Args:
            db: Current database session
            target_language: Target language
            target_currency: Target currency
            image_hash: Image hash
        """
        self.db = db
        self.target_language = target_language
        self.target_currency = target_currency
        self.image_hash = image_hash

    def get_complete_cached_result(self) -> Optional[Dict]:
        """
        Gets complete cached result including OCR, translations, and images

        Returns:
            Complete result dict if fully cached, None if any part missing
        """
        logger.info(
            f"Checking complete cache (hash: {self.image_hash}, lang: {self.target_language}, currency: {self.target_currency or 'none'})")

        cached_menu = (
            self.db.query(MenuImage)
            .filter(
                MenuImage.image_hash == self.image_hash,
                MenuImage.ocr_completed == True
            )
            .first()
        )

        if not cached_menu:
            logger.info("Complete cache miss")
            return None

        cached_items = (
            self.db.query(MenuItem)
            .filter(
                MenuItem.menu_image_id == cached_menu.id,
                MenuItem.target_language == self.target_language,
                MenuItem.translated_text.isnot(None)
            )
            .order_by(MenuItem.position)
            .all()
        )

        if not cached_items:
            logger.info(f"Complete cache miss'")
            return None

        if self.target_currency:
            items_with_prices = [item for item in cached_items if item.original_price is not None]

            if items_with_prices:
                currency_matches = all(
                    item.currency == self.target_currency
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

            if self.target_currency:
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
                    "translated_to": self.target_language,
                    "target_currency": self.target_currency,
                    "processed_at": cached_menu.created_at.isoformat(),
                    "menu_id": cached_menu.id
                }
            }
        }

        logger.info(f"Complete cache HIT - returning {len(items)} items from menu ID {cached_menu.id}")
        return result

    def get_cached_translations(
            self,
            item_names: List[str]
    ) -> Tuple[Dict[str, str], List[str]]:
        """
        Gets cached translations with language filtering

        Args:
           item_names: List of menu item names

        Returns:
            Tuple of (translation_map, items_needing_translation)
        """

        logger.info(f"Translation cache for {len(item_names)} items (lang: {self.target_language})")

        exact_match_menu = (
            self.db.query(MenuImage)
            .filter(MenuImage.image_hash == self.image_hash)
            .first()
        )

        exact_matches = {}
        if exact_match_menu:
            exact_items = (
                self.db.query(MenuItem.text, MenuItem.translated_text)
                .filter(
                    MenuItem.menu_image_id == exact_match_menu.id,
                    MenuItem.text.in_(item_names),
                    MenuItem.target_language == self.target_language,
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
                self.db.query(MenuItem.text, MenuItem.translated_text)
                .filter(
                    MenuItem.text.in_(items_still_needed),
                    MenuItem.target_language == self.target_language,
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

    def get_cached_images(
            self,
            item_names: List[str]
    ) -> Tuple[Dict[str, Optional[str]], List[str]]:
        """
        Gets cached image URLs

        Args:
            item_names: List of menu item names

        Returns:
            Tuple of (image_map, items_needing_search)
        """
        logger.info(f"Image cache check for {len(item_names)} items")

        exact_match_menu = (
            self.db.query(MenuImage)
            .filter(MenuImage.image_hash == self.image_hash)
            .first()
        )

        exact_matches = {}
        if exact_match_menu:
            exact_items = (
                self.db.query(MenuItem.text, MenuItem.image_url)
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
                self.db.query(MenuItem.text, MenuItem.image_url)
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

    def save_to_database(
            self,
            filename: str,
            file_size: int,
            ocr_result: Dict,
            filtered_items: List[Dict],
    ) -> int:
        """Saves menu and items"""
        try:
            existing_menu = (
                self.db.query(MenuImage)
                .filter(MenuImage.image_hash == self.image_hash)
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
                    self.db.commit()
                    logger.info(f"Updated OCR data for menu ID {menu.id}")

                currencies_in_items = {
                    item.get('currency')
                    for item in filtered_items
                    if item.get('currency')
                }
                target_currency = currencies_in_items.pop() if currencies_in_items else None

                existing_items = (
                    self.db.query(MenuItem)
                    .filter(
                        MenuItem.menu_image_id == menu.id,
                        MenuItem.target_language == self.target_language,
                        MenuItem.currency == target_currency
                    )
                    .all()
                )

                if existing_items:
                    logger.info(f"Deleting {len(existing_items)} old items")
                    for item in existing_items:
                        self.db.delete(item)
                    self.db.commit()

            else:
                menu = MenuImage(
                    filename=filename,
                    file_size=file_size,
                    image_hash=self.image_hash,
                    ocr_completed=True,
                    ocr_blocks=ocr_result['blocks'],
                    total_blocks=ocr_result['total_blocks'],
                    language=ocr_result.get('language'),
                    language_confidence=ocr_result.get('language_confidence'),
                    status="completed"
                )

                self.db.add(menu)
                self.db.commit()
                self.db.refresh(menu)

            logger.info(f"Saving {len(filtered_items)} items to database")
            for i, item_data in enumerate(filtered_items):
                logger.info(f"Item {i}: name={item_data.get('name')}, translated={item_data.get('translated_name')}")

                item = MenuItem(
                    menu_image_id=menu.id,
                    text=item_data['name'],
                    translated_text=item_data.get('translated_name'),
                    target_language=self.target_language,
                    original_price=item_data.get('original_price'),
                    converted_price=item_data.get('converted_price'),
                    currency=item_data.get('currency'),
                    image_url=item_data.get('image_url'),
                    bounding_box=item_data.get('bounding_box'),
                    position=i + 1
                )
                self.db.add(item)

            self.db.commit()
            logger.info(f"✅ Saved {len(filtered_items)} items to database")
            return menu.id

        except Exception as e:
            self.db.rollback()
            logger.error(f"Save failed: {e}", exc_info=True)
            raise