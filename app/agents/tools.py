import json
from typing import Optional, List, Dict, Any
from langchain.tools import tool, ToolRuntime
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.services.cache_service import CacheService
from app.services.image_search_service import ImageSearchService
from app.services.translation_service import TranslationService
from app.services.ocr_service import OCRService
from app.services.filtration_service import FiltrationService
from app.services.currency_conversion_service import CurrencyConversionService

class ToolContext(BaseModel):
    """Stores context for the current menu processing session"""
    image_bytes: Optional[bytes] = None
    image_hash: Optional[str] = None
    db: Optional[Session] = None
    ocr_service: Optional[OCRService] = None
    filtration_service: Optional[FiltrationService] = None
    image_search_service: Optional[ImageSearchService] = None
    translation_service: Optional[TranslationService] = None
    currency_conversion_service: Optional[CurrencyConversionService] = None
    cache_service: Optional[CacheService] = None
    ocr_result: Optional[Dict] = None
    filtered_items: Optional[List[Dict]] = None

    class Config:
        arbitrary_types_allowed = True

@tool
def init_cache_service(runtime: ToolRuntime, target_language, target_currency):
    """
    Initializes cache service
    This should always be called first before using any cache-related tools

    Args:
        runtime: ToolRuntime containing context for the current menu processing session
        target_language: the target ISO 639 language
        target_currency: the target ISO 4217 currency code, None if not specified
    """
    cache_service = runtime.context.cache_service
    cache_service.set_session_details(runtime.context.db, target_language, target_currency, runtime.context.image_hash)

@tool
def check_menu_cache(runtime: ToolRuntime[ToolContext]) -> str:
    """
    Check if this exact menu has been processed before using its image hash.
    This should always be called first after initializing the cache service to avoid redundant processing.

    Args:
        runtime: ToolRuntime containing context for the current cache session and image hash

    Returns:
        JSON string with status 'found' or 'not_found' and data if found
    """
    try:
        return json.dumps(runtime.context.cache_service.get_complete_cached_result())
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Cache check failed: {str(e)}"
        })

@tool
def check_translation_cache(runtime: ToolRuntime[ToolContext]) -> str:
    """
    Check which menu items already have cached translations.

    This tool checks both exact matches (from the same menu) and global matches
    (from any menu) to maximize cache reuse and minimize translation API calls.

    Args:
        runtime: ToolRuntime containing context for the current cache session and array of item names to check

    Returns:
        JSON string with cached translations and items needing translation:
        {
            "cached_translations": {
                "Pizza Margherita": "Margherita Pizza",
                "Tiramisu": "Tiramisu"
            },
            "items_needing_translation": ["Caesar Salad"],
            "cache_stats": {
                "exact_matches": 1,
                "global_matches": 1,
                "items_needing_translation": 1,
                "cache_hit_rate": 0.67
            }
        }

    Use this before calling translate_items to avoid redundant translations.
    """
    try:
        if not runtime.context.filtered_items:
            return json.dumps({
                "error": "No filtered items found. Please run identify_menu_items first."
            })

        item_names = [item["name"] for item in runtime.context.filtered_items]

        translation_map, items_needing_translation = runtime.context.cache_service.get_cached_translations(
            item_names=item_names
        )

        total_items = len(item_names)
        cached_count = len(translation_map)
        cache_hit_rate = cached_count / total_items if total_items > 0 else 0.0

        result = {
            "cached_translations": translation_map,
            "items_needing_translation": items_needing_translation,
            "cache_stats": {
                "cached_count": cached_count,
                "items_needing_translation": len(items_needing_translation),
                "cache_hit_rate": round(cache_hit_rate, 2),
                "total_items": total_items
            }
        }

        return json.dumps(result)

    except json.JSONDecodeError as e:
        return json.dumps({
            "error": f"Invalid JSON input: {str(e)}"
        })
    except Exception as e:
        return json.dumps({
            "error": f"Cache check failed: {str(e)}"
        })

@tool
def check_image_cache(runtime: ToolRuntime[ToolContext]) -> str:
    """
    Check which menu items already have cached images.

    This tool checks both exact matches (from the same menu) and global matches
    (from any menu) to maximize cache reuse and minimize image search API calls.

    Args:
        runtime: ToolRuntime containing context for the current cache session and array of item names to check

    Returns:
        JSON string with cached images and items needing images:
        {
            "cached_images": {
                "Pizza Margherita": "IMG_120398.jpg",
                "Tiramisu": "IMG_512908.png"
            },
            "items_needing_images": ["Caesar Salad"],
            "cache_stats": {
                "exact_matches": 1,
                "global_matches": 1,
                "items_needing_translation": 1,
                "cache_hit_rate": 0.67
            }
        }

    Use this before calling find_images to avoid redundant image searches.
    """
    try:
        if not runtime.context.filtered_items:
            return json.dumps({
                "error": "No filtered items found. Please run identify_menu_items first."
            })

        item_names = [item["name"] for item in runtime.context.filtered_items]

        images_map, items_needing_images = runtime.context.cache_service.get_cached_images(
            item_names=item_names
        )

        total_items = len(item_names)
        cached_count = len(images_map)
        cache_hit_rate = cached_count / total_items if total_items > 0 else 0.0

        result = {
            "cached_translations": images_map,
            "items_needing_translation": items_needing_images,
            "cache_stats": {
                "cached_count": cached_count,
                "items_needing_translation": len(items_needing_images),
                "cache_hit_rate": round(cache_hit_rate, 2),
                "total_items": total_items
            }
        }

        return json.dumps(result)

    except json.JSONDecodeError as e:
        return json.dumps({
            "error": f"Invalid JSON input: {str(e)}"
        })
    except Exception as e:
        return json.dumps({
            "error": f"Cache check failed: {str(e)}"
        })

@tool
def save_menu_to_database(
        runtime: ToolRuntime[ToolContext],
        filename: str,
        file_size: int
) -> str:
    """
    Saves menu image metadata and menu items to the database.

    This tool persists the OCR results and filtered menu items to the database.
    It handles both new menu entries and updates to existing menus based on image hash.
    If a menu with the same image hash exists, it reuses that menu and updates OCR data if needed.

    When to use:
    - At the end of the menu processing pipeline

    Args:
        runtime: ToolRuntime containing context for the current database session, ocr result, and filtered ocr blocks
        filename: Name of the menu image file
        file_size: Size of the menu image file in bytes

    Returns:
        JSON string with success status, menu_id, and items_saved count:
        {
            "success": true,
            "menu_id": 123,
            "items_saved": 15,
            "message": "Successfully saved 15 items to database"
        }

        On error:
        {
            "success": false,
            "error": "Error message"
        }
    """
    try:
        if not runtime.context.ocr_result:
            return json.dumps({
                "success": False,
                "error": "No OCR result found. Please run run_ocr first."
            })

        if not runtime.context.filtered_items:
            return json.dumps({
                "success": False,
                "error": "No filtered items found. Please run identify_menu_items first."
            })

        ocr_result = runtime.context.ocr_result
        filtered_items = runtime.context.filtered_items
        cache_service = runtime.context.cache_service

        menu_id = cache_service.save_to_database(
            filename=filename,
            file_size=file_size,
            ocr_result=ocr_result,
            filtered_items=filtered_items
        )

        return json.dumps({
            "success": True,
            "menu_id": menu_id,
            "items_saved": len(filtered_items),
            "message": f"Successfully saved {len(filtered_items)} items to database"
        })

    except json.JSONDecodeError as e:
        return json.dumps({
            "success": False,
            "error": f"Invalid JSON input: {str(e)}"
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })

@tool
def run_ocr(runtime: ToolRuntime[ToolContext]) -> str:
    """
    Extracts text from a menu image using Google Cloud Vision API.

    This tool processes the uploaded menu image and extracts all text blocks with their
    normalized bounding box coordinates, detects the menu's language, and provides
    confidence scores. It's the first step in the menu processing pipeline and provides
    raw OCR data that will be filtered by identify_menu_items.

    The Vision API detects individual text regions (words, phrases, prices) and returns
    their positions as normalized coordinates (0.0 to 1.0) relative to image dimensions.
    Language detection runs automatically across the entire document.

    Call this tool first when processing a new menu image, after checking the cache and before filtering items,
    translating, or saving to the database.

    Args:
        runtime: ToolRuntime containing context with image_bytes and ocr_service

    Returns:
        JSON string with extracted text blocks and language information:
        {
            "success": true,
            "ocr_blocks": [
                {
                    "text": "Margherita Pizza",
                    "bounding_box": {
                        "left": 0.05,
                        "top": 0.1,
                        "width": 0.3,
                        "height": 0.02
                    }
                },
                {
                    "text": "$12.99",
                    "bounding_box": {
                        "left": 0.8,
                        "top": 0.1,
                        "width": 0.1,
                        "height": 0.02
                    }
                }
            ],
            "language": "en",
            "language_confidence": 0.95,
            "total_blocks": 2,
            "message": "Successfully extracted 2 text blocks from image"
        }

        Note: Bounding box coordinates are normalized (0.0 to 1.0) relative to image dimensions.

        On error:
        {
            "success": false,
            "error": "Error message"
        }
    """
    try:
        ocr_info = runtime.context.ocr_service.extract_text(runtime.context.image_bytes)
        runtime.context.ocr_result = ocr_info

        return json.dumps({
            "success": True,
            "language": ocr_info["language"],
            "language_confidence": ocr_info["language_confidence"],
            "total_blocks": ocr_info["total_blocks"],
            "message": f"Successfully extracted {len(ocr_info['blocks'])} text blocks from image"
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })

@tool
def identify_menu_items(runtime: ToolRuntime[ToolContext], extract_prices: bool) -> str:
    """
    Use AI to identify actual menu items from OCR text blocks.

    This tool filters OCR text to extract only menu items (dish names and prices),
    removing headers, descriptions, allergen warnings, contact info, etc.

    The tool uses sophisticated filtering logic with detailed rules for:
    - Identifying actual menu items vs noise
    - Extracting prices separately from names
    - Removing descriptive text like "served with", "comes with"
    - Preserving bounding box information

    When to use:
    - After extracting text from image with run_ocr
    - When you have raw text blocks that need filtering
    - Before translating items (cleaner input = better translations)

    Args:
        runtime: ToolRuntime containing context for the filtration service
            Expected format:
            [
                {
                    "text": "Margherita Pizza",
                    "bounding_box": {"top": 0.1, "left": 0.05, "width": 0.3, "height": 0.02}
                },
                {
                    "text": "$12.99",
                    "bounding_box": {"top": 0.1, "left": 0.8, "width": 0.1, "height": 0.02}
                }
            ]

        extract_prices: Whether to extract and associate prices with items (default: True)

    Returns:
        JSON string with filtered menu items:
        {
            "success": true,
            "items": [
                {
                    "name": "Margherita Pizza",
                    "original_price": 12.99,
                    "currency": "USD",
                    "bounding_box": {"top": 0.1, "left": 0.05, "width": 0.3, "height": 0.02}
                }
            ],
            "item_count": 1,
            "message": "Identified 1 menu items"
        }

        On error:
        {
            "error": "Error message"
        }
    """
    try:
        if not runtime.context.ocr_result:
            return json.dumps({
                "error": "No OCR result found. Please run run_ocr first."
            })

        filtration_service = runtime.context.filtration_service
        ocr_blocks = runtime.context.ocr_result["blocks"]

        result = filtration_service.filter_ocr(
            ocr_blocks=ocr_blocks if ocr_blocks else [],
            extract_prices=extract_prices
        )

        if result is not None:
            runtime.context.filtered_items = result

        return json.dumps({
            "success": True,
            "items": result,
            "item_count": len(result),
            "message": f"Identified {len(result)} menu items"
        })
    except Exception as e:
        return json.dumps({"error": str(e)})

@tool
def convert_currency(
        runtime: ToolRuntime[ToolContext],
        original_prices_json: str,
        to_currency: str
) -> str:
    """
    Converts multiple prices from one currency to another using live exchange rates.

    This tool uses ExchangeRate-API to fetch current conversion rates and convert
    a batch of prices.

    Call this after identifying menu items with prices to convert them to the user's
    preferred currency ONLY IF to_currency is not None. The tool handles fetching the latest exchange rate and
    converting all prices in one operation.

    Args:
        runtime: ToolRuntime containing context for currency conversion
        original_prices_json: JSON array of prices to convert.
            Example: '[10.50, 25.00, 15.75, 8.99]'
        to_currency: Target currency code (e.g., "USD", "EUR", "GBP")

    Returns:
        JSON string with converted prices and conversion rate:
        {
            "success": true,
            "conversions": {
                "10.50": 11.55,
                "25.00": 27.50,
                "15.75": 17.33,
                "8.99": 9.89
            },
            "from_currency": "USD",
            "to_currency": "EUR",
            "exchange_rate": 1.10,
            "total_converted": 4
        }

        On error:
        {
            "success": false,
            "error": "Error message"
        }
    """
    try:
        original_prices: List[float] = json.loads(original_prices_json)

        from_currency = runtime.context.ocr_result["language"]

        if not original_prices:
            return json.dumps({
                "success": True,
                "conversions": {},
                "from_currency": from_currency,
                "to_currency": to_currency,
                "total_converted": 0
            })

        currency_conversion_service = runtime.context.currency_conversion_service
        currency_conversion_service.set_currency_conversion_rate(from_currency, to_currency)
        conversions = currency_conversion_service.convert_batch(original_prices)

        if runtime.context.filtered_items:
            for item in runtime.context.filtered_items:
                original_price = item.get("original_price")
                if original_price is not None and original_price in conversions:
                    item["converted_price"] = conversions[original_price]
                    item["currency"] = to_currency

        return json.dumps({
            "success": True,
            "conversions": conversions,
            "from_currency": from_currency,
            "to_currency": to_currency,
            "total_converted": len(conversions)
        })

    except json.JSONDecodeError as e:
        return json.dumps({
            "success": False,
            "error": f"Invalid JSON input: {str(e)}"
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })

@tool
async def translate_items(
        runtime: ToolRuntime[ToolContext],
        texts_json: str,
        target_language: str,
) -> str:
    """
    Translates multiple menu item names to the target language using DeepL API.

    This tool translates a batch of menu item names in a single call.
    It automatically detects the source language if not specified and returns a
    mapping of original text to translated text.

    Call this after identifying menu items and before saving to the database to
    provide translations for all menu items. The tool handles batch translation
    and maintains the mapping between original and translated text.

    Args:
        runtime: ToolRuntime containing context for translation
        texts_json: JSON array of texts to translate.
            Example: '["Pizza Margherita", "Pasta Carbonara", "Tiramisu"]'
        target_language: Target language code (ISO 639 format).
            Examples: "en", "es", "fr", "de", "it", "ja", "zh"

    Returns:
        JSON string with translation mappings:
        {
            "success": true,
            "translations": {
                "Pizza Margherita": "Margherita Pizza",
                "Pasta Carbonara": "Carbonara Pasta",
                "Tiramisu": "Tiramisu"
            },
            "source_language": "it",
            "target_language": "en",
            "total_translated": 3
        }

        On error (falls back to original text):
        {
            "success": false,
            "translations": {
                "Pizza Margherita": "Pizza Margherita",
                "Pasta Carbonara": "Pasta Carbonara"
            },
            "error": "Error message",
            "total_translated": 0
        }
    """
    try:
        texts: List[str] = json.loads(texts_json)
        source_language = runtime.context.ocr_result["language"] if runtime.context.ocr_result[
                                                                      "language_confidence"] > .5 else 'auto'

        if not texts:
            return json.dumps({
                "success": True,
                "translations": {},
                "source_language": source_language,
                "target_language": target_language,
                "total_translated": 0
            })

        translation_service = runtime.context.translation_service
        translation_map = await translation_service.translate_batch(
            texts=texts,
            target_language=target_language,
            source_language=source_language
        )

        if runtime.context.filtered_items:
            for item in runtime.context.filtered_items:
                item_name = item.get("name")
                if item_name in translation_map:
                    item["translated_name"] = translation_map[item_name]

        return json.dumps({
            "success": True,
            "translations": translation_map,
            "source_language": source_language,
            "target_language": target_language,
            "total_translated": len(translation_map)
        })

    except json.JSONDecodeError as e:
        return json.dumps({
            "success": False,
            "error": f"Invalid JSON input: {str(e)}",
            "translations": {}
        })
    except Exception as e:
        try:
            texts = json.loads(texts_json)
            fallback_map = {text: text for text in texts}
        except:
            fallback_map = {}

        return json.dumps({
            "success": False,
            "error": str(e),
            "translations": fallback_map,
            "total_translated": 0
        })

@tool
async def find_dish_images(
        runtime: ToolRuntime[ToolContext],
        dish_names_json: str
) -> str:
    """
    Searches for images for multiple menu items using Google Custom Search API.

    This tool finds food images for menu items by searching Google Images.
    It processes multiple dish names in batch and returns image URLs that can be
    associated with menu items.

    Call this after translating menu items (if needed) and before saving to the database.
    Use check_image_cache first to avoid redundant searches for items that already
    have cached images.

    Args:
        runtime: ToolRuntime containing context for image search
        dish_names_json: JSON array of dish names to search for.
            Example: '["Margherita Pizza", "Caesar Salad", "Tiramisu"]'

    Returns:
        JSON string with image URLs mapped to dish names:
        {
            "success": true,
            "images": {
                "Margherita Pizza": "https://example.com/pizza.jpg",
                "Caesar Salad": "https://example.com/salad.jpg",
                "Tiramisu": null
            },
            "found_count": 2,
            "total_searched": 3,
            "not_found": ["Tiramisu"]
        }

        On error:
        {
            "success": false,
            "error": "Error message",
            "images": {}
        }
    """
    try:
        dish_names: List[str] = json.loads(dish_names_json)

        if not dish_names:
            return json.dumps({
                "success": True,
                "images": {},
                "found_count": 0,
                "total_searched": 0,
            })

        image_search_service = runtime.context.image_search_service
        results = await image_search_service.search_multiple_dishes(dish_names)
        found_count = len(dish_names)

        if runtime.context.filtered_items:
            for item in runtime.context.filtered_items:
                item_name = item.get("name")
                if item_name in results:
                    item["image_url"] = results[item_name]

        return json.dumps({
            "success": True,
            "images": results,
            "found_count": found_count,
            "total_searched": len(dish_names),
        })

    except json.JSONDecodeError as e:
        return json.dumps({
            "success": False,
            "error": f"Invalid JSON input: {str(e)}",
            "images": {}
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "images": {}
        })

ALL_TOOLS = [
    init_cache_service,
    check_menu_cache,
    check_translation_cache,
    check_image_cache,
    save_menu_to_database,
    run_ocr,
    identify_menu_items,
    convert_currency,
    translate_items,
    find_dish_images
]
