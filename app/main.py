"""
Main FastAPI Application
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import Optional
import logging
from contextlib import asynccontextmanager
import hashlib

from app.database import init_db, get_db
from app.services.cache_service import CacheService
from app.services.image_search_service import ImageSearchService
from app.services.translation_service import TranslationService
from app.services.ocr_service import OCRService
from app.services.filtration_service import FiltrationService
from app.services.currency_conversion_service import CurrencyConversionService
from app.services.menu_processing_service import MenuProcessingService
from app.config import settings
from app.agents.menu_agent import MenuAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    logger.info("Starting Menu Passport Backend")
    logger.info(f"Environment: {settings.environment}")

    init_db()
    logger.info("Database initialized")

    yield

    logger.info("Shutting down Menu Passport Backend")
    logger.info("Shutdown complete")

app = FastAPI(
    title="Menu Passport API",
    description="Agentic AI-powered menu translation API that processes items from foreign restaurant menu images with OCR, enriching with translations, currency conversion, and images",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ocr_service = OCRService()
image_search_service = ImageSearchService(settings.google_custom_search_api_key, settings.google_custom_search_engine_id)
translation_service = TranslationService(settings.deepl_api_key)
currency_service = CurrencyConversionService()
filtration_service = FiltrationService()
cache_service = CacheService()

@app.api_route("/", methods=["GET", "HEAD"], tags=["Info"])
async def root():
    """API information"""
    return {
        "service": "Menu Passport API",
        "services": {
            "ocr": "Google Cloud Vision",
            "agent": "GPT 5.1",
            "translation": "DeepL",
            "currency": "ExchangeRate-API",
            "image_search": "Google Custom Search"
        },
        "endpoints": {
            "traditional processing": "POST /process",
            "agent processing": "POST /process-agent"
        }
    }

@app.post("/process", tags=["Menu Processing (Traditional)"])
async def process_menu(
    file: UploadFile = File(...),
    target_language: str = Query(..., description="Language to convert to (ISO 639: e.g., en, es, fr)"),
    target_currency: Optional[str] = Query(None, description="OPTIONAL: Currency to convert to (ISO 4217: e.g., USD, EUR)"),
    db: Session = Depends(get_db),
):
    """
    Traditional menu processing route

    **Process:**
    1. Check database for cached OCR results
    2. If not cached, run OCR with Vision API
    3. AI agent filters menu items
    4. (Optional) Extract and associate prices with items
    5. (Optional) Convert prices to target currency
    6. Search for images of each menu item
    7. Translate menu item names
    8. Save all results to database

    **Parameters:**
    - file: Menu image (JPEG/PNG)
    - target_language: Target language for translation
    - target_currency: Target currency for conversion

    **Returns:**
    - menu_items: List of structured menu items with:
      - name: Original menu item name
      - translated_name: Translated name
      - bounding_box: Location on image
      - image_url: Google image search result
      - original_price: Original price (if target_currency specified)
      - converted_price: Converted price (if target_currency specified)
    """
    try:
        allowed_types = ["image/jpeg", "image/png", "image/jpg"]
        if file.content_type not in allowed_types:
            raise HTTPException(400, "Invalid file type. Allowed: JPEG, PNG")

        logger.info(f"Processing menu: {file.filename}")
        logger.info(f"Translation: {target_language} (required)")
        logger.info(f"Target Currency: {target_currency or 'none'}")

        image_bytes = await file.read()
        file_size = len(image_bytes)

        max_size = settings.max_image_size_mb * 1024 * 1024
        if file_size > max_size:
            raise HTTPException(400, f"File too large. Max: {settings.max_image_size_mb}MB")

        image_hash = hashlib.sha256(image_bytes).hexdigest()
        logger.info(f"Image hash: {image_hash[:8]}...")
        logger.info(f"File size: {file_size / 1024:.1f} KB")

        menu_processing_service = MenuProcessingService(
            ocr_service=ocr_service,
            currency_service=currency_service,
            image_search_service=image_search_service,
            translation_service=translation_service,
            filtration_service=filtration_service,
            cache_service=cache_service,
        )

        result = await menu_processing_service.process_menu(
            image_bytes=image_bytes,
            image_hash=image_hash,
            db=db,
            target_language=target_language,
            target_currency=target_currency
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(500, str(e))

@app.post("/process-agent", tags=["Menu Processing (Agent)"])
async def process_menu_agent(
    file: UploadFile = File(...),
    target_language: str = Query(..., description="Language to convert to (ISO 639: e.g., en, es, fr)"),
    target_currency: Optional[str] = Query(None, description="OPTIONAL: Currency to convert to (ISO 4217: e.g., USD, EUR)"),
    db: Session = Depends(get_db)
):
    """
        Menu processing route utilizing agentic AI

        **Parameters:**
        - file: Menu image (JPEG/PNG)
        - target_language: Target language for translation
        - target_currency: Target currency for conversion

        **Returns:**
        - menu_items: List of structured menu items with:
          - name: Original menu item name
          - translated_name: Translated name
          - bounding_box: Location on image
          - image_url: Google image search result
          - original_price: Original price (if target_currency specified)
          - converted_price: Converted price (if target_currency specified)
        """
    try:
        allowed_types = ["image/jpeg", "image/png", "image/jpg"]
        if file.content_type not in allowed_types:
            raise HTTPException(400, "Invalid file type. Allowed: JPEG, PNG")

        logger.info(f"Processing menu: {file.filename}")
        logger.info(f"Translation: {target_language} (required)")
        logger.info(f"Target Currency: {target_currency or 'none'}")

        image_bytes = await file.read()
        file_size = len(image_bytes)

        max_size = settings.max_image_size_mb * 1024 * 1024
        if file_size > max_size:
            raise HTTPException(400, f"File too large. Max: {settings.max_image_size_mb}MB")

        image_hash = hashlib.sha256(image_bytes).hexdigest()
        logger.info(f"Image hash: {image_hash[:8]}...")
        logger.info(f"File size: {file_size / 1024:.1f} KB")

        menu_agent = MenuAgent()
        menu_agent.set_tool_context(
            image_bytes,
            image_hash,
            db,
            ocr_service,
            filtration_service,
            image_search_service,
            translation_service,
            currency_service,
            cache_service
        )
        result = await menu_agent.process_menu(target_language, target_currency)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(500, str(e))