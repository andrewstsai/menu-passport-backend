import json
from typing import Optional, Dict, Any
from langchain.agents import create_agent
import logging
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from sqlalchemy.orm import Session

from app.services.cache_service import CacheService
from app.services.image_search_service import ImageSearchService
from app.services.translation_service import TranslationService
from app.services.ocr_service import OCRService
from app.services.filtration_service import FiltrationService
from app.services.currency_conversion_service import CurrencyConversionService
from app.config import settings
from app.agents.tools import ToolContext, ALL_TOOLS
from app.agents.prompts import MENU_AGENT_PROMPT

logger = logging.getLogger(__name__)

class MenuAgent:
    """
    AI agent that orchestrates menu extraction by intelligently
    coordinating multiple tools with smart caching.

    Uses SQLAlchemy models for database operations.
    """

    def __init__(self):
        """
        Initialize the menu orchestration agent.
        """
        self.model = ChatOpenAI(
            model="gpt-5-mini",
            api_key=settings.openai_api_key,
            temperature=0.1,
            max_tokens=16384
        )

        self.context = ToolContext()

        logger.info("MenuAgent initialized successfully")

    async def process_menu(
        self,
        target_language: str = "en",
        target_currency: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a menu using agent orchestration.

        The agent will:
        1. Check if menu already processed
        2. Check dish cache for items
        3. Extract text if needed
        4. Translate only uncached items
        5. Search images for only uncached items
        6. Save results to database

        Args:
            target_language: Target language code (ISO 639)
            target_currency: Target currency code (ISO 4217)

        Returns:
            Dict with success status, data, and metadata
        """
        logger.info(f"Processing menu: hash={self.context.image_hash[:8]}..., lang={target_language}")

        try:
            agent = create_agent(
                model=self.model,
                tools=ALL_TOOLS,
                system_prompt=MENU_AGENT_PROMPT,
                context_schema=ToolContext,
            )

            task = f"""Extract and process this restaurant menu image:

                **Context:**
                - Target language: {target_language}
                - Target currency (Optional): {target_currency}

                **Your Task:**
                1. Check if entire menu is cached
                2. If not cached:
                   - Extract text from image
                   - Identify menu items
                   - Check which dishes already have cached translations
                   - Check which dishes already have cached images
                   - Translate ONLY uncached dishes and ONLY fetch uncached images
                   - Optionally convert currencies to {target_currency} if {target_currency} is specified (not None)
                3. Save complete menu

                Provide final structured menu data.
                IMPORTANT: After calling save_menu_to_database, immediately return the final JSON. Do not call any more tools."""

            response = await agent.ainvoke(
                Command(update={"messages": task}),
                config=RunnableConfig(recursion_limit=35),
                context=self.context
            )

            messages = response.get("messages", [])
            final_message = None

            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content:
                    final_message = msg
                    break

            if not final_message:
                raise ValueError("No final response from agent")

            result = json.loads(final_message.content.strip())

            if hasattr(final_message, 'usage_metadata'):
                total_tokens = final_message.usage_metadata.get("total_tokens", 0)
                if "data" in result and "metadata" in result["data"]:
                    result["data"]["metadata"]["total_tokens_used"] = total_tokens

            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")

            return {
                "status": "error",
                "error": f"Invalid JSON response: {str(e)}",
            }

        except Exception as e:
            logger.error(f"Agent processing failed: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "message": "Agent orchestration failed"
            }

        finally:
            if self.context.db:
                self.context.db.close()

    def set_tool_context(
            self,
            image_bytes: bytes,
            image_hash: str,
            db: Session,
            ocr_service: OCRService,
            filtration_service: FiltrationService,
            image_search_service: ImageSearchService,
            translation_service: TranslationService,
            currency_conversion_service: CurrencyConversionService,
            cache_service: CacheService,

    ):
        """Set context for all tools"""
        self.context.image_bytes = image_bytes
        self.context.image_hash = image_hash
        self.context.db = db
        self.context.ocr_service = ocr_service
        self.context.filtration_service = filtration_service
        self.context.image_search_service = image_search_service
        self.context.translation_service = translation_service
        self.context.currency_conversion_service = currency_conversion_service
        self.context.cache_service = cache_service