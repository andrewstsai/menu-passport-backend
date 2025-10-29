"""
OCR Filtration Service
"""
import json
from typing import List, Dict
from app.agents.prompts import FILTERING_SYSTEM_PROMPT, FILTER_MENU_ITEMS_PROMPT

from openai import OpenAI
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class FiltrationService:
    """OpenAI API service for filtering OCR blocks"""
    def __init__(self):
        """Initializes FiltrationService class"""
        self.client = OpenAI(api_key=settings.openai_api_key)

    def filter_ocr(self, ocr_blocks: List[Dict], extract_prices: bool) -> List[Dict]:
        """
           Calls OpenAI API to filter OCR blocks

           Args:
               ocr_blocks: List of ocr blocks
               extract_prices: If true, extract prices from text blocks

           Returns:
               List representation of ocr blocks containing only menu items
           """
        try:
            sorted_blocks = sorted(
                ocr_blocks,
                key=lambda b: (
                    b.get('bounding_box', {}).get('top', 0),
                    b.get('bounding_box', {}).get('left', 0)
                )
            )

            input_data = {
                "total_blocks": len(sorted_blocks),
                "blocks": [
                    {
                        "position": i + 1,
                        "text": block.get('text', ''),
                        "top": round(block.get('bounding_box', {}).get('top', 0), 4),
                        "left": round(block.get('bounding_box', {}).get('left', 0), 4),
                        "bounding_box": block.get('bounding_box', {})
                    }
                    for i, block in enumerate(sorted_blocks)
                ]
            }

            prompt = FILTER_MENU_ITEMS_PROMPT.format(
                input_data=json.dumps(input_data, indent=2),
                extract_prices=extract_prices
            )

            response = self.client.responses.create(
                model="gpt-5-mini",
                instructions=FILTERING_SYSTEM_PROMPT,
                input=prompt,
            )

            result_text = None
            for output_item in response.output:
                if output_item.type == 'message':
                    for content_item in output_item.content:
                        if content_item.type == 'output_text':
                            result_text = content_item.text
                            break
                    if result_text:
                        break

            parsed = json.loads(result_text)
            items = parsed.get('items', [])
            return items

        except Exception as e:
            logger.error(f"Filtration failed: {str(e)}", exc_info=True)
            return []