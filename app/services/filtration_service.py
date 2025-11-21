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

def _merge_bounding_boxes(boxes: List[Dict]) -> Dict | None:
    """Merge multiple bounding boxes into one encompassing box."""
    if not boxes:
        return None
    if len(boxes) == 1:
        return boxes[0]

    min_left = min(box['left'] for box in boxes)
    min_top = min(box['top'] for box in boxes)
    max_right = max(box['left'] + box['width'] for box in boxes)
    max_bottom = max(box['top'] + box['height'] for box in boxes)

    return {
        'left': round(min_left, 4),
        'top': round(min_top, 4),
        'width': round(max_right - min_left, 4),
        'height': round(max_bottom - min_top, 4)
    }


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
                model="gpt-5.1",
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

            for item in items:
                if 'source_positions' in item and len(item['source_positions']) > 0:
                    source_boxes = []
                    for pos in item['source_positions']:
                        if 0 < pos <= len(sorted_blocks):
                            block = sorted_blocks[pos - 1]
                            if 'bounding_box' in block:
                                source_boxes.append(block['bounding_box'])

                    if source_boxes:
                        item['bounding_box'] = _merge_bounding_boxes(source_boxes)
                    else:
                        item['bounding_box'] = None
                else:
                    item['bounding_box'] = None

                item.pop('source_positions', None)
            return items

        except Exception as e:
            logger.error(f"Filtration failed: {str(e)}", exc_info=True)
            return []

