"""
Google Custom Image Search Service
"""

import httpx
import logging
from typing import Optional, List, Dict
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class ImageSearchService:
    """Google Custom Search API service for finding dish images"""

    BASE_URL = "https://www.googleapis.com/customsearch/v1"

    def __init__(self, api_key: str, search_engine_id: str):
        """
        Initialize Google Custom Search API client

        Args:
            api_key: Google Cloud API key
            search_engine_id: Custom Search Engine ID (cx)
        """
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        logger.info("Google Image search service initialized")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def search_dish_image(
        self,
        dish_name: str,
    ) -> Optional[str]:
        """
        Searches for dish image using Google Custom Search

        Args:
            dish_name: Name of the dish

        Returns:
            URL of best matching image, or None if not found
        """
        try:
            logger.info(f"Searching Google Images for: '{dish_name}'")

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    self.BASE_URL,
                    params={
                        "key": self.api_key,
                        "cx": self.search_engine_id,
                        "q": dish_name,
                        "searchType": "image",
                        "num": 1,
                        "imgSize": "large",
                        "imgType": "photo",
                        "safe": "active",
                        "fileType": "jpg,png",
                    },
                    timeout=10.0
                )

                response.raise_for_status()
                data = response.json()
                items = data.get('items', [])

                if not items:
                    logger.warning(f"No images found for: {dish_name}")
                    return None

                first_result = items[0]
                image_url = first_result.get('link')

                if 'image' in first_result:
                    image_info = first_result['image']
                    logger.info(
                        f"Found image: {image_url} "
                        f"(size: {image_info.get('width')}x{image_info.get('height')})"
                    )
                else:
                    logger.info(f"Found image: {image_url}")

                return image_url

        except httpx.HTTPStatusError as e:
            error_code = e.response.status_code

            if error_code == 429:
                logger.error("Google API quota exceeded")
            elif error_code == 403:
                logger.error("Google API key invalid or Custom Search API not enabled")
            elif error_code == 400:
                error_detail = e.response.json().get('error', {}).get('message', 'Unknown')
                logger.error(f"Bad request: {error_detail}")
            else:
                logger.error(f"Google API error: {error_code}")
            return None

        except httpx.TimeoutException:
            logger.error("Request timeout")
            return None

        except KeyError as e:
            logger.error(f"Unexpected response format: missing {e}")
            return None

        except Exception as e:
            logger.error(f"Image search failed: {e}")
            return None

    async def search_multiple_dishes(
        self,
        dish_names: List[str],
    ) -> Dict[str, Optional[str]]:
        """
        Searches for images for multiple dishes

        Args:
            dish_names: List of dish names

        Returns:
            Dict mapping dish names to image URLs
        """
        results = {}

        for dish_name in dish_names:
            image_url = await self.search_dish_image(dish_name)
            results[dish_name] = image_url

        found_count = sum(1 for url in results.values() if url is not None)
        logger.info(f"Found images for {found_count}/{len(dish_names)} dishes")

        return results