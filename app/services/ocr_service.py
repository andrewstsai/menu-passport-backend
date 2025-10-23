"""
Google Cloud Vision OCR Service
"""

from google.cloud import vision_v1 as vision
import logging
from typing import Dict
from tenacity import retry, stop_after_attempt, wait_exponential
from app.config import settings
import io
from PIL import Image as pillowImage

logger = logging.getLogger(__name__)


class OCRService:
    """Google Cloud Vision OCR service for menu text extraction"""

    def __init__(
        self
    ):
        """
        Initialize Vision API client
        """
        self.client = vision.ImageAnnotatorClient.from_service_account_json(settings.credentials_path)
        logger.info(f"OCR service initialized")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def extract_text(
        self,
        image_bytes: bytes
    ) -> Dict:
        """
        Extracts text from menu image using Vision API

        Args:
            image_bytes: Image data as bytes

        Returns:
            Dict with extracted text blocks, locations, and metadata
        """
        try:
            text_blocks = []
            detected_language = None
            language_confidence = None

            pil_image = pillowImage.open(io.BytesIO(image_bytes))
            img_width, img_height = pil_image.size
            image = vision.Image()
            image.content = image_bytes
            response = self.client.text_detection(image=image)

            if response.error.message:
                raise Exception(f"Vision API error: {response.error.message}")

            for annotation in response.text_annotations[1:]:
                vertices = annotation.bounding_poly.vertices

                xs = [v.x for v in vertices]
                ys = [v.y for v in vertices]

                text_blocks.append({
                    'text': annotation.description,
                    'bounding_box': {
                        'left': min(xs) / img_width,
                        'top': min(ys) / img_height,
                        'width': (max(xs) - min(xs)) / img_width,
                        'height': round((max(ys) - min(ys)) / img_height, 4)
                    }
                })

            if response.full_text_annotation:
                for page in response.full_text_annotation.pages:
                    if page.property and page.property.detected_languages:
                        languages = page.property.detected_languages
                        if languages:
                            detected_language = languages[0].language_code
                            language_confidence = languages[0].confidence

            return {
                'blocks': text_blocks,
                'language': detected_language,
                'language_confidence': language_confidence,
                'total_blocks': len(text_blocks),
            }

        except Exception as e:
            logger.error(f"OCR failed: {e}")
            raise