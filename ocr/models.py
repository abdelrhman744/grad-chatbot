"""
models.py

Data models used throughout the OCR pipeline.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


# ==========================================================
# OCR Result
# ==========================================================

@dataclass
class OCRResult:
    """
    Stores the OCR output for one preprocessing method.
    """

    text: str
    confidence: float
    preprocessing_method: str


# ==========================================================
# Page Data
# ==========================================================

@dataclass
class PageData:
    """
    Represents one page of a document.
    """

    # Basic Information
    page_number: int
    source_name: str

    # Original image loaded from the file
    original_image: Any = None

    # Dictionary of processed images
    #
    # Example:
    # {
    #     "original": image,
    #     "gray": gray_image,
    #     "otsu": otsu_image,
    #     "adaptive": adaptive_image
    # }
    processed_images: Dict[str, Any] = field(default_factory=dict)

    # OCR results for every preprocessing method
    #
    # Example:
    # {
    #     "gray": OCRResult(...),
    #     "otsu": OCRResult(...),
    #     "adaptive": OCRResult(...)
    # }
    ocr_results: Dict[str, OCRResult] = field(default_factory=dict)

    # Final selected OCR text
    text: str = ""

    # Final confidence score
    confidence: float = 0.0


# ==========================================================
# Document Data
# ==========================================================

@dataclass
class DocumentData:
    """
    Represents the entire document.
    """

    # File information
    filename: str
    file_type: str

    # Metadata loaded from <filename>.json
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Pages of the document
    pages: List[PageData] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """
        Returns the concatenated text of all pages.
        """

        return "\n\n".join(page.text for page in self.pages)