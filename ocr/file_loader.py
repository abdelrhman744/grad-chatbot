"""
file_loader.py

Loads supported document types and converts them into a common
DocumentData object for the OCR pipeline.
"""

from pathlib import Path
import json

import cv2
import fitz
import numpy as np
from PIL import Image

from .config import Config
from .models import DocumentData, PageData


class FileLoader:

    def __init__(self):
        pass

    # ==========================================================
    # Public Function
    # ==========================================================

    def load(self, file_path: str) -> DocumentData:
        """
        Load any supported file.

        Parameters
        ----------
        file_path : str

        Returns
        -------
        DocumentData
        """

        self._validate_file(file_path)

        extension = self._get_extension(file_path)

        metadata = self._load_metadata(file_path)

        if extension in Config.IMAGE_EXTENSIONS:

            image = cv2.imread(file_path)

            if image is None:
                raise ValueError(f"Cannot read image: {file_path}")

            page = self._load_image(
                image=image,
                page_number=1,
                source_name=Path(file_path).name
            )

            return DocumentData(
                filename=Path(file_path).name,
                file_type=extension,
                metadata=metadata,
                pages=[page]
            )

        elif extension in Config.PDF_EXTENSIONS:

            return self._load_pdf(
                file_path=file_path,
                metadata=metadata
            )

        elif extension in Config.TEXT_EXTENSIONS:

            return self._load_text(
                file_path=file_path,
                metadata=metadata
            )

        elif extension in Config.PPT_EXTENSIONS:

            return self._load_pptx(
                file_path=file_path,
                metadata=metadata
            )

        else:
            raise ValueError(f"Unsupported file type: {extension}")

    # ==========================================================
    # Image
    # ==========================================================

    def _load_image(
        self,
        image,
        page_number: int,
        source_name: str,
        text: str = ""
    ) -> PageData:
        """
        Convert an OpenCV image into a PageData object.
        """

        return PageData(
            page_number=page_number,
            source_name=source_name,
            original_image=image,
            text=text
        )

    # ==========================================================
    # PDF
    # ==========================================================

    def _load_pdf(
        self,
        file_path: str,
        metadata: dict
    ) -> DocumentData:

        pdf = fitz.open(file_path)

        pages = []

        for page_number, page in enumerate(pdf, start=1):

            pix = page.get_pixmap(dpi=Config.DEFAULT_DPI)

            img = Image.frombytes(
                "RGB",
                [pix.width, pix.height],
                pix.samples
            )

            image = cv2.cvtColor(
                np.array(img),
                cv2.COLOR_RGB2BGR
            )

            page_data = self._load_image(
                image=image,
                page_number=page_number,
                source_name=Path(file_path).name
            )

            pages.append(page_data)

        pdf.close()

        return DocumentData(
            filename=Path(file_path).name,
            file_type=".pdf",
            metadata=metadata,
            pages=pages
        )

    # ==========================================================
    # TXT
    # ==========================================================

    def _load_text(
        self,
        file_path: str,
        metadata: dict
    ) -> DocumentData:

        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        page = self._load_image(
            image=None,
            page_number=1,
            source_name=Path(file_path).name,
            text=text
        )

        return DocumentData(
            filename=Path(file_path).name,
            file_type=".txt",
            metadata=metadata,
            pages=[page]
        )

    # ==========================================================
    # PPTX
    # ==========================================================

    def _load_pptx(
        self,
        file_path: str,
        metadata: dict
    ):

        raise NotImplementedError(
            "PPTX support will be implemented later."
        )

    # ==========================================================
    # Metadata
    # ==========================================================

    def _load_metadata(self, file_path: str) -> dict:
        """
        Loads metadata from a JSON file with the same name.

        Example:
            lecture.pdf
            lecture.json
        """

        json_path = Path(file_path).with_suffix(".json")

        if not json_path.exists():
            return {}

        with open(json_path, "r", encoding="utf-8") as file:
            return json.load(file)

    # ==========================================================
    # Helpers
    # ==========================================================

    def _validate_file(self, file_path: str):

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        extension = path.suffix.lower()

        if extension not in Config.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension: {extension}"
            )

    def _get_extension(self, file_path: str):

        return Path(file_path).suffix.lower()