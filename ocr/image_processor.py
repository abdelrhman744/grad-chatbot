"""
image_processor.py

Applies multiple image preprocessing techniques to a PageData object.
The processed images are stored inside page.processed_images.
"""

import cv2

from .config import Config
from .models import PageData


class ImageProcessor:
    """
    Generates multiple preprocessed versions of an image
    for ensemble OCR.
    """

    def process(self, page: PageData) -> PageData:
        """
        Apply all preprocessing methods specified in Config.

        Parameters
        ----------
        page : PageData

        Returns
        -------
        PageData
        """

        # Skip pages that already contain text (e.g., TXT files)
        if page.original_image is None:
            return page

        # Upscale small images first so every downstream method benefits
        base_image = self._upscale_if_small(page.original_image)

        for method in Config.PREPROCESSING_METHODS:

            if method == "original":
                page.processed_images["original"] = base_image

            elif method == "grayscale":
                page.processed_images["grayscale"] = self._grayscale(
                    base_image
                )

            elif method == "otsu":
                page.processed_images["otsu"] = self._otsu(
                    base_image
                )

            elif method == "adaptive":
                page.processed_images["adaptive"] = self._adaptive(
                    base_image
                )

            elif method == "denoise":
                page.processed_images["denoise"] = self._denoise(
                    base_image
                )

            elif method == "sharpen":
                page.processed_images["sharpen"] = self._sharpen(
                    base_image
                )

            elif method == "contrast":
                page.processed_images["contrast"] = self._contrast(
                    base_image
                )

        return page

    # ======================================================
    # Upscaling
    # ======================================================

    def _upscale_if_small(self, image):
        """
        Upscale the image if it is smaller than Config.MIN_HEIGHT /
        Config.MIN_WIDTH. Small scans hurt OCR accuracy badly, so we
        enlarge them before any other preprocessing happens.
        """

        h, w = image.shape[:2]

        if h >= Config.MIN_HEIGHT and w >= Config.MIN_WIDTH:
            return image

        scale = max(
            Config.MIN_HEIGHT / h,
            Config.MIN_WIDTH / w,
            Config.RESIZE_SCALE
        )

        return cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

    # ======================================================
    # Private Processing Methods
    # ======================================================

    def _grayscale(self, image):
        """
        Convert image to grayscale.
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _otsu(self, image):
        """
        Apply Otsu thresholding (with a light blur first to reduce noise).
        """

        gray = self._grayscale(image)

        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        _, otsu = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        return otsu

    def _adaptive(self, image):
        """
        Apply adaptive thresholding (with a light blur first to reduce noise).
        """

        gray = self._grayscale(image)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        adaptive = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11
        )

        return adaptive

    def _denoise(self, image):
        """
        Denoise the image (Non-Local Means) then apply Otsu thresholding.
        Helps with grainy scans / photographed documents.
        """

        gray = self._grayscale(image)

        denoised = cv2.fastNlMeansDenoising(
            gray,
            h=15,
            templateWindowSize=7,
            searchWindowSize=21
        )

        _, processed = cv2.threshold(
            denoised,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        return processed

    def _sharpen(self, image):
        """
        Apply an unsharp mask to sharpen faint text, then Otsu threshold.
        """

        gray = self._grayscale(image)

        blurred = cv2.GaussianBlur(gray, (0, 0), 3)

        sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

        _, processed = cv2.threshold(
            sharpened,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        return processed

    def _contrast(self, image):
        """
        Apply CLAHE contrast enhancement, then Otsu threshold.
        Helps with faded/low-contrast text.
        """

        gray = self._grayscale(image)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        equalized = clahe.apply(gray)

        _, processed = cv2.threshold(
            equalized,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        return processed