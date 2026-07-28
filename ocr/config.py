"""
config.py
Configuration file for the OCR package.

This file contains all configurable settings used across the OCR pipeline.
Changing values here will automatically affect the rest of the project.
"""

import shutil
from pathlib import Path


class Config:
    """
    Central configuration class for the OCR system.
    """

    # ==================================================
    # Project Paths
    # ==================================================

    # Folder containing this config.py file
    OCR_FOLDER = Path(__file__).resolve().parent

    # Folder containing input files
    DATA_FOLDER = OCR_FOLDER / "data"

    # Folder for successfully processed files
    PROCESSED_FOLDER = OCR_FOLDER / "processed"

    # Folder for temporary files
    TEMP_FOLDER = OCR_FOLDER / "temp"

    # Folder for OCR outputs
    OUTPUT_FOLDER = OCR_FOLDER / "output"

    # ==================================================
    # Supported File Types
    # ==================================================

    IMAGE_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp"
    }

    PDF_EXTENSIONS = {
        ".pdf"
    }

    PPT_EXTENSIONS = {
        ".pptx"
    }

    TEXT_EXTENSIONS = {
        ".txt"
    }

    SUPPORTED_EXTENSIONS = (
        IMAGE_EXTENSIONS
        | PDF_EXTENSIONS
        | PPT_EXTENSIONS
        | TEXT_EXTENSIONS
    )

    # ==================================================
    # OCR Settings
    # ==================================================

    # Path to Tesseract executable.
    # Falls back to whatever is on PATH first (Linux/Mac), then to the
    # default Windows install location, so the same config works cross-platform.
    TESSERACT_CMD = shutil.which("tesseract") or r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    # Arabic + English
    OCR_LANGUAGE = "ara+eng"

    # OCR Engine Mode (1 = LSTM only, matches ocr2's higher-accuracy setting)
    TESSERACT_OEM = 1

    # Page Segmentation Mode used when only a single pass is needed
    # (e.g. the empty-result fallback pass)
    TESSERACT_PSM = 6

    # Every processed image is run through Tesseract once per PSM mode here,
    # so each preprocessing method effectively gets several OCR attempts.
    # 6 = uniform block, 3 = auto, 4 = single column, 11 = sparse text
    TESSERACT_PSM_MODES = [6, 3, 4, 11]

    TESSERACT_CONFIG = (
        f"--oem {TESSERACT_OEM} "
        f"--psm {TESSERACT_PSM}"
    )

    def tesseract_config_for(psm: int) -> str:
        return f"--oem {Config.TESSERACT_OEM} --psm {psm}"

    # ==================================================
    # Image Processing
    # ==================================================

    DEFAULT_DPI = 300

    RESIZE_SCALE = 2.0

    # Images smaller than this get upscaled before preprocessing/OCR,
    # since small scans hurt Tesseract accuracy badly.
    MIN_HEIGHT = 800
    MIN_WIDTH = 600

    PREPROCESSING_METHODS = [
        "original",
        "grayscale",
        "otsu",
        "adaptive",
        "denoise",
        "sharpen",
        "contrast",
    ]

    # ==================================================
    # Ensemble Settings
    # ==================================================

    # "merge"  -> combine unique lines from every OCR attempt (best recall,
    #             matches ocr2's behaviour)
    # "best"   -> pick the single highest-scoring attempt (original ocr/
    #             behaviour, kept for cases where merge is too aggressive)
    ENSEMBLE_STRATEGY = "best"

    ENSEMBLE_CONFIDENCE_WEIGHT = 1.0

    # Slightly higher so complete pages (with full numbers/tables) are preferred
    ENSEMBLE_LENGTH_WEIGHT = 0.035

    # ==================================================
    # Output
    # ==================================================

    SAVE_PREPROCESSED_IMAGES = False

    SAVE_OCR_RESULTS = False


# ==================================================
# Create folders automatically
# ==================================================

Config.DATA_FOLDER.mkdir(exist_ok=True)
Config.PROCESSED_FOLDER.mkdir(exist_ok=True)
Config.TEMP_FOLDER.mkdir(exist_ok=True)
Config.OUTPUT_FOLDER.mkdir(exist_ok=True)