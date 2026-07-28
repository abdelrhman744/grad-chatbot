"""
main.py

Runs the complete OCR pipeline.
"""

from pathlib import Path
import shutil

from .config import Config
from .file_loader import FileLoader
from .image_processor import ImageProcessor
from .ocr_engine import TesseractEngine
from .ensemble import Ensemble


class OCRPipeline:

    def __init__(self):

        self.loader = FileLoader()
        self.processor = ImageProcessor()
        self.engine = TesseractEngine()
        self.ensemble = Ensemble()

    # ==========================================================
    # Process one document
    # ==========================================================

    def run(self, file_path: str):

        document = self.loader.load(file_path)

        for page in document.pages:

            # TXT files already contain text
            if page.original_image is None:
                continue

            self.processor.process(page)
            self.engine.process(page)
            self.ensemble.process(page)

        return document

    # ==========================================================
    # Process all documents
    # ==========================================================

    def run_all(self):

        documents = []

        files = sorted(Config.DATA_FOLDER.iterdir())

        for file in files:

            if not file.is_file():
                continue

            # Ignore metadata files
            if file.suffix.lower() == ".json":
                continue

            if file.suffix.lower() not in Config.SUPPORTED_EXTENSIONS:
                continue

            print(f"\nProcessing: {file.name}")

            try:

                document = self.run(str(file))

                self.save_output(document)

                self.move_to_processed(file)

                documents.append(document)

                print(f"✓ Finished: {file.name}")

            except Exception as e:

                print(f"✗ Failed: {file.name}")
                print(e)

        return documents

    # ==========================================================
    # Save OCR Output
    # ==========================================================

    def save_output(self, document):

        base_name = Path(document.filename).stem

        # Create output folder for this document
        document_folder = Config.OUTPUT_FOLDER / base_name
        document_folder.mkdir(exist_ok=True)

        # Save extracted text
        text_file = document_folder / "text.txt"

        with open(text_file, "w", encoding="utf-8") as file:
            file.write(document.full_text)

        # Copy metadata if it exists
        source_json = Config.DATA_FOLDER / f"{base_name}.json"

        if source_json.exists():

            shutil.copy2(
                source_json,
                document_folder / "metadata.json"
            )

    # ==========================================================
    # Move processed files
    # ==========================================================

    def move_to_processed(self, source_file: Path):

        # Move original document
        shutil.move(
            str(source_file),
            Config.PROCESSED_FOLDER / source_file.name
        )

        # Move metadata
        json_file = source_file.with_suffix(".json")

        if json_file.exists():

            shutil.move(
                str(json_file),
                Config.PROCESSED_FOLDER / json_file.name
            )


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    pipeline = OCRPipeline()

    documents = pipeline.run_all()

    print("\n" + "=" * 60)
    print(f"Processed {len(documents)} document(s).")
    print("=" * 60)