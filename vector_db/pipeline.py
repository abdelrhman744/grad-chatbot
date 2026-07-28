"""
pipeline.py

Build the vector database from OCR output and Excel documents.
"""

import json
import shutil
from pathlib import Path

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from .whoosh_index import WhooshIndexer
from .cleaning import Cleaner, is_meaningful
from .arabic_normalizer import ArabicNormalizer
from .chunking import SemanticChunker
from .structured_chunker import StructuredChunker
from .excel_loader import ExcelLoader
from .embeddings import EmbeddingGenerator
from .upload import QdrantUploader


class VectorDBPipeline:

    def __init__(self):

        BASE_DIR = Path(__file__).resolve().parent.parent

        # =====================================================
        # OCR
        # =====================================================

        self.output_folder = BASE_DIR / "ocr" / "output"

        self.done_folder = BASE_DIR / "vector_db" / "done_docs"
        self.done_folder.mkdir(exist_ok=True)

        # =====================================================
        # Excel
        # =====================================================

        self.xlsx_folder = BASE_DIR / "vector_db" / "xlsx"

        self.done_xlsx_folder = (
            BASE_DIR / "vector_db" / "done_xlsx"
        )
        self.done_xlsx_folder.mkdir(exist_ok=True)

    

        # =====================================================
        # Components
        # =====================================================

        self.whoosh = WhooshIndexer()

        self.chunker = SemanticChunker()
        self.structured_chunker = StructuredChunker()

        self.excel_loader = ExcelLoader()

        self.embedder = EmbeddingGenerator()

        self.uploader = QdrantUploader()

    # =====================================================
    # Public Method
    # =====================================================

    def run(self):

        print("=" * 60)
        print("OCR PIPELINE")
        print("=" * 60)

        self._run_ocr_pipeline()

        print("\n" + "=" * 60)
        print("EXCEL PIPELINE")
        print("=" * 60)

        self._run_excel_pipeline()

        print("\nDone!")

    # =====================================================
    # OCR Pipeline
    # =====================================================

    def _run_ocr_pipeline(self):

        if not self.output_folder.exists():

            print("OCR output folder not found.")

            return

        folders = sorted(
            self.output_folder.iterdir()
        )

        for folder in folders:

            if not folder.is_dir():
                continue

            print(f"\nProcessing OCR: {folder.name}")

            try:

                self._process_ocr(folder)

                destination = self.done_folder / folder.name

                if destination.exists():
                    shutil.rmtree(destination)

                shutil.move(
                    str(folder),
                    str(destination)
                )

                print("Done.")

            except Exception as e:

                print(f"Failed: {e}")

    # =====================================================
    # Excel Pipeline
    # =====================================================

    def _run_excel_pipeline(self):

        if not self.xlsx_folder.exists():

            print("Excel folder not found.")

            return

        folders = sorted(
            self.xlsx_folder.iterdir()
        )

        for folder in folders:

            if not folder.is_dir():
                continue

            print(f"\nProcessing Excel: {folder.name}")

            try:

                self._process_excel(folder)

                destination = (
                    self.done_xlsx_folder /
                    folder.name
                )

                if destination.exists():
                    shutil.rmtree(destination)

                shutil.move(
                    str(folder),
                    str(destination)
                )

                print("Done.")

            except Exception as e:

                print(f"Failed: {e}")

    # =====================================================
    # OCR
    # =====================================================

    def _process_ocr(
        self,
        folder: Path
    ):

        text_path = folder / "text.txt"

        metadata_path = folder / "metadata.json"

        if not text_path.exists():
            return

        if not metadata_path.exists():
            return

        with open(
            text_path,
            "r",
            encoding="utf-8"
        ) as file:

            text = file.read()

        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as file:

            metadata = json.load(file)

        cleaned_text = Cleaner(text)

        # Normalize Arabic (diacritics, alef/ya/waw/ta-marbuta variants)
        # BEFORE chunking, so the normalized form is what actually gets
        # embedded and stored — not just what Whoosh sees.
        normalized_text = ArabicNormalizer.normalize(
            cleaned_text
        )

        chunks = self.chunker.chunk(
            normalized_text
        )

        # Drop OCR noise / empty fragments before they're embedded
        # and written to Qdrant or Whoosh.
        chunks = [
            chunk
            for chunk in chunks
            if is_meaningful(chunk)
        ]

        if not chunks:
            print("No meaningful chunks after cleaning/normalization.")
            return

        documents = self.embedder.embed(
            chunks
        )

        for document in documents:

            document["metadata"] = metadata

        self.uploader.upload(documents)

        self.whoosh.add_documents(documents)

    # =====================================================
    # Excel
    # =====================================================

    def _process_excel(
        self,
        folder: Path
    ):

        workbook = self.excel_loader.load(
            folder
        )

        documents = self.structured_chunker.chunk(
            workbook
        )

        # Normalize Arabic cell/column values before embedding, same
        # as the OCR path, so vector search benefits from it too.
        for document in documents:

            document["text"] = ArabicNormalizer.normalize(
                document["text"]
            )

        # Drop empty/near-empty table chunks (e.g. all-blank row batches).
        documents = [
            document
            for document in documents
            if is_meaningful(document["text"])
        ]

        if not documents:
            print("No meaningful rows after cleaning/normalization.")
            return

        documents = self.embedder.embed_documents(
            documents
        )

        self.uploader.upload(documents)

        self.whoosh.add_documents(documents)