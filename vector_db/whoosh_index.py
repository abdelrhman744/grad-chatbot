"""
whoosh_index.py

Create and update the Whoosh BM25 index.
"""

from pathlib import Path

from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import ID
from whoosh.fields import STORED
from whoosh.fields import TEXT
from whoosh.fields import Schema
from whoosh.index import create_in
from whoosh.index import exists_in
from whoosh.index import open_dir

from .arabic_normalizer import ArabicNormalizer


class WhooshIndexer:

    def __init__(self):

        base_dir = Path(__file__).resolve().parent

        self.index_dir = base_dir / "whoosh_index"

        self.index_dir.mkdir(exist_ok=True)

        analyzer = StemmingAnalyzer()

        self.schema = Schema(

            id=ID(
                stored=True,
                unique=True
            ),

            title=TEXT(
                stored=True,
                analyzer=analyzer,
                field_boost=3.0
            ),

            text=TEXT(
                stored=True,
                analyzer=analyzer
            ),

            document_id=STORED,

            document_type=STORED,

            uploaded_by=STORED,

            roles=STORED,

            document_scope=STORED
        )

        if not exists_in(self.index_dir):

            self.index = create_in(
                self.index_dir,
                self.schema
            )

        else:

            self.index = open_dir(
                self.index_dir
            )

    # =====================================================
    # Public
    # =====================================================

    def add_documents(
        self,
        documents
    ):

        writer = self.index.writer()

        for document in documents:

            metadata = document["metadata"]

            # Title comes straight from metadata.json and never passes
            # through pipeline.py's normalization step, so it still
            # needs normalizing here.
            title = ArabicNormalizer.normalize(
                metadata.get("title", "")
            )

            # document["text"] is already normalized upstream in
            # pipeline.py before embedding, so no need to redo it here.
            text = document["text"]

            writer.update_document(

                id=document["id"],

                title=title,

                text=text,

                document_id=metadata.get(
                    "document_id",
                    ""
                ),

                document_type=metadata.get(
                    "document_type",
                    ""
                ),

                uploaded_by=metadata.get(
                    "uploaded_by",
                    ""
                ),

                roles=str(
                    metadata.get(
                        "roles",
                        []
                    )
                ),

                document_scope=str(
                    metadata.get(
                        "document_scope",
                        []
                    )
                )

            )

        writer.commit()