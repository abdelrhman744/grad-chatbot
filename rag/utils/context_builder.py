"""
context_builder.py

Build context from retrieved documents.
"""


def build_context(documents):

    context = ""

    for i, document in enumerate(documents, start=1):

        metadata = document.get("metadata", {})

        context += (
            f"Document {i}\n"
            f"Title: {metadata.get('title', 'Unknown')}\n"
            f"Content:\n"
            f"{document['text']}\n\n"
        )

    return context