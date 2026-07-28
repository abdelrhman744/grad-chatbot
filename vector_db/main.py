"""
main.py
"""

from .pipeline import VectorDBPipeline


def main():

    # Build the vector database
    pipeline = VectorDBPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()