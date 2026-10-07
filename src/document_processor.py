from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_pdf_documents(pdf_file) -> list[Document]:
    """
    Extract text from an uploaded PDF.

    Each page becomes a LangChain Document with metadata
    containing the source filename and page number.
    """

    reader = PdfReader(pdf_file)

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        text = text.strip()

        if not text:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": Path(pdf_file.name).name,
                    "page": page_number,
                },
            )
        )

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Split policy pages into smaller chunks for retrieval.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
        ],
    )

    chunks = splitter.split_documents(documents)

    return chunks