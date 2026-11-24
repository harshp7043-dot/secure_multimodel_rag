
import io
import re
import base64

import pymupdf as fitz
from PIL import Image

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    PDF_PATH,
    ALL_LEVELS,
    EXECUTIVE,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

from rag.embeddings import embed_image


MARKER = re.compile(r"CLASSIFICATION:\s*([A-Za-z]+)")


def page_access_level(page_text):
    """
    Missing or unknown classification defaults to EXECUTIVE.
    """

    match = MARKER.search(page_text)

    level = match.group(1).upper() if match else EXECUTIVE

    return level if level in ALL_LEVELS else EXECUTIVE


def add_image(
    pil_image,
    image_id,
    page_no,
    access,
    image_docs,
    image_vecs,
    image_store
):
    buffer = io.BytesIO()

    pil_image.save(buffer, format="PNG")

    image_store[image_id] = base64.b64encode(
        buffer.getvalue()
    ).decode()

    image_vecs.append(embed_image(pil_image))

    image_docs.append(
        Document(
            page_content=f"[Image: {image_id}]",
            metadata={
                "page": page_no,
                "type": "image",
                "image_id": image_id,
                "access": access
            }
        )
    )


def process_pdf(pdf_path=None):

    pdf_path = pdf_path or PDF_PATH

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    text_docs = []
    image_docs = []
    image_vecs = []
    image_store = {}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    print(f"Processing PDF: {pdf_path}")

    doc = fitz.open(str(pdf_path))

    try:
        for page_no, page in enumerate(doc):

            raw = page.get_text()

            access = page_access_level(raw)

            clean = MARKER.sub("", raw)

            # Process text
            if clean.strip():

                temp = Document(
                    page_content=clean,
                    metadata={
                        "page": page_no,
                        "type": "text",
                        "access": access
                    }
                )

                chunks = splitter.split_documents([temp])

                text_docs.extend(chunks)

            # Process embedded images
            found = False

            for idx, img in enumerate(page.get_images(full=True)):

                try:
                    base = doc.extract_image(img[0])

                    pil = Image.open(
                        io.BytesIO(base["image"])
                    ).convert("RGB")

                    add_image(
                        pil,
                        f"page_{page_no}_img_{idx}",
                        page_no,
                        access,
                        image_docs,
                        image_vecs,
                        image_store
                    )

                    found = True

                except Exception as error:
                    print(
                        f"Image processing error on page "
                        f"{page_no}: {error}"
                    )

            # Render pages containing drawings when no image was found
            if not found and page.get_drawings():

                try:
                    pix = page.get_pixmap(dpi=100)

                    pil = Image.frombytes(
                        "RGB",
                        [pix.width, pix.height],
                        pix.samples
                    )

                    add_image(
                        pil,
                        f"page_{page_no}_render",
                        page_no,
                        access,
                        image_docs,
                        image_vecs,
                        image_store
                    )

                except Exception as error:
                    print(
                        f"Page rendering error: {error}"
                    )

    finally:
        doc.close()

    return {
        "text_docs": text_docs,
        "image_docs": image_docs,
        "image_vecs": image_vecs,
        "image_store": image_store
    }
