
import faiss
import numpy as np

from rag.embeddings import embed_texts, embed_clip_text


class VectorStore:

    def __init__(self):
        self.text_index = None
        self.image_index = None

        self.text_docs = []
        self.image_docs = []

    def build(self, text_docs, image_docs, image_vecs):

        self.text_docs = text_docs
        self.image_docs = image_docs

        # Text FAISS index
        if text_docs:

            text_vectors = embed_texts(
                [doc.page_content for doc in text_docs]
            )

            dimension = text_vectors.shape[1]

            self.text_index = faiss.IndexFlatIP(dimension)

            self.text_index.add(text_vectors)

        else:
            self.text_index = None

        # Image FAISS index
        if image_vecs:

            image_vectors = np.asarray(
                image_vecs,
                dtype="float32"
            )

            dimension = image_vectors.shape[1]

            self.image_index = faiss.IndexFlatIP(dimension)

            self.image_index.add(image_vectors)

        else:
            self.image_index = None

        print(
            f"Indexed {len(text_docs)} text chunks "
            f"and {len(image_docs)} images."
        )

    def search_text(self, query, k=8):

        if self.text_index is None:
            return []

        query_vector = embed_texts([query])

        scores, ids = self.text_index.search(
            query_vector,
            min(k, self.text_index.ntotal)
        )

        results = []

        for idx, score in zip(ids[0], scores[0]):

            if idx == -1:
                continue

            results.append(
                (self.text_docs[idx], float(score))
            )

        return results

    def search_images(self, query, k=3):

        if self.image_index is None:
            return []

        query_vector = embed_clip_text(query)[None, :]

        scores, ids = self.image_index.search(
            query_vector,
            min(k, self.image_index.ntotal)
        )

        results = []

        for idx, score in zip(ids[0], scores[0]):

            if idx == -1:
                continue

            results.append(
                (self.image_docs[idx], float(score))
            )

        return results
