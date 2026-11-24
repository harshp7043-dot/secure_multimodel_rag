
import functools

import torch
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer


@functools.lru_cache(maxsize=1)
def load_embedding_models():
    print("Loading embedding models...")

    text_encoder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    clip_processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    clip_model.eval()

    return text_encoder, clip_model, clip_processor


def _to_tensor(features):
    if isinstance(features, torch.Tensor):
        return features

    if hasattr(features, "pooler_output"):
        return features.pooler_output

    return features[0]


def embed_texts(texts):
    text_encoder, _, _ = load_embedding_models()

    return text_encoder.encode(
        texts,
        normalize_embeddings=True
    ).astype("float32")


def embed_image(pil_image):
    _, clip_model, clip_processor = load_embedding_models()

    inputs = clip_processor(
        images=pil_image,
        return_tensors="pt"
    )

    with torch.no_grad():
        features = _to_tensor(
            clip_model.get_image_features(**inputs)
        )

        features = features / features.norm(
            dim=-1,
            keepdim=True
        )

    return features.squeeze().numpy().astype("float32")


def embed_clip_text(text):
    _, clip_model, clip_processor = load_embedding_models()

    inputs = clip_processor(
        text=text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    )

    with torch.no_grad():
        features = _to_tensor(
            clip_model.get_text_features(**inputs)
        )

        features = features / features.norm(
            dim=-1,
            keepdim=True
        )

    return features.squeeze().numpy().astype("float32")
