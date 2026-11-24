
from config import (
    TEXT_MIN_SCORE,
    IMAGE_MIN_SCORE,
    TEXT_TOP_K,
    IMAGE_TOP_K,
    MAX_CONTEXT_CHUNKS,
    MAX_CONTEXT_IMAGES
)


def split_by_access(hits, allowed):

    ok = [
        (doc, score)
        for doc, score in hits
        if doc.metadata["access"] in allowed
        and score >= TEXT_MIN_SCORE
    ]

    blocked = [
        (doc, score)
        for doc, score in hits
        if doc.metadata["access"] not in allowed
        and score >= TEXT_MIN_SCORE
    ]

    return ok, blocked


def needs_higher_access(ok, blocked):

    if not blocked:
        return False

    best_ok = max(
        (score for _, score in ok),
        default=-1.0
    )

    best_blocked = max(
        score for _, score in blocked
    )

    return best_blocked > best_ok


def retrieve_context(query, vector_store, allowed, image_store):

    # 1. Search text
    text_hits = vector_store.search_text(
        query,
        k=TEXT_TOP_K
    )

    ok_text, blocked_text = split_by_access(
        text_hits,
        allowed
    )

    # 2. Search images
    image_hits = vector_store.search_images(
        query,
        k=IMAGE_TOP_K
    )

    ok_images = [
        (doc, score)
        for doc, score in image_hits
        if doc.metadata["access"] in allowed
        and score >= IMAGE_MIN_SCORE
    ]

    blocked_images = [
        (doc, score)
        for doc, score in image_hits
        if doc.metadata["access"] not in allowed
        and score >= IMAGE_MIN_SCORE
    ]

    # 3. Determine whether authentication is required
    text_login_required = needs_higher_access(
        ok_text,
        blocked_text
    )

    image_login_required = needs_higher_access(
        ok_images,
        blocked_images
    )

    # 4. Keep only authorized text
    ok_docs = [
        doc for doc, score in ok_text
    ][:MAX_CONTEXT_CHUNKS]

    # 5. Collect authorized images from relevant text pages
    image_ids = []

    for doc in ok_docs[:3]:

        page_no = doc.metadata["page"]

        for image_doc in vector_store.image_docs:

            if (
                image_doc.metadata["page"] == page_no
                and image_doc.metadata["access"] in allowed
            ):

                image_id = image_doc.metadata["image_id"]

                if image_id in image_store:
                    image_ids.append(image_id)

    # 6. Add directly retrieved authorized images
    for doc, score in ok_images:

        image_id = doc.metadata["image_id"]

        if image_id in image_store:
            image_ids.append(image_id)

    # Remove duplicates
    image_ids = list(dict.fromkeys(image_ids))

    image_ids = image_ids[:MAX_CONTEXT_IMAGES]

    # 7. Return only authorized content
    return {
        "documents": ok_docs,
        "image_ids": image_ids,
        "requires_login": (
            text_login_required or image_login_required
        ),
        "has_content": bool(ok_docs or image_ids)
    }
