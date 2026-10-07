import math

from production_rag.embeddings import HashingEmbedder


def test_hashing_embedder_is_deterministic_and_normalized() -> None:
    embedder = HashingEmbedder(dim=128)
    [a], [b] = embedder.embed(["refund policy"]), embedder.embed(["refund policy"])
    assert a == b
    assert len(a) == 128
    assert math.isclose(sum(x * x for x in a), 1.0, rel_tol=1e-9)


def test_similar_texts_score_higher() -> None:
    embedder = HashingEmbedder(dim=256)
    q, near, far = embedder.embed(
        ["refund window days", "refunds within 30 days window", "two-factor authentication"]
    )
    dot = lambda x, y: sum(i * j for i, j in zip(x, y, strict=True))  # noqa: E731
    assert dot(q, near) > dot(q, far)
