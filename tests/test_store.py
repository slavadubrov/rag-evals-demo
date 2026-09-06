import pytest

from rag_evals.index.qdrant_store import QdrantStore
from rag_evals.types import Chunk


def test_embedded_roundtrip_and_lock_release(tmp_path):
    path = str(tmp_path / "qdrant")
    with QdrantStore(path=path, dense_dim=2) as store:
        store.ensure_collection()
        store.upsert(
            [
                Chunk("a::0", "a", "evidence", 0, {"tenant": "allowed"}),
                Chunk("b::0", "b", "private", 0, {"tenant": "blocked"}),
            ],
            [[1.0, 0.0], [0.0, 1.0]],
            [([1], [1.0]), ([2], [1.0])],
        )
        assert store.count() == 2
        assert store.search_dense([1.0, 0.0], predicates={"tenant": "allowed"})[0].doc_id == "a"
        assert store.search_sparse([1], [1.0])[0].doc_id == "a"
        assert store.survivor_ids({"tenant": "allowed"}) == {"a"}
    with (
        QdrantStore(path=path, dense_dim=3) as reopened,
        pytest.raises(ValueError, match="dimension"),
    ):
        reopened.ensure_collection()
