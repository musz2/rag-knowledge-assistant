from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock


@dataclass
class VectorRecord:
    id: str
    text: str
    source: str
    embedding: list[float]


class JsonVectorStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = Lock()
        self._records: list[VectorRecord] = []
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self._records = []
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self._records = [VectorRecord(**item) for item in data]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps([asdict(record) for record in self._records], indent=2),
            encoding="utf-8",
        )

    def add(self, records: list[VectorRecord]) -> None:
        with self._lock:
            self._records.extend(records)
            self.save()

    def search(self, query_embedding: list[float], top_k: int) -> list[VectorRecord]:
        scored = [
            (cosine_similarity(query_embedding, record.embedding), record)
            for record in self._records
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored[:top_k]]

    def count(self) -> int:
        return len(self._records)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left)) or 1.0
    right_norm = math.sqrt(sum(b * b for b in right)) or 1.0
    return dot / (left_norm * right_norm)
