"""SQLQueryCache: in-memory TTL cache for query/answer payloads.

Keys combine the normalised prompt with everything else that can change the result: SQL, parameters,
date range, filters, permission scope and dataset version. The cache lives in process memory and is
reset whenever the app restarts; changing the dataset version clears it so stale results are never served.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Any


class SQLQueryCache:
    def __init__(self, ttl_seconds: float = 900, dataset_version: str = "unknown", clock=time.monotonic):
        self.ttl = float(ttl_seconds)
        self.dataset_version = dataset_version
        self._clock = clock
        self._store: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def normalize(prompt: str) -> str:
        return " ".join((prompt or "").strip().lower().split())

    def key(self, prompt: str, context: dict | None = None) -> str:
        ident = {"prompt": self.normalize(prompt), "context": context or {}, "dataset_version": self.dataset_version}
        return hashlib.sha256(json.dumps(ident, sort_keys=True, default=str).encode()).hexdigest()

    def get(self, prompt: str, context: dict | None = None):
        k = self.key(prompt, context)
        with self._lock:
            item = self._store.get(k)
            if item is None or self._clock() - item[0] > self.ttl:
                if item is not None:
                    del self._store[k]
                self.misses += 1
                return None
            self.hits += 1
            return item[1]

    def set(self, prompt: str, payload, context: dict | None = None) -> None:
        with self._lock:
            self._store[self.key(prompt, context)] = (self._clock(), payload)

    def saved_at(self, prompt: str, context: dict | None = None) -> float | None:
        item = self._store.get(self.key(prompt, context))
        return item[0] if item else None

    def set_dataset_version(self, version: str) -> None:
        if version != self.dataset_version:
            self.dataset_version = version
            self.clear()

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        return len(self._store)

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "entries": len(self), "ttl_s": self.ttl}
