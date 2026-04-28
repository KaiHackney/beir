from __future__ import annotations

import heapq
import logging
import os
import pickle
from collections import defaultdict
from collections.abc import Iterable

import numpy as np
import torch
from tqdm.autonotebook import trange

from .. import BaseSearch

logger = logging.getLogger(__name__)


class SparseInvertedSearch(BaseSearch):
    """Simple SPLADE-style inverted-index search.

    This indexes sparse impact vectors as posting lists and scores queries with
    sparse dot product. It is intended as a lightweight BEIR-native indexing
    path for experiments before moving to Lucene/Pyserini/PISA-scale engines.
    """

    def __init__(
        self,
        model,
        batch_size: int = 32,
        corpus_chunk_size: int = 10000,
        index_dir: str | None = None,
        initialize: bool = True,
        doc_max_active_dims: int | None = None,
        query_max_active_dims: int | None = None,
        doc_weight_threshold: float = 0.0,
        query_weight_threshold: float = 0.0,
        impact_scale: float | None = None,
        show_progress_bar: bool = True,
        **kwargs,
    ):
        self.model = model
        self.batch_size = batch_size
        self.corpus_chunk_size = corpus_chunk_size
        self.index_dir = index_dir
        self.initialize = initialize
        self.doc_max_active_dims = doc_max_active_dims
        self.query_max_active_dims = query_max_active_dims
        self.doc_weight_threshold = doc_weight_threshold
        self.query_weight_threshold = query_weight_threshold
        self.impact_scale = impact_scale
        self.show_progress_bar = show_progress_bar
        self.results = {}
        self.doc_ids: list[str] = []
        self.postings: dict[int, list[tuple[int, float]]] = {}

    def search(
        self,
        corpus: dict[str, dict[str, str]],
        queries: dict[str, str],
        top_k: int,
        score_function: str = "dot",
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        if score_function != "dot":
            raise ValueError("SparseInvertedSearch only supports dot-product scoring.")

        if self.index_dir and os.path.exists(self._index_path()) and not self.initialize:
            self.load(self.index_dir)
        elif self.initialize or not self.postings:
            self.index(corpus)
            if self.index_dir:
                self.save(self.index_dir)

        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]
        self.results = {qid: {} for qid in query_ids}

        logger.info("Searching sparse inverted index...")
        for start in trange(0, len(query_texts), self.batch_size, desc="query", disable=not self.show_progress_bar):
            batch_queries = query_texts[start : start + self.batch_size]
            query_vectors = self.model.encode_queries(
                batch_queries,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
            for offset, (indices, values) in enumerate(
                self._iter_sparse_rows(
                    query_vectors,
                    max_active_dims=self.query_max_active_dims,
                    weight_threshold=self.query_weight_threshold,
                    quantize=False,
                )
            ):
                qid = query_ids[start + offset]
                scores = defaultdict(float)
                for term_id, query_weight in zip(indices, values):
                    for doc_idx, doc_weight in self.postings.get(int(term_id), []):
                        scores[doc_idx] += float(query_weight) * float(doc_weight)

                best = heapq.nlargest(top_k + 1, scores.items(), key=lambda item: item[1])
                self.results[qid] = {}
                for doc_idx, score in best:
                    doc_id = self.doc_ids[doc_idx]
                    if doc_id == qid:
                        continue
                    self.results[qid][doc_id] = float(score)
                    if len(self.results[qid]) >= top_k:
                        break

        return self.results

    def index(self, corpus: dict[str, dict[str, str]]) -> None:
        logger.info("Building sparse inverted index...")
        self.doc_ids = list(corpus.keys())
        documents = [corpus[doc_id] for doc_id in self.doc_ids]
        postings = defaultdict(list)

        for start in trange(
            0,
            len(documents),
            self.corpus_chunk_size,
            desc="index",
            disable=not self.show_progress_bar,
        ):
            docs_batch = documents[start : start + self.corpus_chunk_size]
            embeddings = self.model.encode_corpus(
                docs_batch,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
            rows = self._iter_sparse_rows(
                embeddings,
                max_active_dims=self.doc_max_active_dims,
                weight_threshold=self.doc_weight_threshold,
                quantize=self.impact_scale is not None,
            )
            for row_offset, (indices, values) in enumerate(rows):
                doc_idx = start + row_offset
                for term_id, weight in zip(indices, values):
                    postings[int(term_id)].append((doc_idx, float(weight)))

        self.postings = dict(postings)
        logger.info(
            "Built sparse index with %d docs, %d posting lists, %.2f avg postings/doc.",
            len(self.doc_ids),
            len(self.postings),
            sum(len(v) for v in self.postings.values()) / max(len(self.doc_ids), 1),
        )

    def encode(
        self,
        corpus: dict[str, dict[str, str]],
        queries: dict[str, str],
        encode_output_path: str = "./embeddings/",
        overwrite: bool = False,
        query_filename: str = "queries.pkl",
        corpus_filename: str = "corpus.*.pkl",
        **kwargs,
    ) -> None:
        if os.path.exists(self._index_path(encode_output_path)) and not overwrite:
            logger.info("Sparse index already exists at %s, skipping indexing.", encode_output_path)
            return
        self.index(corpus)
        self.save(encode_output_path)

    def search_from_files(
        self,
        query_embeddings_file: str,
        corpus_embeddings_files: list[str],
        top_k: int,
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        raise NotImplementedError("Use search() with index_dir/load() for SparseInvertedSearch.")

    def save(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        with open(self._index_path(output_dir), "wb") as f:
            pickle.dump(
                {
                    "doc_ids": self.doc_ids,
                    "postings": self.postings,
                    "doc_max_active_dims": self.doc_max_active_dims,
                    "doc_weight_threshold": self.doc_weight_threshold,
                    "impact_scale": self.impact_scale,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logger.info("Saved sparse index to %s", self._index_path(output_dir))

    def load(self, input_dir: str) -> None:
        with open(self._index_path(input_dir), "rb") as f:
            data = pickle.load(f)
        self.doc_ids = data["doc_ids"]
        self.postings = data["postings"]
        logger.info("Loaded sparse index from %s", self._index_path(input_dir))

    def _index_path(self, index_dir: str | None = None) -> str:
        return os.path.join(index_dir or self.index_dir or ".", "sparse-inverted.pkl")

    def _iter_sparse_rows(
        self,
        embeddings,
        max_active_dims: int | None,
        weight_threshold: float,
        quantize: bool,
    ) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        matrix = self._to_dense_numpy(embeddings)
        for row in matrix:
            indices = np.flatnonzero(row > weight_threshold)
            values = row[indices].astype(np.float32, copy=False)
            if max_active_dims is not None and len(indices) > max_active_dims:
                keep = np.argpartition(values, -max_active_dims)[-max_active_dims:]
                indices = indices[keep]
                values = values[keep]
            if quantize:
                values = np.maximum(1, np.rint(values * self.impact_scale)).astype(np.float32)
            yield indices, values

    @staticmethod
    def _to_dense_numpy(embeddings) -> np.ndarray:
        if isinstance(embeddings, torch.Tensor):
            if embeddings.is_sparse:
                embeddings = embeddings.to_dense()
            embeddings = embeddings.detach().cpu().numpy()
        elif isinstance(embeddings, list):
            rows = []
            for embedding in embeddings:
                if isinstance(embedding, torch.Tensor):
                    if embedding.is_sparse:
                        embedding = embedding.to_dense()
                    rows.append(embedding.detach().cpu().numpy())
                else:
                    rows.append(np.asarray(embedding))
            embeddings = np.vstack(rows)
        else:
            embeddings = np.asarray(embeddings)

        if embeddings.ndim == 1:
            embeddings = np.expand_dims(embeddings, axis=0)
        return embeddings
