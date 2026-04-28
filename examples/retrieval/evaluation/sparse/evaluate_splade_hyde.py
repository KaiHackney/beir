from __future__ import annotations

import logging
import os
import pathlib

from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

dataset = os.getenv("BEIR_DATASET", "scifact")
split = os.getenv("BEIR_SPLIT", "test")
splade_model_name = os.getenv("SPLADE_MODEL", "naver/splade_v2_max")
splade_revision = os.getenv("SPLADE_REVISION", "refs/pr/1")
splade_backend = os.getenv("SPLADE_BACKEND", "masked_lm")
hyde_generator_model = os.getenv("HYDE_GENERATOR_MODEL", "google/flan-t5-base")
hyde_generator_slug = hyde_generator_model.replace("/", "_")
temperature = os.getenv("HYDE_TEMPERATURE")
top_p = os.getenv("HYDE_TOP_P")
fusion = os.getenv("HYDE_FUSION", "rrf")
fusion_alpha = float(os.getenv("HYDE_FUSION_ALPHA", "0.8"))
rrf_k = int(os.getenv("HYDE_RRF_K", "60"))


def _ranked_docs(run: dict[str, float]) -> list[str]:
    return [doc_id for doc_id, _ in sorted(run.items(), key=lambda item: item[1], reverse=True)]


def _minmax_scores(run: dict[str, float]) -> dict[str, float]:
    if not run:
        return {}
    values = list(run.values())
    low, high = min(values), max(values)
    if high == low:
        return {doc_id: 1.0 for doc_id in run}
    return {doc_id: (score - low) / (high - low) for doc_id, score in run.items()}


def fuse_results(
    base_results: dict[str, dict[str, float]],
    hyde_results: dict[str, dict[str, float]],
    method: str,
    alpha: float,
    top_k: int,
) -> dict[str, dict[str, float]]:
    if method == "none":
        return hyde_results

    fused = {}
    query_ids = set(base_results) | set(hyde_results)
    for qid in query_ids:
        base_run = base_results.get(qid, {})
        hyde_run = hyde_results.get(qid, {})
        doc_ids = set(base_run) | set(hyde_run)

        if method == "linear":
            base_scores = _minmax_scores(base_run)
            hyde_scores = _minmax_scores(hyde_run)
            scores = {
                doc_id: alpha * base_scores.get(doc_id, 0.0) + (1 - alpha) * hyde_scores.get(doc_id, 0.0)
                for doc_id in doc_ids
            }
        elif method == "rrf":
            base_ranks = {doc_id: rank for rank, doc_id in enumerate(_ranked_docs(base_run), start=1)}
            hyde_ranks = {doc_id: rank for rank, doc_id in enumerate(_ranked_docs(hyde_run), start=1)}
            scores = {
                doc_id: alpha / (rrf_k + base_ranks[doc_id]) if doc_id in base_ranks else 0.0
                for doc_id in doc_ids
            }
            for doc_id in doc_ids:
                if doc_id in hyde_ranks:
                    scores[doc_id] = scores.get(doc_id, 0.0) + (1 - alpha) / (rrf_k + hyde_ranks[doc_id])
        else:
            raise ValueError("HYDE_FUSION must be one of 'rrf', 'linear', or 'none'")

        fused[qid] = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k])
    return fused

url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)

generator = models.HuggingFaceHypothesisGenerator(
    model_name=hyde_generator_model,
    n=int(os.getenv("HYDE_N", "3")),
    max_new_tokens=int(os.getenv("HYDE_MAX_NEW_TOKENS", "256")),
    temperature=float(temperature) if temperature else 0.7,
    top_p=float(top_p) if top_p else 0.95,
    do_sample=os.getenv("HYDE_DO_SAMPLE", "true").lower() == "true",
    device=os.getenv("HYDE_GENERATOR_DEVICE") or None,
)

if splade_backend == "sparse_encoder":
    base_model = models.SparseEncoderSPLADE(
        splade_model_name,
        max_active_dims=int(os.getenv("SPLADE_MAX_ACTIVE_DIMS")) if os.getenv("SPLADE_MAX_ACTIVE_DIMS") else None,
    )
else:
    base_model = models.SPLADE(splade_model_name, revision=splade_revision)

hyde_model = models.HyDE(
    base_model=base_model,
    generator=generator,
    prompt_builder=models.HyDEPromptBuilder(
        dataset=dataset,
        template=os.getenv("HYDE_PROMPT_TEMPLATE") or None,
    ),
    cache_path=os.path.join(
        pathlib.Path(__file__).parent.absolute(),
        "results",
        "hyde_cache",
        f"{dataset}.{hyde_generator_slug}.jsonl",
    ),
    include_original_query=os.getenv("HYDE_INCLUDE_QUERY", "true").lower() == "true",
    hypothesis_encoder=os.getenv("HYDE_HYPOTHESIS_ENCODER", "corpus"),
    max_hypotheses=int(os.getenv("HYDE_MAX_HYPOTHESES")) if os.getenv("HYDE_MAX_HYPOTHESES") else None,
    aggregation=os.getenv("HYDE_AGGREGATION", "mean"),
)

hyde_retriever = EvaluateRetrieval(
    DRES(
        hyde_model,
        batch_size=int(os.getenv("BEIR_BATCH_SIZE", "8")),
        corpus_chunk_size=int(os.getenv("BEIR_CORPUS_CHUNK_SIZE", "1000")),
    ),
    score_function="dot",
)
hyde_results = hyde_retriever.retrieve(corpus, queries)

if fusion == "none":
    retriever = hyde_retriever
    results = hyde_results
else:
    base_retriever = EvaluateRetrieval(
        DRES(
            base_model,
            batch_size=int(os.getenv("BEIR_BATCH_SIZE", "8")),
            corpus_chunk_size=int(os.getenv("BEIR_CORPUS_CHUNK_SIZE", "1000")),
        ),
        score_function="dot",
    )
    base_results = base_retriever.retrieve(corpus, queries)
    retriever = base_retriever
    results = fuse_results(base_results, hyde_results, fusion, fusion_alpha, retriever.top_k)

logging.info(f"Retriever evaluation for k in: {retriever.k_values}")
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")

results_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "results")
os.makedirs(results_dir, exist_ok=True)
run_name = f"{dataset}.splade-hyde-{fusion}"
util.save_runfile(os.path.join(results_dir, f"{run_name}.run.trec"), results)
util.save_results(os.path.join(results_dir, f"{run_name}.json"), ndcg, _map, recall, precision, mrr)
