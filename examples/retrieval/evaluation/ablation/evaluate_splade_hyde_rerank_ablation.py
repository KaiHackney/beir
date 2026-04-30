from __future__ import annotations

"""Run a SPLADE ablation across HyDE and reranking.

Produces up to four runs:
- splade
- splade_hyde
- splade_rerank
- splade_hyde_rerank
"""

import csv
import hashlib
import json
import logging
import os
import pathlib
from collections import defaultdict

from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.reranking import Rerank
from beir.reranking.models import CrossEncoder
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.sparse import SparseInvertedSearch

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
logger = logging.getLogger(__name__)


def parse_list(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


def copy_run(results: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {qid: dict(scores) for qid, scores in results.items()}


def ranked_docs(run: dict[str, float]) -> list[str]:
    return [doc_id for doc_id, _ in sorted(run.items(), key=lambda item: item[1], reverse=True)]


def rrf_fuse(
    runs: list[tuple[float, dict[str, dict[str, float]]]],
    top_k: int,
    rrf_k: int = 60,
) -> dict[str, dict[str, float]]:
    fused = {}
    query_ids = set()
    for _, run in runs:
        query_ids.update(run.keys())

    for qid in query_ids:
        scores = defaultdict(float)
        for weight, run in runs:
            for rank, doc_id in enumerate(ranked_docs(run.get(qid, {})), start=1):
                scores[doc_id] += weight / (rrf_k + rank)
        fused[qid] = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k])
    return fused


def minmax_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    low = min(values)
    high = max(values)
    if high == low:
        return {doc_id: 1.0 for doc_id in scores}
    return {doc_id: (score - low) / (high - low) for doc_id, score in scores.items()}


def fuse_rerank_with_first_stage(
    first_stage: dict[str, dict[str, float]],
    reranked: dict[str, dict[str, float]],
    alpha: float,
    top_k: int,
) -> dict[str, dict[str, float]]:
    fused = {}
    for qid, original_scores in first_stage.items():
        rerank_scores = reranked.get(qid, {})
        original_norm = minmax_scores(original_scores)
        rerank_norm = minmax_scores(rerank_scores)
        scores = {}

        for doc_id in original_scores:
            if doc_id in rerank_norm:
                scores[doc_id] = alpha * rerank_norm[doc_id] + (1.0 - alpha) * original_norm.get(doc_id, 0.0)
            else:
                scores[doc_id] = (1.0 - alpha) * original_norm.get(doc_id, 0.0)

        fused[qid] = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k])
    return fused


def evaluate_and_save(
    name: str,
    results: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
    k_values: list[int],
    output_dir: str,
) -> dict[str, object]:
    run_for_eval = copy_run(results)
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, run_for_eval, k_values)
    mrr = EvaluateRetrieval.evaluate_custom(qrels, run_for_eval, k_values, metric="mrr")

    util.save_runfile(os.path.join(output_dir, f"{name}.run.trec"), results)
    util.save_results(os.path.join(output_dir, f"{name}.json"), ndcg, _map, recall, precision, mrr)

    return {
        "name": name,
        "NDCG@10": ndcg.get("NDCG@10"),
        "Recall@100": recall.get("Recall@100"),
        "Recall@1000": recall.get("Recall@1000"),
        "MRR@10": mrr.get("MRR@10") if isinstance(mrr, dict) else None,
        "ndcg": ndcg,
        "map": _map,
        "recall": recall,
        "precision": precision,
        "mrr": mrr,
    }


def build_splade_model():
    return models.SparseEncoderSPLADE(
        os.getenv("SPLADE_MODEL", "naver/splade-v3"),
        max_active_dims=int(os.getenv("SPLADE_MAX_ACTIVE_DIMS")) if os.getenv("SPLADE_MAX_ACTIVE_DIMS") else None,
        title_weight=float(os.getenv("SPLADE_TITLE_WEIGHT", "1.0")),
    )


def build_sparse_searcher(model, index_dir: str, initialize: bool):
    return SparseInvertedSearch(
        model,
        batch_size=int(os.getenv("BEIR_BATCH_SIZE", "8")),
        corpus_chunk_size=int(os.getenv("BEIR_CORPUS_CHUNK_SIZE", "1000")),
        index_dir=index_dir,
        initialize=initialize,
        doc_max_active_dims=int(os.getenv("SPLADE_DOC_MAX_ACTIVE_DIMS", "128")),
        query_max_active_dims=int(os.getenv("SPLADE_QUERY_MAX_ACTIVE_DIMS", "128")),
        doc_weight_threshold=float(os.getenv("SPLADE_DOC_WEIGHT_THRESHOLD", "0.0")),
        query_weight_threshold=float(os.getenv("SPLADE_QUERY_WEIGHT_THRESHOLD", "0.0")),
        impact_scale=float(os.getenv("SPLADE_IMPACT_SCALE")) if os.getenv("SPLADE_IMPACT_SCALE") else None,
    )


def build_hyde_model(base_model, dataset: str, results_dir: str):
    generator_model = os.getenv("HYDE_GENERATOR_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    generator_slug = generator_model.replace("/", "_")
    temperature = os.getenv("HYDE_TEMPERATURE")
    top_p = os.getenv("HYDE_TOP_P")
    hyde_n = int(os.getenv("HYDE_N", "1"))
    max_new_tokens = int(os.getenv("HYDE_MAX_NEW_TOKENS", "48"))
    do_sample = os.getenv("HYDE_DO_SAMPLE", "false").lower() == "true"
    prompt_template = os.getenv("HYDE_PROMPT_TEMPLATE") or ""
    prompt_hash = hashlib.sha1(prompt_template.encode("utf-8")).hexdigest()[:8]
    cache_name = os.getenv(
        "HYDE_CACHE_NAME",
        f"{dataset}.{generator_slug}.n{hyde_n}.tok{max_new_tokens}.sample{int(do_sample)}.{prompt_hash}.jsonl",
    )

    generator = models.HuggingFaceHypothesisGenerator(
        model_name=generator_model,
        n=hyde_n,
        max_new_tokens=max_new_tokens,
        temperature=float(temperature) if temperature else 0.7,
        top_p=float(top_p) if top_p else 0.95,
        do_sample=do_sample,
        device=os.getenv("HYDE_GENERATOR_DEVICE") or None,
    )

    return models.HyDE(
        base_model=base_model,
        generator=generator,
        prompt_builder=models.HyDEPromptBuilder(
            dataset=dataset,
            template=prompt_template or None,
        ),
        cache_path=os.path.join(results_dir, "hyde_cache", cache_name),
        include_original_query=os.getenv("HYDE_INCLUDE_QUERY", "true").lower() == "true",
        hypothesis_encoder=os.getenv("HYDE_HYPOTHESIS_ENCODER", "corpus"),
        max_hypotheses=int(os.getenv("HYDE_MAX_HYPOTHESES")) if os.getenv("HYDE_MAX_HYPOTHESES") else None,
        aggregation=os.getenv("HYDE_AGGREGATION", "max"),
    )


dataset = os.getenv("BEIR_DATASET", "scifact")
split = os.getenv("BEIR_SPLIT", "test")
variants = parse_list("SPLADE_ABLATION_VARIANTS", "splade,splade_hyde,splade_rerank,splade_hyde_rerank")
top_k = int(os.getenv("ABLATION_TOP_K", "1000"))
hyde_fusion = os.getenv("HYDE_FUSION", "none").lower()
hyde_fusion_weight = float(os.getenv("HYDE_FUSION_WEIGHT", "0.5"))
hyde_rrf_k = int(os.getenv("HYDE_RRF_K", "60"))
rerank_score_alpha = float(os.getenv("RERANK_SCORE_ALPHA", "1.0"))
k_values = [1, 3, 5, 10, 100, 1000]

script_dir = pathlib.Path(__file__).parent.absolute()
results_dir = os.path.join(script_dir, "results", dataset, "splade_hyde_rerank")
os.makedirs(results_dir, exist_ok=True)

url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
data_path = util.download_and_unzip(url, os.path.join(script_dir, "datasets"))
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)

splade_model_name = os.getenv("SPLADE_MODEL", "naver/splade-v3")
index_name = os.getenv("SPLADE_INDEX_NAME", splade_model_name.replace("/", "_"))
index_dir = os.getenv("SPLADE_INDEX_DIR", os.path.join(script_dir, "indexes", f"{dataset}.{index_name}"))

splade_model = build_splade_model()
summaries: list[dict[str, object]] = []
runs: dict[str, dict[str, dict[str, float]]] = {}

logger.info("Running base SPLADE retrieval...")
splade_searcher = build_sparse_searcher(
    splade_model,
    index_dir=index_dir,
    initialize=os.getenv("SPLADE_REBUILD_INDEX", "false").lower() == "true",
)
runs["splade"] = splade_searcher.search(corpus, queries, top_k, score_function="dot")

if "splade" in variants:
    summaries.append(evaluate_and_save(f"{dataset}.splade", runs["splade"], qrels, k_values, results_dir))

if "splade_hyde" in variants or "splade_hyde_rerank" in variants:
    logger.info("Running SPLADE with HyDE query expansion...")
    hyde_model = build_hyde_model(splade_model, dataset, results_dir)
    hyde_searcher = build_sparse_searcher(hyde_model, index_dir=index_dir, initialize=False)
    runs["splade_hyde_raw"] = hyde_searcher.search(corpus, queries, top_k, score_function="dot")
    if hyde_fusion == "rrf":
        runs["splade_hyde"] = rrf_fuse(
            [(1.0 - hyde_fusion_weight, runs["splade"]), (hyde_fusion_weight, runs["splade_hyde_raw"])],
            top_k=top_k,
            rrf_k=hyde_rrf_k,
        )
    elif hyde_fusion == "none":
        runs["splade_hyde"] = runs["splade_hyde_raw"]
    else:
        raise ValueError("HYDE_FUSION must be either 'none' or 'rrf'.")

    if "splade_hyde" in variants:
        summaries.append(
            evaluate_and_save(f"{dataset}.splade_hyde", runs["splade_hyde"], qrels, k_values, results_dir)
        )

if "splade_rerank" in variants or "splade_hyde_rerank" in variants:
    logger.info("Loading reranker...")
    cross_encoder = CrossEncoder(
        os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        max_length=int(os.getenv("RERANKER_MAX_LENGTH", "512")),
    )
    reranker = Rerank(cross_encoder, batch_size=int(os.getenv("RERANK_BATCH_SIZE", "32")))
    rerank_top_k = int(os.getenv("RERANK_TOP_K", "100"))

    if "splade_rerank" in variants:
        logger.info("Reranking base SPLADE results...")
        reranked = reranker.rerank(corpus, queries, runs["splade"], top_k=rerank_top_k)
        runs["splade_rerank"] = fuse_rerank_with_first_stage(
            runs["splade"],
            reranked,
            alpha=rerank_score_alpha,
            top_k=top_k,
        )
        summaries.append(evaluate_and_save(f"{dataset}.splade_rerank", runs["splade_rerank"], qrels, k_values, results_dir))

    if "splade_hyde_rerank" in variants:
        if "splade_hyde" not in runs:
            raise ValueError("splade_hyde_rerank requires splade_hyde retrieval.")
        logger.info("Reranking SPLADE+HyDE results...")
        reranked = reranker.rerank(corpus, queries, runs["splade_hyde"], top_k=rerank_top_k)
        runs["splade_hyde_rerank"] = fuse_rerank_with_first_stage(
            runs["splade_hyde"],
            reranked,
            alpha=rerank_score_alpha,
            top_k=top_k,
        )
        summaries.append(
            evaluate_and_save(f"{dataset}.splade_hyde_rerank", runs["splade_hyde_rerank"], qrels, k_values, results_dir)
        )

summary_path = os.path.join(results_dir, "summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summaries, f, indent=2)

summary_csv_path = os.path.join(results_dir, "summary.csv")
with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["name", "NDCG@10", "Recall@100", "Recall@1000", "MRR@10"])
    writer.writeheader()
    for row in summaries:
        writer.writerow({key: row.get(key) for key in writer.fieldnames})

logger.info("Saved ablation summary to %s and %s", summary_path, summary_csv_path)
for row in summaries:
    logger.info(
        "%s | NDCG@10=%s | Recall@100=%s | Recall@1000=%s | MRR@10=%s",
        row["name"],
        row["NDCG@10"],
        row["Recall@100"],
        row["Recall@1000"],
        row["MRR@10"],
    )
