from __future__ import annotations

"""Run a SPLADE ablation across HyDE and reranking.

Produces up to four runs:
- splade
- splade_hyde
- splade_rerank
- splade_hyde_rerank
"""

import csv
import json
import logging
import os
import pathlib

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

    generator = models.HuggingFaceHypothesisGenerator(
        model_name=generator_model,
        n=int(os.getenv("HYDE_N", "1")),
        max_new_tokens=int(os.getenv("HYDE_MAX_NEW_TOKENS", "48")),
        temperature=float(temperature) if temperature else 0.7,
        top_p=float(top_p) if top_p else 0.95,
        do_sample=os.getenv("HYDE_DO_SAMPLE", "false").lower() == "true",
        device=os.getenv("HYDE_GENERATOR_DEVICE") or None,
    )

    return models.HyDE(
        base_model=base_model,
        generator=generator,
        prompt_builder=models.HyDEPromptBuilder(
            dataset=dataset,
            template=os.getenv("HYDE_PROMPT_TEMPLATE") or None,
        ),
        cache_path=os.path.join(results_dir, "hyde_cache", f"{dataset}.{generator_slug}.jsonl"),
        include_original_query=os.getenv("HYDE_INCLUDE_QUERY", "true").lower() == "true",
        hypothesis_encoder=os.getenv("HYDE_HYPOTHESIS_ENCODER", "corpus"),
        max_hypotheses=int(os.getenv("HYDE_MAX_HYPOTHESES")) if os.getenv("HYDE_MAX_HYPOTHESES") else None,
        aggregation=os.getenv("HYDE_AGGREGATION", "max"),
    )


dataset = os.getenv("BEIR_DATASET", "scifact")
split = os.getenv("BEIR_SPLIT", "test")
variants = parse_list("SPLADE_ABLATION_VARIANTS", "splade,splade_hyde,splade_rerank,splade_hyde_rerank")
top_k = int(os.getenv("ABLATION_TOP_K", "1000"))
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
    runs["splade_hyde"] = hyde_searcher.search(corpus, queries, top_k, score_function="dot")

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
        runs["splade_rerank"] = reranked
        summaries.append(evaluate_and_save(f"{dataset}.splade_rerank", reranked, qrels, k_values, results_dir))

    if "splade_hyde_rerank" in variants:
        if "splade_hyde" not in runs:
            raise ValueError("splade_hyde_rerank requires splade_hyde retrieval.")
        logger.info("Reranking SPLADE+HyDE results...")
        reranked = reranker.rerank(corpus, queries, runs["splade_hyde"], top_k=rerank_top_k)
        runs["splade_hyde_rerank"] = reranked
        summaries.append(evaluate_and_save(f"{dataset}.splade_hyde_rerank", reranked, qrels, k_values, results_dir))

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
