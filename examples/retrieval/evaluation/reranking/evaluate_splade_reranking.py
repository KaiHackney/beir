from __future__ import annotations

import logging
import os
import pathlib

from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.reranking import Rerank
from beir.reranking.models import CrossEncoder
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.search.sparse import SparseInvertedSearch

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

dataset = os.getenv("BEIR_DATASET", "scifact")
split = os.getenv("BEIR_SPLIT", "test")
splade_model_name = os.getenv("SPLADE_MODEL", "naver/splade-v3")
splade_backend = os.getenv("SPLADE_BACKEND", "sparse_encoder")
search_backend = os.getenv("SPLADE_SEARCH_BACKEND", "indexed")
cross_encoder_name = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
rerank_top_k = int(os.getenv("RERANK_TOP_K", "100"))
results_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "results")
index_name = os.getenv("SPLADE_INDEX_NAME", splade_model_name.replace("/", "_"))
index_dir = os.getenv(
    "SPLADE_INDEX_DIR",
    os.path.join(pathlib.Path(__file__).parent.parent, "sparse", "indexes", f"{dataset}.{index_name}"),
)

url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)

if splade_backend == "sparse_encoder":
    splade_model = models.SparseEncoderSPLADE(
        splade_model_name,
        max_active_dims=int(os.getenv("SPLADE_MAX_ACTIVE_DIMS")) if os.getenv("SPLADE_MAX_ACTIVE_DIMS") else None,
        title_weight=float(os.getenv("SPLADE_TITLE_WEIGHT", "1.0")),
    )
else:
    splade_model = models.SPLADE(
        splade_model_name,
        revision=os.getenv("SPLADE_REVISION", "refs/pr/1"),
    )

if search_backend == "indexed":
    searcher = SparseInvertedSearch(
        splade_model,
        batch_size=int(os.getenv("BEIR_BATCH_SIZE", "8")),
        corpus_chunk_size=int(os.getenv("BEIR_CORPUS_CHUNK_SIZE", "1000")),
        index_dir=index_dir,
        initialize=os.getenv("SPLADE_REBUILD_INDEX", "false").lower() == "true",
        doc_max_active_dims=int(os.getenv("SPLADE_DOC_MAX_ACTIVE_DIMS", "128")),
        query_max_active_dims=int(os.getenv("SPLADE_QUERY_MAX_ACTIVE_DIMS", "128")),
        doc_weight_threshold=float(os.getenv("SPLADE_DOC_WEIGHT_THRESHOLD", "0.0")),
        query_weight_threshold=float(os.getenv("SPLADE_QUERY_WEIGHT_THRESHOLD", "0.0")),
        impact_scale=float(os.getenv("SPLADE_IMPACT_SCALE")) if os.getenv("SPLADE_IMPACT_SCALE") else None,
    )
else:
    searcher = DRES(
        splade_model,
        batch_size=int(os.getenv("BEIR_BATCH_SIZE", "8")),
        corpus_chunk_size=int(os.getenv("BEIR_CORPUS_CHUNK_SIZE", "1000")),
    )

retriever = EvaluateRetrieval(searcher, score_function="dot")
first_stage_results = retriever.retrieve(corpus, queries)

logging.info("First-stage SPLADE evaluation for k in: %s", retriever.k_values)
first_ndcg, first_map, first_recall, first_precision = retriever.evaluate(qrels, first_stage_results, retriever.k_values)
first_mrr = retriever.evaluate_custom(qrels, first_stage_results, retriever.k_values, metric="mrr")

cross_encoder = CrossEncoder(
    cross_encoder_name,
    max_length=int(os.getenv("RERANKER_MAX_LENGTH", "512")),
)
reranker = Rerank(cross_encoder, batch_size=int(os.getenv("RERANK_BATCH_SIZE", "32")))
rerank_results = reranker.rerank(corpus, queries, first_stage_results, top_k=rerank_top_k)

logging.info("Reranked SPLADE evaluation for k in: %s", retriever.k_values)
rerank_ndcg, rerank_map, rerank_recall, rerank_precision = EvaluateRetrieval.evaluate(
    qrels,
    rerank_results,
    retriever.k_values,
)
rerank_mrr = EvaluateRetrieval.evaluate_custom(qrels, rerank_results, retriever.k_values, metric="mrr")

os.makedirs(results_dir, exist_ok=True)
run_name = f"{dataset}.splade-rerank.top{rerank_top_k}"
first_stage_name = f"{dataset}.splade-first-stage"

util.save_runfile(os.path.join(results_dir, f"{first_stage_name}.run.trec"), first_stage_results)
util.save_results(
    os.path.join(results_dir, f"{first_stage_name}.json"),
    first_ndcg,
    first_map,
    first_recall,
    first_precision,
    first_mrr,
)
util.save_runfile(os.path.join(results_dir, f"{run_name}.run.trec"), rerank_results)
util.save_results(
    os.path.join(results_dir, f"{run_name}.json"),
    rerank_ndcg,
    rerank_map,
    rerank_recall,
    rerank_precision,
    rerank_mrr,
)
