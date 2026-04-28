from __future__ import annotations

import logging
import os
import pathlib

from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.sparse import SparseInvertedSearch

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

dataset = os.getenv("BEIR_DATASET", "scifact")
split = os.getenv("BEIR_SPLIT", "test")
model_name = os.getenv("SPLADE_MODEL", "naver/splade-v3")
index_name = os.getenv("SPLADE_INDEX_NAME", model_name.replace("/", "_"))
results_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "results")
index_dir = os.getenv(
    "SPLADE_INDEX_DIR",
    os.path.join(pathlib.Path(__file__).parent.absolute(), "indexes", f"{dataset}.{index_name}"),
)

url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)

model = models.SparseEncoderSPLADE(
    model_name,
    max_active_dims=int(os.getenv("SPLADE_MAX_ACTIVE_DIMS")) if os.getenv("SPLADE_MAX_ACTIVE_DIMS") else None,
    title_weight=float(os.getenv("SPLADE_TITLE_WEIGHT", "1.0")),
)

searcher = SparseInvertedSearch(
    model,
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
retriever = EvaluateRetrieval(searcher, score_function="dot")
results = retriever.retrieve(corpus, queries)

logging.info(f"Retriever evaluation for k in: {retriever.k_values}")
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")

os.makedirs(results_dir, exist_ok=True)
run_name = f"{dataset}.splade-indexed"
util.save_runfile(os.path.join(results_dir, f"{run_name}.run.trec"), results)
util.save_results(os.path.join(results_dir, f"{run_name}.json"), ndcg, _map, recall, precision, mrr)
