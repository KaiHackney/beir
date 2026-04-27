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
model_name = os.getenv("BEIR_DENSE_MODEL", "facebook/contriever")
generator_model = os.getenv("HYDE_GENERATOR_MODEL", "gpt-4o-mini")
generator_model_slug = generator_model.replace("/", "_")
results_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "results")
temperature = os.getenv("HYDE_TEMPERATURE")

url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

base_model = models.SentenceBERT(model_name)
generator = models.OpenAIHypothesisGenerator(
    model_name=generator_model,
    n=int(os.getenv("HYDE_N", "5")),
    max_tokens=int(os.getenv("HYDE_MAX_TOKENS", "512")),
    temperature=float(temperature) if temperature else None,
)
hyde_model = models.HyDE(
    base_model=base_model,
    generator=generator,
    dataset=dataset,
    cache_path=os.path.join(results_dir, "hyde_cache", f"{dataset}.{generator_model_slug}.jsonl"),
    include_original_query=os.getenv("HYDE_INCLUDE_QUERY", "true").lower() == "true",
    hypothesis_encoder=os.getenv("HYDE_HYPOTHESIS_ENCODER", "corpus"),
)

retriever = EvaluateRetrieval(DRES(hyde_model, batch_size=16), score_function="cos_sim")
results = retriever.retrieve(corpus, queries)

ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")

os.makedirs(results_dir, exist_ok=True)
util.save_runfile(os.path.join(results_dir, f"{dataset}.hyde.run.trec"), results)
util.save_results(os.path.join(results_dir, f"{dataset}.hyde.json"), ndcg, _map, recall, precision, mrr)
