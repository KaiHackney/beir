import logging
import os
import pathlib
import random

from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
#### /print debug information to stdout

#### Download NFCorpus dataset and unzip the dataset
dataset = "fiqa"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data path where nfcorpus has been downloaded and unzipped to the data loader
# data folder would contain these files:
# (1) nfcorpus/corpus.jsonl  (format: jsonlines)
# (2) nfcorpus/queries.jsonl (format: jsonlines)
# (3) nfcorpus/qrels/test.tsv (format: tsv ("\t"))

corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

#### SPARSE Retrieval using SPLADE ####
# The SPLADE model provides a weight for each query token and document token
# The final score is taken using a dot-product between the weights of the common tokens.
# To learn more, please refer to the link below:
# https://europe.naverlabs.com/blog/splade-a-sparse-bi-encoder-bert-based-model-achieves-effective-and-efficient-first-stage-ranking/

#################################################
#### 1. Loading SPLADE model from NAVER LABS ####
#################################################
# The safetensors version for this checkpoint lives on Hugging Face's conversion
# PR branch, so we load that revision explicitly to avoid torch.load on .bin weights.
model_path = os.getenv("SPLADE_MODEL", "naver/splade_v2_max")
splade_backend = os.getenv("SPLADE_BACKEND", "masked_lm")
if splade_backend == "sparse_encoder":
    splade = models.SparseEncoderSPLADE(
        model_path,
        max_active_dims=int(os.getenv("SPLADE_MAX_ACTIVE_DIMS")) if os.getenv("SPLADE_MAX_ACTIVE_DIMS") else None,
        title_weight=float(os.getenv("SPLADE_TITLE_WEIGHT", "1.0")),
    )
else:
    splade = models.SPLADE(model_path, revision=os.getenv("SPLADE_REVISION", "refs/pr/1"))

model = DRES(splade, batch_size=int(os.getenv("BEIR_BATCH_SIZE", "16")))
retriever = EvaluateRetrieval(model, score_function="dot")

#### Retrieve dense results (format of results is identical to qrels)
results = retriever.retrieve(corpus, queries)

#### Evaluate your retrieval using NDCG@k, MAP@K ...

logging.info(f"Retriever evaluation for k in: {retriever.k_values}")
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

results_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "results")
os.makedirs(results_dir, exist_ok=True)

util.save_runfile(os.path.join(results_dir, f"{dataset}.splade.run.trec"), results)
util.save_results(os.path.join(results_dir, f"{dataset}.splade.json"), ndcg, _map, recall, precision)

#### Print top-k documents retrieved ####
top_k = 10

query_id, ranking_scores = random.choice(list(results.items()))
scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
logging.info(f"Query : {queries[query_id]}\n")

for rank in range(top_k):
    doc_id = scores_sorted[rank][0]
    # Format: Rank x: ID [Title] Body
    logging.info(f"Rank {rank + 1}: {doc_id} [{corpus[doc_id].get('title')}] - {corpus[doc_id].get('text')}\n")
