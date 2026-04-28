from __future__ import annotations

import gzip
import json
import logging
import os
import pathlib
import random

from datasets import Dataset
from sentence_transformers import SparseEncoder, SparseEncoderTrainer, SparseEncoderTrainingArguments
from sentence_transformers.sparse_encoder import losses
from tqdm.autonotebook import tqdm

from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

dataset = "msmarco"
model_name = os.getenv("SPLADE_TRAIN_MODEL", "naver/splade-v3-distilbert")
output_dir = os.getenv(
    "SPLADE_TRAIN_OUTPUT",
    os.path.join(pathlib.Path(__file__).parent.absolute(), "output", f"{model_name.replace('/', '_')}-{dataset}"),
)
max_train_samples = int(os.getenv("SPLADE_MAX_TRAIN_SAMPLES", "50000"))
ce_score_margin = float(os.getenv("SPLADE_CE_SCORE_MARGIN", "3"))
num_negs_per_query = int(os.getenv("SPLADE_NEGS_PER_QUERY", "1"))

url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
data_path = util.download_and_unzip(url, os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets"))
corpus, queries, _ = GenericDataLoader(data_path).load(split="train")

triplets_url = "https://sbert.net/datasets/msmarco-hard-negatives.jsonl.gz"
triplets_path = os.path.join(data_path, "msmarco-hard-negatives.jsonl.gz")
if not os.path.isfile(triplets_path):
    util.download_url(triplets_url, triplets_path)

records = []
with gzip.open(triplets_path, "rt", encoding="utf8") as f_in:
    for line in tqdm(f_in, total=502939, desc="hard negatives"):
        data = json.loads(line)
        pos = data.get("pos", [])
        neg = data.get("neg", {})
        if not pos:
            continue

        pos_item = random.choice(pos)
        pos_score = pos_item["ce-score"]
        negatives = []
        for system_negs in neg.values():
            for item in system_negs:
                if item["ce-score"] <= pos_score - ce_score_margin:
                    negatives.append(item["pid"])
                    if len(negatives) >= num_negs_per_query:
                        break
            if len(negatives) >= num_negs_per_query:
                break

        if not negatives:
            continue

        try:
            record = {
                "anchor": queries[data["qid"]],
                "positive": corpus[pos_item["pid"]]["text"],
            }
            for idx, neg_pid in enumerate(negatives):
                record[f"negative_{idx}"] = corpus[neg_pid]["text"]
        except KeyError:
            continue

        records.append(record)
        if len(records) >= max_train_samples:
            break

logging.info("Prepared %d sparse training records.", len(records))
train_dataset = Dataset.from_list(records)

model = SparseEncoder(model_name)
base_loss = losses.SparseMultipleNegativesRankingLoss(model)
train_loss = losses.SpladeLoss(
    model=model,
    loss=base_loss,
    document_regularizer_weight=float(os.getenv("SPLADE_DOC_REG_WEIGHT", "3e-5")),
    query_regularizer_weight=float(os.getenv("SPLADE_QUERY_REG_WEIGHT", "5e-5")),
)

args = SparseEncoderTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=int(os.getenv("SPLADE_TRAIN_BATCH_SIZE", "8")),
    gradient_accumulation_steps=int(os.getenv("SPLADE_GRAD_ACCUM_STEPS", "1")),
    learning_rate=float(os.getenv("SPLADE_LR", "2e-5")),
    num_train_epochs=float(os.getenv("SPLADE_EPOCHS", "1")),
    warmup_steps=int(os.getenv("SPLADE_WARMUP_STEPS", "1000")),
    logging_steps=int(os.getenv("SPLADE_LOGGING_STEPS", "100")),
    save_steps=int(os.getenv("SPLADE_SAVE_STEPS", "5000")),
    save_total_limit=int(os.getenv("SPLADE_SAVE_TOTAL_LIMIT", "2")),
    fp16=os.getenv("SPLADE_FP16", "false").lower() == "true",
    bf16=os.getenv("SPLADE_BF16", "false").lower() == "true",
    use_mps_device=os.getenv("SPLADE_USE_MPS", "false").lower() == "true",
)

trainer = SparseEncoderTrainer(model=model, args=args, train_dataset=train_dataset, loss=train_loss)
trainer.train()
model.save_pretrained(output_dir)
logging.info("Saved fine-tuned SPLADE model to %s", output_dir)
