from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from tqdm.autonotebook import tqdm

logger = logging.getLogger(__name__)


HYDE_PROMPTS = {
    "arguana": "Please write a counter argument for the passage.\nPassage: {query}\nCounter Argument:",
    "dbpedia-entity": "Please write a passage to answer the question.\nQuestion: {query}\nPassage:",
    "fiqa": "Please write a financial article passage to answer the question.\nQuestion: {query}\nPassage:",
    "scifact": "Please write a scientific paper passage to support/refute the claim.\nClaim: {query}\nPassage:",
    "trec-covid": "Please write a scientific paper passage to answer the question.\nQuestion: {query}\nPassage:",
    "trec-news": "Please write a news passage about the topic.\nTopic: {query}\nPassage:",
    "web-search": "Please write a passage to answer the question.\nQuestion: {query}\nPassage:",
}

DEFAULT_HYDE_PROMPT = "Please write a passage to answer the question.\nQuestion: {query}\nPassage:"


class HypothesisGenerator(Protocol):
    def generate(self, prompt: str) -> list[str] | str:
        pass


class HyDEPromptBuilder:
    """Builds dataset-aware prompts for hypothetical document generation."""

    def __init__(self, dataset: str | None = None, template: str | None = None):
        self.dataset = dataset
        self.template = template

    def build_prompt(self, query: str) -> str:
        template = self.template
        if template is None and self.dataset:
            template = HYDE_PROMPTS.get(self.dataset)
        template = template or DEFAULT_HYDE_PROMPT
        return template.format(query=query)


class OpenAIHypothesisGenerator:
    """OpenAI-backed hypothetical document generator.

    The OpenAI package is imported lazily so BEIR keeps working without an
    OpenAI dependency unless this generator is used.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        n: int = 5,
        max_tokens: int = 512,
        temperature: float | None = None,
        top_p: float | None = None,
        system_prompt: str | None = None,
        api: str = "responses",
    ):
        if api not in {"responses", "chat_completions"}:
            raise ValueError("api must be either 'responses' or 'chat_completions'")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install the OpenAI client with `pip install openai` to use HyDE generation.") from exc

        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"), base_url=base_url)
        self.model_name = model_name
        self.n = n
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = system_prompt
        self.api = api

    def generate(self, prompt: str) -> list[str]:
        if self.api == "responses":
            return self._generate_responses(prompt)
        return self._generate_chat_completions(prompt)

    def _generate_responses(self, prompt: str) -> list[str]:
        texts = []
        for _ in range(self.n):
            params = {
                "model": self.model_name,
                "input": prompt,
                "max_output_tokens": max(self.max_tokens, 16),
            }
            if self.system_prompt:
                params["instructions"] = self.system_prompt
            if self.temperature is not None:
                params["temperature"] = self.temperature
            if self.top_p is not None:
                params["top_p"] = self.top_p

            response = self.client.responses.create(**params)
            text = response.output_text.strip()
            if text:
                texts.append(text)
        return texts

    def _generate_chat_completions(self, prompt: str) -> list[str]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        params = {
            "model": self.model_name,
            "messages": messages,
            "n": self.n,
            "max_completion_tokens": self.max_tokens,
        }
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p

        response = self.client.chat.completions.create(**params)
        return [choice.message.content.strip() for choice in response.choices if choice.message.content]


class HuggingFaceHypothesisGenerator:
    """Local Hugging Face generator for HyDE hypotheses.

    Supports encoder-decoder models such as FLAN-T5 and causal language models
    such as Mistral-style instruction models.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        n: int = 3,
        max_new_tokens: int = 256,
        temperature: float | None = 0.7,
        top_p: float | None = 0.95,
        do_sample: bool = True,
        device: str | None = None,
        model_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        generation_kwargs: dict | None = None,
    ):
        try:
            from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install transformers and torch to use local Hugging Face HyDE generation.") from exc

        self.model_name = model_name
        self.n = n
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.generation_kwargs = generation_kwargs or {}

        tokenizer_kwargs = tokenizer_kwargs or {}
        model_kwargs = model_kwargs or {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        config = AutoConfig.from_pretrained(model_name)
        self.is_encoder_decoder = bool(getattr(config, "is_encoder_decoder", False))
        model_cls = AutoModelForSeq2SeqLM if self.is_encoder_decoder else AutoModelForCausalLM
        self.model = model_cls.from_pretrained(model_name, **model_kwargs)

        self.device = device or self._default_device()
        logger.info(f"Use pytorch device for HyDE generation: {self.device}")
        self.model = self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @staticmethod
    def _default_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def generate(self, prompt: str) -> list[str]:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        params = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "num_return_sequences": self.n,
            **self.generation_kwargs,
        }
        if self.temperature is not None and self.do_sample:
            params["temperature"] = self.temperature
        if self.top_p is not None and self.do_sample:
            params["top_p"] = self.top_p

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **params)

        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        if not self.is_encoder_decoder:
            decoded = [text[len(prompt) :].strip() if text.startswith(prompt) else text.strip() for text in decoded]
        return [text.strip() for text in decoded if text and text.strip()]


class JsonlHyDECache:
    """Small append-only cache keyed by the raw query text."""

    def __init__(self, path: str | os.PathLike | None = None):
        self.path = Path(path) if path else None
        self.cache: dict[str, list[str]] = {}
        if self.path and self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    self.cache[record["query"]] = record["hypotheses"]

    def get(self, query: str) -> list[str] | None:
        return self.cache.get(query)

    def set(self, query: str, prompt: str, hypotheses: list[str]) -> None:
        self.cache[query] = hypotheses
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"query": query, "prompt": prompt, "hypotheses": hypotheses}) + "\n")


class HyDE:
    """Hypothetical Document Embeddings wrapper for BEIR dense retrieval.

    Wrap any BEIR dense model that implements `encode_queries()` and
    `encode_corpus()`, then pass this wrapper to `DenseRetrievalExactSearch`.
    """

    def __init__(
        self,
        base_model,
        generator: HypothesisGenerator | Callable[[str], list[str] | str] | None = None,
        prompt_builder: HyDEPromptBuilder | None = None,
        dataset: str | None = None,
        cache_path: str | os.PathLike | None = None,
        include_original_query: bool = True,
        hypothesis_encoder: str = "corpus",
        max_hypotheses: int | None = None,
        aggregation: str = "mean",
    ):
        if hypothesis_encoder not in {"corpus", "query"}:
            raise ValueError("hypothesis_encoder must be either 'corpus' or 'query'")
        if aggregation not in {"mean", "sum", "max"}:
            raise ValueError("aggregation must be one of 'mean', 'sum', or 'max'")

        self.base_model = base_model
        self.generator = generator
        self.prompt_builder = prompt_builder or HyDEPromptBuilder(dataset=dataset)
        self.cache = JsonlHyDECache(cache_path)
        self.include_original_query = include_original_query
        self.hypothesis_encoder = hypothesis_encoder
        self.max_hypotheses = max_hypotheses
        self.aggregation = aggregation

    def encode_corpus(self, corpus, batch_size: int = 8, **kwargs):
        return self.base_model.encode_corpus(corpus, batch_size=batch_size, **kwargs)

    def encode_queries(self, queries: list[str], batch_size: int = 16, **kwargs):
        embeddings = []
        show_progress_bar = kwargs.get("show_progress_bar", True)
        inner_kwargs = {**kwargs, "show_progress_bar": False}
        iterator = tqdm(queries, desc="HyDE queries", disable=not show_progress_bar)

        for query in iterator:
            hypotheses = self._get_hypotheses(query)
            query_embeddings = []

            if self.include_original_query:
                query_embeddings.append(
                    self._to_tensor_matrix(
                        self.base_model.encode_queries([query], batch_size=1, **inner_kwargs)
                    )[0]
                )

            if hypotheses:
                if self.hypothesis_encoder == "corpus":
                    hypothesis_embeddings = self.base_model.encode_corpus(
                        hypotheses, batch_size=batch_size, **inner_kwargs
                    )
                else:
                    hypothesis_embeddings = self.base_model.encode_queries(
                        hypotheses, batch_size=batch_size, **inner_kwargs
                    )
                query_embeddings.extend(self._to_tensor_matrix(hypothesis_embeddings))

            if not query_embeddings:
                raise ValueError("HyDE could not create any embeddings for a query.")

            embeddings.append(self._aggregate_embeddings(torch.stack(query_embeddings)))

        embeddings = torch.stack(embeddings)
        if kwargs.get("convert_to_tensor", False):
            return embeddings
        return embeddings.detach().cpu().numpy()

    def _get_hypotheses(self, query: str) -> list[str]:
        cached = self.cache.get(query)
        if cached is not None:
            return cached

        if self.generator is None:
            raise ValueError(
                "No HyDE hypotheses found in cache and no generator was provided. "
                "Pass a generator or pre-populate the cache_path JSONL file."
            )

        prompt = self.prompt_builder.build_prompt(query)
        if hasattr(self.generator, "generate"):
            raw_hypotheses = self.generator.generate(prompt)
        else:
            raw_hypotheses = self.generator(prompt)
        hypotheses = self._clean_hypotheses(raw_hypotheses)
        self.cache.set(query, prompt, hypotheses)
        return hypotheses

    def _clean_hypotheses(self, hypotheses: list[str] | str) -> list[str]:
        if isinstance(hypotheses, str):
            hypotheses = [hypotheses]
        cleaned = [hypothesis.strip() for hypothesis in hypotheses if hypothesis and hypothesis.strip()]
        if self.max_hypotheses is not None:
            cleaned = cleaned[: self.max_hypotheses]
        return cleaned

    def _aggregate_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.aggregation == "mean":
            return embeddings.mean(dim=0)
        if self.aggregation == "sum":
            return embeddings.sum(dim=0)
        return embeddings.max(dim=0).values

    @staticmethod
    def _to_tensor_matrix(embeddings) -> torch.Tensor:
        if isinstance(embeddings, torch.Tensor):
            if len(embeddings.shape) == 1:
                embeddings = embeddings.unsqueeze(0)
            return embeddings

        if isinstance(embeddings, np.ndarray):
            return torch.from_numpy(np.atleast_2d(embeddings))

        rows = []
        for embedding in embeddings:
            if isinstance(embedding, torch.Tensor):
                rows.append(embedding)
            else:
                rows.append(torch.tensor(embedding))
        return torch.stack(rows)
