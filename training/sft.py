"""SFT rendering and loss helpers shared by training runners."""

from __future__ import annotations

from typing import Any

from tinker import Datum, ForwardBackwardOutput
from transformers import PreTrainedTokenizerBase


def render_tokens(row: dict[str, Any], tokenizer: PreTrainedTokenizerBase) -> list[int]:
    """Render one frozen SFT row exactly as the model sees it during training.

    The rendered dataset stores chat messages, not pre-tokenized IDs. This
    helper applies the Qwen3 chat template with thinking disabled so both
    training scripts use the same prompt/answer formatting.
    """

    rendered = tokenizer.apply_chat_template(
        row["messages"],
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
        return_dict=True,
    )
    return list(rendered["input_ids"])


def render_prompt_tokens(
    row: dict[str, Any], tokenizer: PreTrainedTokenizerBase
) -> list[int]:
    """Render only the prompt side of a row, including the assistant prefix.

    The length returned here is the loss-mask boundary. Tokens before this
    boundary are context that the model reads, while tokens after it are the
    assistant answer tokens we train against.
    """

    rendered = tokenizer.apply_chat_template(
        row["messages"][:-1],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
    )
    return list(rendered["input_ids"])


def render_text(row: dict[str, Any], tokenizer: PreTrainedTokenizerBase) -> str:
    """Return the human-inspectable chat-template render for audit artifacts."""

    return tokenizer.apply_chat_template(
        row["messages"],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )


def build_datum(row: dict[str, Any], tokenizer: PreTrainedTokenizerBase) -> Datum:
    """Build one Tinker datum for supervised fine-tuning.

    A Tinker datum is the backend's training example object: model-input tokens
    plus loss-function inputs. For SFT we pass the full prompt and answer, but
    set loss weights to 0 for prompt tokens and 1 for assistant answer tokens.
    That teaches the model to predict the answer without penalizing it for the
    user/system prompt text it was given as context.
    """

    import torch
    from tinker_cookbook.supervised.common import datum_from_tokens_weights

    tokens = render_tokens(row, tokenizer)
    prompt_len = len(render_prompt_tokens(row, tokenizer))
    if prompt_len >= len(tokens):
        raise ValueError(
            f"prompt length {prompt_len} leaves no assistant tokens for {row['row_id']}"
        )
    weights = torch.zeros(len(tokens), dtype=torch.float32)
    weights[prompt_len:] = 1.0
    return datum_from_tokens_weights(torch.tensor(tokens, dtype=torch.int64), weights)


def build_datums(
    rows: list[dict[str, Any]], tokenizer: PreTrainedTokenizerBase
) -> list[Datum]:
    """Convert rendered JSONL rows into Tinker datums."""

    return [build_datum(row, tokenizer) for row in rows]


def answer_weight_count(datum: Datum) -> float:
    """Count answer tokens that contribute to SFT loss for one datum.

    Tinker cookbook versions may expose weights as a torch-like tensor or as
    backend `TensorData`. This helper keeps that SDK detail out of the runners.
    """

    weights: object = datum.loss_fn_inputs["weights"]
    if hasattr(weights, "sum"):
        return float(weights.sum().item())
    if hasattr(weights, "data"):
        return sum(float(value) for value in weights.data)
    raise TypeError(f"unsupported weight tensor type: {type(weights)!r}")


def datum_token_count(datum: Datum) -> int:
    """Return total model-input tokens, including prompt and answer tokens."""

    return int(datum.model_input.length)


def mean_nll(output: ForwardBackwardOutput, data: list[Datum]) -> float:
    """Compute weighted mean negative log-likelihood for SFT examples.

    NLL is the cross-entropy-style training loss over the tokens whose weights
    are nonzero. Lower NLL means the model assigned higher probability to the
    expected assistant answer tokens on that split.
    """

    from tinker_cookbook.supervised.common import compute_mean_nll

    logprobs = [x["logprobs"] for x in output.loss_fn_outputs]
    weights = [datum.loss_fn_inputs["weights"] for datum in data]
    return float(compute_mean_nll(logprobs, weights))


def aggregate_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    """Add backend metric dictionaries from several Tinker calls."""

    totals: dict[str, float] = {}
    for row in rows:
        for key, value in row.items():
            totals[key] = totals.get(key, 0.0) + float(value)
    return totals
