# Text Encoder

Export tooling for the `google/siglip2-base-patch16-naflex` text tower. Produces the ONNX model that `backend-api` uses to encode search queries, so the api and worker images can run without torch.

Nothing in this project runs in production. The api downloads the exported artifacts from `listlessbird/siglip2-base-patch16-naflex-text-onnx` on Hugging Face, pinned to a revision in backend settings.

## Setup

```bash
uv sync --all-groups
```

## Commands

```bash
just bench        # torch baseline timings + RSS
just test         # fast tests, no model downloads
just test-model   # parity gates against the real model (downloads it)
just lint
just type
just check
```

Rebuild the export ladder into `artifacts/` (add `--push` to publish to Hugging Face):

```bash
uv run python scripts/export_text_onnx.py
```

Benchmark an exported model (needs only onnxruntime, tokenizers, and numpy, so it also runs on the Pi in a plain `python:3.13-slim` container):

```bash
uv run python scripts/bench_onnx.py artifacts/text_model_int8.onnx
```

Regenerate the torch reference fixture that `backend-api/model_smoke` asserts against (run this after any export change):

```bash
uv run python scripts/dump_torch_reference.py
```

## Exported Variants

- `text_model.onnx`: fp32 reference
- `text_model_fp16.onnx`: half the disk size, same RSS (onnxruntime upcasts to fp32 on CPU)
- `text_model_int8.onnx`: the shipped variant. fp16 token embedding table plus per-channel dynamic int8 on the qkv and fc1 matmuls. out_proj and fc2 stay fp32 because fc2's pre-GELU activations have outliers that collapse quality when quantized.

## Constraints

- Query embeddings must stay in the same space as the index embeddings built on Modal. The tokenizer, `padding="max_length", max_length=64`, and pooling are frozen; `TorchTextEncoder` mirrors the backend implementation and is the parity reference for every variant.
- torch and transformers pins match `backend-api`'s `local-gpu` extra (`torch==2.9.0`, `transformers==4.52.4`).
- Plain `uv run pytest` must never download models. Model-loading tests are opt-in behind `RUN_MODEL_SMOKE=1`.
