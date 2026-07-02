from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import onnx
import onnxruntime as ort
import torch
import transformers
from onnxruntime.quantization.shape_inference import quant_pre_process
from onnxruntime.transformers.float16 import convert_float_to_float16
from text_encoder.metrics import cosine_diagonal
from text_encoder.quantize import build_int8_hybrid
from text_encoder.queries import BENCH_QUERIES
from text_encoder.torch_encoder import EMBED_DIM, MAX_LENGTH, MODEL_ID, TorchTextEncoder

FP32_MIN_COSINE = 0.999
QUANT_MIN_COSINE = 0.99
OPSET = 17


class TextTower(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model: Any = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_text_features(input_ids=input_ids)


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def export_fp32(encoder: TorchTextEncoder, path: Path) -> None:
    tower = TextTower(encoder._model).eval()
    sample = torch.zeros((1, MAX_LENGTH), dtype=torch.int64)
    torch.onnx.export(
        tower,
        (sample,),
        str(path),
        input_names=["input_ids"],
        output_names=["text_embeds"],
        dynamic_axes={"input_ids": {0: "batch"}, "text_embeds": {0: "batch"}},
        opset_version=OPSET,
        dynamo=False,
    )
    onnx.checker.check_model(str(path), full_check=False)


def onnx_embeddings(path: Path, input_ids: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outputs = [
        cast(np.ndarray, session.run(["text_embeds"], {"input_ids": row[None, :]})[0])[0]
        for row in input_ids
    ]
    return np.stack(outputs).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--repo-id", default="listlessbird/siglip2-base-patch16-naflex-text-onnx")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    encoder = TorchTextEncoder(model_id=args.model_id, device="cpu")
    source_revision = getattr(encoder._model.config, "_commit_hash", None) or "unknown"
    tokenizer = encoder._processor.tokenizer

    input_ids = np.stack(
        [encoder.tokenize(q)["input_ids"].numpy()[0] for q in BENCH_QUERIES]
    ).astype(np.int64)
    reference = np.stack([encoder.encode(q) for q in BENCH_QUERIES])
    assert reference.shape == (len(BENCH_QUERIES), EMBED_DIM)

    fp32_path = out / "text_model.onnx"
    print(f"exporting fp32 → {fp32_path}")
    started = time.monotonic()
    export_fp32(encoder, fp32_path)
    print(f"fp32 export done in {time.monotonic() - started:.0f}s")

    fp32_cosines = cosine_diagonal(reference, onnx_embeddings(fp32_path, input_ids))
    print(f"fp32 parity: min cosine {fp32_cosines.min():.6f} (gate ≥ {FP32_MIN_COSINE})")
    if fp32_cosines.min() < FP32_MIN_COSINE:
        worst = BENCH_QUERIES[int(fp32_cosines.argmin())]
        sys.exit(f"FAIL: fp32 export parity below {FP32_MIN_COSINE} (worst query: {worst!r})")

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        preprocessed = Path(tmp.name)
    quant_pre_process(str(fp32_path), str(preprocessed), skip_symbolic_shape=False)

    fp16_path = out / "text_model_fp16.onnx"
    print(f"converting fp16 → {fp16_path}")
    fp16_model = convert_float_to_float16(onnx.load(str(preprocessed)), keep_io_types=True)
    onnx.save_model(fp16_model, str(fp16_path))

    int8_path = out / "text_model_int8.onnx"
    print(f"quantizing int8 (dynamic int8 qkv/fc1 + fp16 embedding table) → {int8_path}")
    build_int8_hybrid(preprocessed, int8_path)
    preprocessed.unlink()

    parity = {"text_model.onnx": float(fp32_cosines.min())}
    for path in (fp16_path, int8_path):
        embeddings = onnx_embeddings(path, input_ids)
        assert np.isfinite(embeddings).all(), f"{path.name}: non-finite embeddings"
        cosines = cosine_diagonal(reference, embeddings)
        parity[path.name] = float(cosines.min())
        gate = "PASS" if cosines.min() >= QUANT_MIN_COSINE else "FAIL"
        print(f"{path.name}: min cosine {cosines.min():.6f} [{gate} vs ≥ {QUANT_MIN_COSINE}]")

    tok_dir = out / "tokenizer"
    tokenizer.save_pretrained(str(tok_dir))
    for tok_file in tok_dir.iterdir():
        tok_file.replace(out / tok_file.name)
    tok_dir.rmdir()

    meta = {
        "source_model": args.model_id,
        "source_revision": source_revision,
        "max_length": MAX_LENGTH,
        "embed_dim": EMBED_DIM,
        "opset": OPSET,
        "inputs": ["input_ids"],
        "outputs": ["text_embeds"],
        "pad_token_id": tokenizer.pad_token_id,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "onnxruntime": ort.__version__,
        "onnx": onnx.__version__,
        "export_git_commit": git_commit(),
        "int8_recipe": "per-channel dynamic int8 qkv/fc1 matmuls; out_proj/fc2 fp32; fp16 embedding table",
        "min_cosine_vs_torch": parity,
    }
    (out / "export_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print()
    for name, min_cosine in parity.items():
        size_mb = (out / name).stat().st_size / (1024 * 1024)
        print(f"{name:<28} {size_mb:>8.1f} MB  min_cosine={min_cosine:.6f}")

    if args.push:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.repo_id, private=False, exist_ok=True)
        api.upload_folder(
            folder_path=str(out),
            repo_id=args.repo_id,
            commit_message=f"export {args.model_id}@{source_revision} opset{OPSET}",
        )
        revision = api.list_repo_commits(args.repo_id)[0].commit_id
        print(f"pushed to {args.repo_id} revision {revision}")
    else:
        print("\nskipped push (rerun with --push to publish to HF)")


if __name__ == "__main__":
    main()
