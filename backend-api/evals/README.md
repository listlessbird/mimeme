# Search Eval Set

`search_eval_set.json` is the frozen ground-truth retrieval eval for the ONNX text-encoder work. Each entry is `{query, expected_image_id, origin}`. Queries were written by an LLM from the image's `annotations` row (ocr_text, caption_text, tags) to mimic what a real user would type, then pruned by hand.

Do not edit entries. Adding, removing, or rewording anything invalidates every baseline number measured against the set.

How it was built:

1. Sampled 150 images from the prod DB, ImgFlip dataset only since the 49 reddit-sourced images post-date the active index build `v20260227-061522` and are not in it. `embed_status = DONE`, stratified 50/50/50 across tall/square/wide aspect buckets, seed 42. Reproduce with `DATABASE_URL=... uv run python scripts/gen_eval_queries.py`.
2. LLM pass over 25-record chunks, 1-2 queries per image. The instructions forbid copying more than 4 consecutive OCR words and overly generic queries.
3. `scripts/assemble_eval_set.py` drops cross-image duplicate queries, OCR-verbatim leaks, and single-word queries. A manual pass then removed ambiguous or unnatural leftovers.

`eval_queries_input.jsonl` is the intermediate LLM input, kept so the set can be rebuilt.

The architecture-rewrite gate runs this frozen set through the public search
interface in both image and hybrid modes:

```sh
uv run python scripts/eval_search.py \
  --base-url http://localhost:8000 \
  --fixture evals/search_eval_set.json
```

The recorded exact-search baseline is recall@10 = 0.968 in both modes. The
gate permits at most a 0.010 absolute drop. Artifact metadata for the rewritten
compute interface records FAISS `1.13.2`, ONNX Runtime `1.27.0`, encoder repo
`listlessbird/siglip2-base-patch16-naflex-text-onnx`, revision
`092dc08370b1a01d69c78067051b124881a95407`, and variant
`text_model_int8.onnx`.
