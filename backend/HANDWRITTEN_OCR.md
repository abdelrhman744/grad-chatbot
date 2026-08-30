# Handwritten OCR (Arabic + English)

Free, local, offline-capable handwriting recognition — no paid or external
OCR API of any kind (no OpenAI, Gemini, Google Cloud Vision, Azure OCR, AWS
Textract, etc.). Runs entirely inside the existing backend process using
Hugging Face `transformers`.

This is a **separate feature from the existing printed-text OCR**
(`services/ocr_service.py`, Tesseract + OpenCV), which continues to power
scanned-PDF/image ingestion in the upload pipeline unchanged. Handwriting
recognition needs a different model family (TrOCR, a ViT-encoder /
RoBERTa-decoder transformer) than printed text, so it's implemented as its
own service and its own endpoint instead of being bolted onto Tesseract.

```
                 OCR System
                     │
          ┌──────────┴──────────┐
          │                     │
      Printed OCR          Handwritten OCR
   (services/ocr_service.py) (services/handwritten_ocr_service.py)
      Tesseract + OpenCV        Hugging Face TrOCR
   used automatically by            │
   the upload pipeline      ┌───────┴───────┐
   for scanned PDFs/images  │               │
                          Arabic          English
                       RayR1/trocr-     microsoft/trocr-
                       base-arabic-     base-handwritten
                       handwritten
                          │               │
                          └───────┬───────┘
                                  │
                           POST /api/ocr/handwritten
                                  │
                            Extracted text
                                  │ (optional, opt-in — see below)
                                  ▼
                     Existing chunk → embed → Qdrant
                     pipeline (services/rag_service.py),
                     same document isolation as any upload
```

## Models

| Language | Model | Loaded via |
|---|---|---|
| English | [`microsoft/trocr-small-handwritten`](https://huggingface.co/microsoft/trocr-small-handwritten) | `TrOCRProcessor` + `VisionEncoderDecoderModel` |
| Arabic | [`RayR1/trocr-base-arabic-handwritten`](https://huggingface.co/RayR1/trocr-base-arabic-handwritten) | `TrOCRProcessor` + `VisionEncoderDecoderModel` |

**English default changed from `trocr-base-handwritten` to `trocr-small-handwritten`** after a real-handwriting evaluation (`scripts/evaluate_handwritten_ocr.py`, IAM handwriting-line dataset): once the line-segmentation aspect-ratio bug below was fixed, both models scored statistically indistinguishable CER (0.253 small vs. 0.248 base over a 6-sample benchmark) but the small checkpoint ran 3-6x faster per line on CPU with a much smaller footprint (~62M vs ~334M params) — a clear win for the "weak university servers, no GPU" target environment with no measured accuracy cost. No lighter realistic Arabic alternative was found (see Limitations).

Both are configurable via `HANDWRITTEN_OCR_EN_MODEL` / `HANDWRITTEN_OCR_AR_MODEL`
in `backend/.env` (see `backend/.env.example`) if you ever want to swap in a
different checkpoint — no code changes needed.

### Download & caching behavior

Models are **not** bundled in the repo or the Docker image, and no path is
hardcoded. The first time a language is actually requested, `transformers`
downloads that model from the Hugging Face Hub automatically and caches it
under the standard Hugging Face cache directory (`~/.cache/huggingface` —
`/root/.cache/huggingface` inside the backend container). Every later
request, and every later process/container restart, reuses that cache — no
re-download.

- **Docker Compose**: `docker-compose.yml`'s `backend_model_cache` volume is
  already mounted at `/root/.cache` (it's what the embedding model,
  cross-encoder, and Whisper already persist through), and Hugging Face's
  default cache dir falls under that same path — so the handwritten OCR
  models persist across `docker compose up`/`--build` restarts with **no
  extra volume or config needed**.
- **Native (non-Docker) dev**: cached under your OS user profile's default
  Hugging Face cache dir, same as every other Hugging Face model this
  project already downloads (sentence-transformers, cross-encoder, Whisper).
- Optional overrides `HF_HOME` / `HF_HUB_CACHE` are documented (commented
  out) in `backend/.env.example` if you want the cache somewhere else
  entirely.

### Loading behavior (in-process)

- Each language's model is loaded **lazily** — nothing loads at import time
  or app startup, so a backend that never receives a handwritten-OCR
  request never pays the load cost. See
  `backend/services/handwritten_ocr_service.py`.
- Once loaded, the `(processor, model)` pair for that language is cached in
  memory for the process lifetime and reused for every subsequent request —
  never reloaded per request. A lock guards the first load per language so
  concurrent requests can't trigger duplicate loads.
- Device: CUDA is used automatically if `torch.cuda.is_available()`,
  otherwise CPU — no configuration needed, and it never crashes on a
  CPU-only machine (this is the same auto/fallback approach
  `backend/utils/device.py` uses for the embedding/cross-encoder models,
  implemented locally here since it's an independent model family).
- Inference runs under `torch.inference_mode()`.

## API

### `POST /api/ocr/handwritten`

`multipart/form-data`:

| Field | Required | Values | Notes |
|---|---|---|---|
| `file` | yes | image file | `.png .jpg .jpeg .bmp .tiff .webp` |
| `language` | yes | `ar` or `en` | No `auto` — see "Why no automatic language detection" below |
| `conversation_id` | no | string | Only used with `index` (see "Pipeline integration") |
| `index` | no | `true`/`false`, default `false` | Only used with `conversation_id` |

Response:

```json
{
  "text": "Hello, this is handwritten text.",
  "language": "en",
  "type": "handwritten"
}
```

```json
{
  "text": "السلام عليكم",
  "language": "ar",
  "type": "handwritten"
}
```

Example:

```bash
curl -X POST \
  -F "file=@test_english.png" \
  -F "language=en" \
  http://localhost:8000/api/ocr/handwritten

curl -X POST \
  -F "file=@test_arabic.png" \
  -F "language=ar" \
  http://localhost:8000/api/ocr/handwritten
```

Errors use the same `{"detail": "..."}` shape as the rest of the API:

| Status | Cause |
|---|---|
| 400 | missing/unsupported `language` (including `auto` — not offered here), missing file, unsupported file extension, empty file, undecodable image |
| 413 | file exceeds `MAX_UPLOAD_SIZE_MB` |
| 422 | `language`/`file` form field missing entirely (FastAPI's own validation) |
| 500 | model failed to load or inference failed |

### Why no automatic language detection

Detecting Arabic vs. English handwriting *before* OCR has run is not
reliable — script shape alone isn't a trustworthy signal for short/messy
handwritten crops, and there is no free/local, non-OCR-based way to do it
well. Rather than silently guessing and returning a plausible-looking but
wrong transcription, `language` is a required, explicit `ar`/`en` choice.
This matches item 4 of the feature request: "if automatic language
detection would be unreliable, do NOT fake it."

## Preprocessing

Deliberately conservative — see `_preprocess()` in
`handwritten_ocr_service.py`. Aggressive binarization/thresholding is **not**
applied, because it tends to destroy Arabic diacritic dots and thin
handwritten strokes, which hurts TrOCR more than it helps.

What actually happens: decode & validate → EXIF auto-orient → convert to
RGB → resize only if far outside a sane range (upscale tiny crops so thin
strokes survive the model's own internal resize; downscale huge images to
bound memory/compute) → mild, non-destructive contrast stretch
(`ImageOps.autocontrast`). TrOCR's own image processor (inside
`TrOCRProcessor`) handles the final model-specific resize/normalization.

## Full-page documents — automatic line segmentation

TrOCR's handwritten checkpoints are trained on **single text-line images**,
not full multi-paragraph pages — feeding a whole page directly into TrOCR
produces near-useless output (verified directly, not assumed: a real
photographed page returned a single stray character). Rather than
requiring the caller to pre-crop every line by hand, `recognize()` in
`handwritten_ocr_service.py` **automatically segments a page-sized image
into individual lines first**:

```
Full page
  → _preprocess()          (orient, RGB, contrast — same as before)
  → _segment_lines()        (classic OpenCV, NOT a learned model — see below)
      → tightly-cropped line 1
      → tightly-cropped line 2
      → ...
  → TrOCR on each line separately
  → join with newlines
  → final text
```

**How `_segment_lines()` works** (no new ML model, no new dependency —
`opencv-python` and `numpy` are already used by the existing printed-OCR
pipeline):

1. Build an "ink" mask from **gradient/edge magnitude** (Scharr + Otsu), not
   raw pixel brightness. This was a deliberate, verified choice: on a real
   photographed notebook page (uneven lighting, a shadow, a laptop and desk
   visible in the background, paper texture), plain intensity thresholding
   misclassified almost the *entire* page as "ink" — a shadow or textured
   background has real brightness variation but almost no sharp edges,
   while a pen stroke always does.
2. Sum ink pixels per row → a smoothed row profile; split it into bands
   using a **per-image adaptive threshold** (1D Otsu on that profile, not a
   single fixed constant — a fixed threshold that worked on one photo's
   lighting failed on another, verified across 4 real test pages).
3. Merge bands separated by a small gap (keeps dots/diacritics attached to
   their line) and force-split any band that grows implausibly tall
   (prevents one failed merge from producing a giant multi-line blob —
   the exact failure this feature exists to avoid).
4. For each surviving band, trim blank left/right margin using the same
   ink mask restricted to that band, then crop the **original, non-
   binarized** image (never the ink mask itself) with padding — see
   "Preserve aspect ratio" below.

**Preserve aspect ratio / avoid clipping**: each crop gets vertical padding
(35% of the detected band's own height, covering ascenders/descenders/dots)
and a fixed horizontal padding (14px, avoiding clipped leading/trailing
strokes or punctuation). Cropping tightly to the actual text — instead of
a full page-width strip with a large blank margin — is also the single
biggest accuracy lever found during testing: TrOCR resizes every input to
a fixed 384×384 square, so a very wide/thin strip gets squashed far more
than a tightly-cropped line.

**Backward compatible**: an image that's already a single pre-cropped line
still works exactly as before — the segmenter just detects one line
spanning the image (or falls back to the whole image if it can't find a
confident structure at all).

**Known remaining limitation**: the row/column projection-profile approach
assumes roughly horizontal, non-overlapping lines (true for photographed
notebook-style pages, which is what was tested). It is not a general
document-layout model — multi-column layouts, heavily skewed/rotated
photos, or very tightly-spaced lines can still occasionally merge two
lines into one crop or split one line into two. This is a deliberate
scope decision (see the feature's original design notes: "do not introduce
a huge new computer-vision system unnecessarily") rather than an oversight.

**Fixed bug (found via real-handwriting evaluation, not by inspection
alone)**: the per-line vertical crop was tight enough that on an input
that was ALREADY a fairly tight single line (a pre-cropped benchmark
image, or a close-up user crop — not just a huge photographed page), the
same tight crop could make the line's aspect ratio MORE extreme than the
input (measured turning a ~10:1 crop into a ~20:1 one), pushing it further
into exactly the "very wide/thin crop gets squashed" failure this
document already warned about below. `_segment_lines()` now widens a
crop's vertical bounds (symmetrically, clamped to the image) whenever its
width:height ratio exceeds `_LINE_MAX_ASPECT_RATIO` (10:1, matching this
doc's own "good" ratio finding) — this keeps the original benefit for
genuinely wide page-width bands while no longer actively hurting
already-tight input. Measured impact on the real IAM/KHATT evaluation
samples: English CER 0.356 → 0.248, Arabic CER 0.844 → 0.762 — at zero
latency/RAM cost.

## Pipeline integration (optional)

Passing `conversation_id` **and** `index=true` feeds the extracted text into
the *exact same* chunk → embed → Qdrant pipeline every normal file upload
goes through (`services/rag_service.update_db_files`) — as a synthetic
`<original-filename>.handwritten-ocr.txt` "file", tagged with that
`conversation_id` exactly like any other upload, preserving document
isolation. This makes the extracted text queryable via the existing chat/
RAG flow without adding a second, parallel ingestion path:

```
Handwritten Image → Handwritten OCR → Extracted Text
                                            │ (index=true & conversation_id set)
                                            ▼
                          Existing chunk → embed → Qdrant → retrieval → chat
```

Response gains `"indexed": true, "chunks_added": <n>` when this runs (or
`"indexed": false, "index_error": "..."` if indexing itself fails — the OCR
result is still returned either way, since indexing failure shouldn't hide
a successful OCR result). Left off by default so a plain "just OCR this
image" call stays a single fast, side-effect-free request, matching the
response shape in the feature spec exactly.

## Docker

No Dockerfile changes were needed — no new system (`apt`) packages are
required for TrOCR (PIL is already used elsewhere; no OpenCV dependency for
this feature). `transformers`, `sentencepiece`, `accelerate`, and
`huggingface_hub` were added to `backend/requirements.txt` (installed into
the existing CPU-only-`torch` build already set up by the Dockerfile — see
`backend/Dockerfile`'s builder stage).

Because model loading is fully lazy (nothing loads at import/startup time),
adding this feature does **not** change backend startup time or the
existing `HEALTHCHECK start_period` in `docker-compose.yml` — a fresh
container reports healthy exactly as before; the (one-time, per language)
handwritten-model download only happens on the first actual
`POST /api/ocr/handwritten` call, not at boot.

```bash
docker compose build
docker compose up -d
docker compose ps
docker compose logs backend
```

## CPU / GPU requirements

- **CPU-only**: works out of the box, no configuration. Inference on a
  single line-sized image takes on the order of a few seconds on CPU after
  the model is loaded (model load/first-download is the slow part, not
  steady-state inference).
- **GPU**: used automatically if `torch.cuda.is_available()` returns `True`
  in the backend's Python/torch environment (i.e. a CUDA-enabled torch
  build is installed and a GPU is visible) — no extra configuration.

## Batched multi-line inference

A page with more than one detected line is now recognized in RAM-bounded
sub-batches (`HANDWRITTEN_OCR_MAX_BATCH_SIZE`, default 8 lines per model
call) instead of one full model call per line. Benchmarked before
adopting (`scripts/evaluate_ocr_followup.py`): batching measured a
consistent 15-54% latency reduction for English with IDENTICAL output
text (not a quality/speed tradeoff) across two independent page sizes,
and was latency-neutral for Arabic (never a regression) — so it was
adopted for both languages rather than special-cased. Any sub-batch that
fails falls back to sequential per-image recognition for just that
sub-batch, so one malformed line image can't fail an entire page. A
single line (the majority of real calls — one pre-cropped image, or a
page that segments into one line) skips batching entirely, since there's
nothing to batch.

## Limitations (explicit, not glossed over)

- Full pages are now handled automatically (see "Full-page documents"
  above), but the segmenter assumes roughly horizontal, non-overlapping
  lines — not a general document-layout model. Multi-column layouts,
  heavily skewed photos, or very tightly-spaced lines can still
  occasionally merge two lines into one crop.
- Even on a well-segmented single line, the underlying "base" (not
  "large") models are not perfectly accurate on real, messy cursive
  handwriting — verified directly: recognizable partial-word matches on
  well-cropped real handwriting, not flawless transcription. This is a
  model-capability ceiling, not a pipeline defect.
- No language auto-detection — see "Why no automatic language detection"
  above.
- Recognition quality depends on handwriting legibility and image quality,
  same as any OCR model; this is not a guarantee of perfect transcription
  for messy handwriting, unusual scripts, or very low-resolution photos.
- The Arabic model (`RayR1/trocr-base-arabic-handwritten`) is a
  community-fine-tuned checkpoint, not an official Microsoft release like
  the English one — quality/coverage may vary more than the English model.
- **Crop aspect ratio measurably affects accuracy.** TrOCR resizes every
  input to a fixed square (384×384) before the encoder, regardless of the
  original image's aspect ratio. A crop that keeps a full page's width
  (e.g. a thin ~2500×150px strip, ratio ~16:1) gets squashed far more
  aggressively than a crop trimmed to just the text's own width (ratio
  ~10:1 or less) — verified directly: on the same handwritten line, a
  full-page-width crop produced a mostly-wrong transcription, while
  trimming the same crop to the actual text bounding box produced a
  substantially more accurate one (correct words: "Manipulators", "chair"
  [1 letter off from "chain"], "of", "Regid" [1 letter off from "Rigid"],
  "bodies"). **Recommendation: crop as tightly as practical around the
  text itself, not just a horizontal band across the full image width.**
- **Arabic accuracy is genuinely poor on real handwriting**, now quantified
  (`scripts/evaluate_handwritten_ocr.py` against real KHATT corpus
  samples): average CER ≈0.76 even after the aspect-ratio fix above —
  several real samples recognize as just "." or a few stray characters.
  No realistic lighter or more accurate free/local alternative was found
  on the Hugging Face Hub during this evaluation. This is a real,
  high-severity limitation of the current Arabic checkpoint, not a
  pipeline defect — flagged here rather than glossed over.
