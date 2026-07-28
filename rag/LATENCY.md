# Latency-oriented changes

## What changed

### Defaults (see `.env`)
| Setting | Old | New |
|---------|-----|-----|
| `RAG_ENABLE_QUERY_EXPANSION` | false | false (unchanged, still opt-in) |
| `RAG_FUSED_CANDIDATE_MULTIPLIER` | 2 | **1** |
| `RAG_SEARCH_CANDIDATE_MULTIPLIER` | (hardcoded ×2) | **1.5** |
| `RAG_MAX_CONTEXT_CHUNKS` | 10 | **6** |
| `RAG_RERANK_THRESHOLD` | 0.3 | **0.4** |
| `AGENT_MAX_ITERATIONS` | 6 | **4** |
| `LLM_MAX_TOKENS` | 800 | **500** |
| `LLM_TEMPERATURE` | 0.2 | **0.1** |
| `AGENT_MODEL` | llama-3.1-8b-instant | unchanged (fast planner) |

### Code improvements
1. **search.py**
   - Configurable embedding model (`RAG_EMBEDDING_MODEL`)
   - Device selection (`RAG_DEVICE=auto|cuda|cpu`)
   - Process-wide LRU cache for query embeddings (`RAG_EMBED_CACHE_SIZE`)
   - Stage timing when `RAG_LOG_LATENCY=true`

2. **reranker.py**
   - Configurable model (`RAG_RERANKER_MODEL`) — can point at a lighter CE
   - Device selection
   - Higher default threshold
   - Stage timing

3. **retriever.py**
   - Search candidate multiplier configurable
   - Lower fused multiplier default
   - End-to-end stage timings (expansion / dense+sparse / RRF / total)

4. **warmup.py**
   - Call `from rag.warmup import warm_models; warm_models()` at process
     start (e.g. in your FastAPI lifespan / Gunicorn post-fork) so the
     first request does not pay model load time.

5. **generate_tool / generator / agent**
   - Tighter defaults for context size, max tokens, temperature, max iterations

## Recommended production setup

```bash
# .env
RAG_DEVICE=cuda          # if you have a GPU
RAG_LOG_LATENCY=true     # while tuning; turn off later
RAG_EMBEDDING_MODEL=intfloat/multilingual-e5-large   # or e5-base for more speed
RAG_RERANKER_MODEL=BAAI/bge-reranker-v2-m3           # or a lighter CE if needed
```

```python
# at process start
from rag.warmup import warm_models
warm_models()
```

Prefer `agent.run_stream(...)` in any HTTP layer so tokens reach the
client as soon as the final generation starts.

## Optional further wins
- Switch to `intfloat/multilingual-e5-base` if quality holds.
- Point `RAG_RERANKER_MODEL` at a smaller cross-encoder.
- Raise `RAG_RERANK_THRESHOLD` further if you still see noisy chunks.
- Profile with the latency logs to confirm which stage dominates.
