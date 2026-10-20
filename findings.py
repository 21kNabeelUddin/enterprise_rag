# These three bullets are the ones an interviewer can actually pressure-test. The latency and behaviour numbers are in the committed benchmark JSON. The corpus size is from your own startup fingerprint in `EVALUATION_LOG.txt` (6,852 filings, 1,298,998 chunks, HNSW index about 9.1 GB). Re-run that count before you put it on a CV if the database has changed.
#
# **1. Hybrid search over 1.3 million SEC chunks, with access control inside the query**
#
# Built a local RAG index over the 2020 SEC EDGAR 10-K corpus: 6,852 filings, 1,298,998 section-aware chunks, 1,024-dimensional BGE-large embeddings in Postgres/pgvector (HNSW, ~9.1 GB) plus a GIN full-text index. Dense top-50 and lexical top-50 are fused with Reciprocal Rank Fusion in Python, because an `OR` across common terms was scanning the full ~1.3M-row table (~2.9 s). ACL group overlap is applied inside both the vector query and the lexical query. Measured retrieval on that index: encode 56 ms, retrieve 207 ms (n=10).
#
# **2. Stopped one company from swallowing comparison questions, and stopped neighbour chunks from bypassing ACL**
#
# When a question names more than one filer, a single `cik IN (...)` top-k lets the company whose filing language is closest fill every slot. The retriever now resolves names and tickers against the SEC directory (10,443 tickers), runs a separate search per CIK (capped at 3), and round-robin interleaves those rankings before the Qwen3-Reranker-0.6B cross-encoder. Adjacent context (±1 chunk) is loaded only after the same ACL check is applied again, so a neighbour cannot leak a chunk the caller is not allowed to see.
#
# **3. Answers are checked before they are streamed, and the speedup is measured as a repeat-query effect**
#
# Shipped a 14-step query service (auth, rate limit, Llama Prompt Guard 2, PII, rewrite, hybrid retrieve, rerank, 6,500-token context, local Qwen3.5-9B Q4 on an 8 GB laptop GPU, then citation, embedding-grounding, and secret-leak checks). Tokens are streamed only after the output guard passes. On a 12-question behaviour set: 6/6 answerable questions returned cited answers, 3/3 prompt-injection attempts were blocked before retrieval (~0.4–0.5 s versus ~25 s for a full answer), 3/3 out-of-corpus questions were refused, 0/9 benign questions were false-blocked, and 6/6 answers cited only real sources. A repeated 5,040-token prompt went from a 0% to a 99.9% KV-cache hit (5,036/5,040 tokens) and prompt-eval time fell from 3,631 ms to 148 ms (−95.9%). End-to-end fell from 32.4 s to 26.5 s (−18%, n=3 off / n=2 warm), because generation is 80.2% of the 26.7 s mean path (n=5, guards on).
#
# Keep the claims inside these bounds if someone follows up:
#
# - Call the 12/12 result **behaviour accuracy** (answer, refuse, or block). Exact match, token F1, NDCG, and claim-level faithfulness are not measured. The grounding check is embedding cosine similarity (threshold 0.45, at least 60% of sentences), not an NLI model.
# - The earlier refusal score moved from 66.7% to 100% because refusal wording was relabelled. That is a status fix, not a better model.
# - The −95.9% figure is prompt-eval time on a repeated identical query. A new prompt still has a 0% cache hit. Do not write “95% faster” as a general latency claim. With n=5, the reported p95 is the maximum observation, not a production SLA.