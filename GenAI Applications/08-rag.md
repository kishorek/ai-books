# Chapter 8: Retrieval Augmented Generation

RAG is the technique that grounds LLM responses in factual data. Instead of relying solely on the model's training knowledge, you retrieve relevant documents and include them in the prompt. RAG reduces hallucinations, keeps knowledge current without retraining, and enables domain-specific expertise. But RAG is fragile — retrieval quality depends on chunking strategy, embedding model, search algorithm, reranking, and context assembly.

## Why RAG

LLMs have three fundamental limitations that RAG addresses. Knowledge cutoff means models do not know about events after training. Hallucination means models generate plausible but incorrect information. Domain specificity means models lack specialized organizational knowledge.

RAG is the default choice for knowledge-intensive applications. Fine-tuning is better when you need consistent formatting, domain-specific reasoning, or cost reduction at very high volumes. But for most use cases — where knowledge changes frequently, where you need source citations, or where training data is limited — RAG is the right approach.

## The RAG Pipeline

The standard RAG pipeline has five stages. Documents are ingested, chunked, embedded, and indexed. At query time, the query is embedded, candidates are retrieved, reranked, assembled into context, and fed to the LLM for generation.

Each stage introduces potential quality loss. Poor chunking breaks important context. Weak embeddings miss semantic similarity. Inadequate retrieval returns irrelevant documents. Missing reranking leaves noise in the results. Bad context assembly buries relevant information.

## Embeddings and Vector Search

Embeddings convert text to dense vectors that capture semantic meaning. The quality of embeddings directly affects retrieval accuracy. BGE-M3 offers the best quality-to-cost ratio for self-hosted deployments. OpenAI embeddings are easiest to use but cost more at scale. Cohere provides a strong managed option.

Vector databases store and retrieve these embeddings. Pinecone is best for managed, zero-ops. Qdrant and Milvus are strong for self-hosted performance. pgvector is excellent if you already use PostgreSQL. Chroma is great for development but not production.

## Chunking: The Most Impactful Early Decision

How you split documents directly determines retrieval quality. Fixed-size chunking (256-512 tokens) works for simple documents. Semantic chunking splits at natural breaks in meaning. Section-based chunking respects document structure. Parent-child chunking creates small chunks for retrieval precision and large chunks for context richness.

The key finding across research and practice: parent-child chunking — retrieve with small chunks (256 tokens), return context from large chunks (1024 tokens) — combines the precision of small chunks with the context preservation of large chunks. It consistently outperforms fixed-size approaches.

Chunk size matters. Too small (128 tokens) and you lose context. Too large (2048 tokens) and you add noise. The sweet spot for most applications is 256-512 tokens, but the optimal size depends on your document types and should be tuned with your actual data.

## The Retrieval Pipeline

The retrieval pipeline has three layers: initial retrieval (dense plus sparse search), reranking, and context assembly.

**Hybrid search** combines dense (semantic) and sparse (BM25) retrieval. Dense search captures meaning — "vehicle" matches "car." Sparse search captures exact terms — "Q4 2024 revenue" matches the exact phrase. Neither alone is sufficient. Hybrid search with reciprocal rank fusion consistently outperforms either approach.

**Reranking** is the highest-ROI investment in RAG quality. After initial retrieval returns 20 candidates, a cross-encoder reranks them by relevance and returns the top 5. This improves precision 20-30 percent for 50-200ms latency. Cohere rerank-v3.5 and BGE-reranker-v2-m3 are the best production choices.

**Context assembly** orders and fits retrieved documents into the token budget. The most relevant document goes first (maximum attention), followed by supporting documents. Deduplication removes redundant content.

## Advanced RAG Patterns

**Multi-query retrieval** handles ambiguous queries by generating multiple query variations and merging results. **Hyde** (Hypothetical Document Embeddings) generates a hypothetical answer and uses it for retrieval, improving results for complex questions. **Agentic RAG** uses an agent to decide when, how, and what to retrieve — decomposing complex queries, filling information gaps, and iterating until quality thresholds are met.

## Case Study: Legal Research RAG

A legal firm built a RAG system for case law research across 100,000 documents. The optimization journey illustrates how each improvement stacked:

| Phase | Technique | Precision@5 | Latency |
|-------|-----------|-------------|---------|
| 1 | Dense search only | 58% | 2.1s |
| 2 | + Hybrid search | 72% | 2.3s |
| 3 | + Reranking | 85% | 2.5s |
| 4 | + Semantic chunking | 91% | 2.4s |
| 5 | + Query decomposition | 94% | 2.8s |

The biggest single improvement was reranking (+13 percent). The combination achieved 94 percent precision compared to 58 percent for basic dense search.

## Key Takeaways

- RAG reduces hallucinations and keeps knowledge current — but retrieval quality is fragile
- Hybrid search (dense + sparse) outperforms either alone — always use both
- Reranking is the highest-ROI investment — 20-30 percent precision improvement for minimal latency cost
- Chunking strategy is the most impactful early decision — parent-child often wins
- Measure retrieval quality separately from generation quality — bad retrieval = bad answers
- Agentic RAG handles complex queries that static pipelines miss

## Further Reading

- Lewis et al., "Retrieval-Augmented Generation" (2020)
- Gao et al., "RAG for LLMs: A Survey" (2024)
- Liu et al., "Lost in the Middle" (2023)
