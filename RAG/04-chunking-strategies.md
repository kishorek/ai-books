# Chapter 4: Chunking Strategies

> **Last verified: June 2026.**

Chunking is one of the most impactful decisions in RAG. How you split documents directly determines retrieval quality, context relevance, and answer quality. The chunking decision is made early and affects everything downstream.

## Why Chunking Matters

Retrieval works by finding chunks relevant to a query. If a chunk is too small, it lacks context — the model sees a fragment without understanding what it means. If a chunk is too large, it contains noise — the relevant information is buried in irrelevant content.

The optimal chunk size depends on your documents and queries. Technical documentation with dense information benefits from smaller chunks (256 tokens). Narrative content with flowing context benefits from larger chunks (512-1024 tokens). The only way to find the right size is to test with your actual data.

## Chunking Techniques

### Fixed-Size Chunking

The simplest approach: split text into equal-sized chunks with optional overlap. Fast and predictable, but it breaks sentences and paragraphs at arbitrary boundaries. Use when document structure is uniform and retrieval quality is not critical.

### Semantic Chunking

Split at semantic boundaries — where the meaning shifts. Compare embeddings of consecutive sentences; when similarity drops below a threshold, start a new chunk. This preserves meaning better than fixed-size chunking but requires an embedding model pass over the entire document.

### Section-Based Chunking

Split by document structure — headers, sections, and subsections. This is the most natural approach for structured documents (technical documentation, legal contracts, reports). It preserves the author's organizational intent.

### Parent-Child Chunking

Create two levels of chunks: small "child" chunks (256 tokens) for retrieval precision, and large "parent" chunks (1024 tokens) for context. Retrieve with children (high precision), return parents (rich context). This combines the best of both approaches.

The retrieval process: search with small chunks to find precise matches, then expand to the parent chunk for full context. The model sees relevant, context-rich information without the noise of irrelevant large chunks.

## Advanced Chunking

**Hierarchical chunking** creates a tree of chunks at multiple levels of granularity. **Code chunking** respects function and class boundaries. **Table chunking** preserves tabular structure. **Multimodal chunking** handles mixed content (text, images, tables).

The right chunking strategy depends on your document types. A system processing only PDFs might use section-based chunking. A system processing code, documentation, and tables needs multiple strategies.

## Chunk Optimization

### Size Selection

The research and practice consensus: 256-512 tokens is the sweet spot for most applications. Smaller chunks (128 tokens) improve precision but lose context. Larger chunks (1024+ tokens) improve recall but add noise.

### Overlap

Overlapping chunks prevents context loss at boundaries. A 10-20 percent overlap ensures that sentences split across chunks appear in at least one complete form. Too much overlap wastes tokens on duplicate content.

### Deduplication

After chunking, deduplicate near-identical chunks. Different documents may contain similar content. Duplicate chunks waste retrieval slots and context space.

## Case Study: Chunk Size Impact

A customer support RAG system tested different chunk sizes:

| Chunk Size | Precision@5 | Recall@5 | Latency |
|-----------|-------------|----------|---------|
| 128 tokens | 82% | 65% | 1.2s |
| 256 tokens | 88% | 78% | 1.4s |
| 512 tokens | 85% | 85% | 1.6s |
| 1024 tokens | 78% | 91% | 2.0s |

256 tokens provided the best precision. 512 tokens balanced precision and recall. 1024 tokens had the highest recall but lowest precision. The team chose 256 tokens with parent-child expansion for the best overall quality.

## Key Takeaways

- Chunking is the most impactful early decision — test with your actual data
- Parent-child chunking combines precision (small chunks) with context (large chunks)
- 256-512 tokens is the sweet spot for most applications
- Semantic chunking outperforms fixed-size for natural language documents
- Overlap prevents context loss at boundaries — 10-20 percent is typical
- The right strategy depends on your document types — one size does not fit all

## Further Reading

- Chunking strategies research
- LangChain text splitters documentation
