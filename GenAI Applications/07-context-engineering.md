# Chapter 7: Context Engineering

Context engineering is one of the most important modern skills. It is the discipline of selecting, organizing, and optimizing the information you provide to an LLM. Every token you include competes for attention space. Poor context engineering means the model sees irrelevant information, misses critical details, and produces lower quality outputs.

Context engineering is not just "putting documents in the prompt." It is a systematic approach to token budgeting, context selection, compression, and prioritization that directly determines system quality and cost.

## Context Windows as Working Memory

The context window is the model's working memory. Understanding its limitations is the foundation of context engineering. A typical allocation for a 128K token window: 500 tokens for the system prompt, 40K for chat history, 50K for retrieved documents, 200 for the user query, and 32K reserved for the response.

The key principle is that every token used for one purpose is a token not available for another. Extensive chat history means less room for retrieved documents. A long system prompt means less room for context. Design your token allocation strategy before implementing.

## Context Selection Strategies

Not all information is equally relevant. Selection strategies include semantic search (embed query, find similar documents), keyword search (BM25 matching), hybrid (combine both), recency (most recent first), and importance (score by relevance).

Hybrid selection — combining semantic and keyword search with recency and importance scoring — consistently outperforms any single strategy. The architecture is straightforward: run multiple searches in parallel, merge results with reciprocal rank fusion, deduplicate, and fit within the token budget.

## Context Compression

When context exceeds the token budget, compression reduces token count while preserving essential information. Techniques include extractive summarization (60-80 percent reduction, low quality loss), key point extraction (80-95 percent reduction, medium quality loss), and sentence trimming (30-50 percent reduction, low quality loss).

The choice depends on the compression ratio needed. For moderate reduction (context is 10-30 percent over budget), extractive summarization preserves the most detail. For heavy reduction (context is 50+ percent over budget), key point extraction focuses on the most important information.

## Context Sources

Six sources typically contribute to the context window:

**User input** — the current query. Always placed last (closest to output generation) for maximum attention. Analyze it for intent and key entities to guide retrieval.

**Chat history** — previous conversation turns. The most expensive context source in terms of token consumption. Strategies include sliding window (last N turns), summarized history (compress older turns), and hybrid (recent turns full, older turns summarized).

**Documents** — external knowledge retrieved via RAG. The most valuable source for factual accuracy. Quality depends entirely on the retrieval pipeline.

**Databases** — structured data accessed via function calling. Useful for quantitative queries that need precise data.

**APIs** — external services accessed via tool calling. Extends the model's capabilities beyond its training data.

**Enterprise systems** — internal CRM, ERP, and ticketing systems. Provides organizational context.

## Token Reduction and Cost Optimization

Token reduction techniques include concise system prompts (10-20 percent savings), deduplication (5-15 percent), structured formats like tables versus prose (20-40 percent), and selective inclusion based on relevance (30-60 percent).

Cost optimization goes beyond token reduction. It includes routing to cheaper models for simple tasks, caching responses for repeated queries, batching similar requests, and compressing long contexts to reduce input tokens.

## Context Quality Measurement

Measure context quality across four dimensions: relevance (does context contain answer-relevant information), completeness (is all necessary information present), noise (does context contain irrelevant information), and freshness (is context up to date).

These measurements close the evaluation loop. Without them, you are optimizing blindly.

## Case Study: Context Engineering for Customer Support

A customer support system using GPT-5.4 was achieving only 72 percent answer quality despite sending 45K tokens of context per query. The root cause: full product documentation (80K tokens) was being sent for every query.

The fix was multi-stage context engineering: analyze the query to determine what information is needed, retrieve only relevant chunks (256 tokens each, top 10), compress if still over budget, add recent conversation context (last 3 turns), and assemble the final context.

Results: answer quality improved from 72 to 91 percent, average context tokens dropped from 45K to 8K (82 percent reduction), cost per query dropped from $0.015 to $0.003 (80 percent reduction), and latency improved from 6.2s to 1.8s (71 percent reduction).

The insight: less context, better targeted, improved quality. The model focused on relevant information instead of wading through noise.

## Key Takeaways

- Context engineering is the single biggest quality lever — it matters more than model selection for most applications
- Token budgeting is mandatory — allocate context window explicitly, not arbitrarily
- Less relevant context beats more irrelevant context — targeted retrieval improves quality and reduces cost
- Context compression saves money and improves quality — summarize long histories, extract key points from documents
- Context quality measurement closes the loop — evaluate relevance, completeness, noise, and freshness

## Further Reading

- Liu et al., "Lost in the Middle" (2023) — context position effects
- Anthropic's "Building Effective Agents" guide — context management patterns
