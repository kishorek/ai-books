# Chapter 6: Building AI Applications

This is where theory becomes products. Building a GenAI application is not just calling an API — it is designing an architecture that handles streaming, state, context, retrieval, tools, evaluation, monitoring, and scaling. This chapter covers application types and the architecture patterns that make them work.

## Application Types

### Chatbots and Assistants

The simplest application type. User sends a message, model responds. The architecture requires streaming response delivery (SSE or WebSocket), chat history management with token budget awareness, session persistence, rate limiting per user, and content filtering.

The key architectural decision is context management: how much history to retain, when to summarize, and how to balance history against retrieved documents in the context window.

### Copilots

AI assistants embedded in existing workflows — GitHub Copilot, Notion AI, Figma AI. The architecture requires ultra-low latency (under 500ms for completions), context extraction from editor state, incremental streaming, and background processing for complex tasks.

The distinguishing characteristic is the latency requirement. Code completions must feel instant. This means using fast, cheap models (GPT-5.4 nano) for completions and reserving expensive models for complex refactoring tasks that can run in the background.

### Knowledge Assistants

RAG-based systems that answer questions from organizational knowledge. The architecture centers on the retrieval pipeline: document ingestion, embedding generation, hybrid search (dense plus sparse), reranking, context assembly, and citation extraction.

### Customer Support Systems

Automated support with human escalation. The architecture adds intent classification (routing to the right handler), knowledge base integration, escalation logic, SLA tracking, and sentiment monitoring.

The critical design decision is the escalation threshold. Too aggressive escalation wastes human agent time. Too passive escalation frustrates customers. The threshold should be configurable and tuned based on measured quality metrics.

### Research Assistants and Content Generation

Deep analysis and synthesis tasks, and automated content creation at scale. Both require multi-step reasoning, external tool integration, and quality gates.

## The Architecture Layers

A production GenAI application has six distinct layers, each with specific responsibilities.

**The frontend layer** handles user interaction. For chat applications, streaming is mandatory — chunked rendering, typing indicators, and partial response display are UX requirements that affect frontend architecture.

**The backend layer** orchestrates between frontend and LLM. It should be stateless for horizontal scaling, separate LLM calls from business logic, implement circuit breakers for LLM APIs, queue long-running operations, and cache responses for identical queries.

**The LLM layer** is the abstraction over model providers. It handles provider selection, fallback chains, rate limiting, and cost tracking. This layer should be provider-agnostic — you should be able to switch providers without changing business logic.

**The data layer** provides persistent storage. Chat history goes in Redis (fast retrieval, session state). Documents go in object storage (S3, GCS). Embeddings go in a vector database. Metadata goes in Postgres. Response caches go in Redis.

**The memory layer** manages state across conversations and sessions. Short-term memory (recent conversation) lives in Redis. Long-term memory (relevant past interactions) lives in a vector database. The memory manager retrieves both and assembles them into context.

**The observability layer** provides monitoring, tracing, and debugging. OpenTelemetry for distributed tracing, Prometheus and Grafana for metrics, structured logging for debugging, and custom dashboards for cost tracking and quality scoring.

## Enterprise Constraint Decision Table

| Constraint | Recommended Architecture |
|-----------|------------------------|
| Under 500ms latency | Streaming plus fast model (GPT-5.4 nano) |
| Over 1M requests/day | Multi-provider routing plus caching |
| Data sovereignty | Self-hosted (vLLM plus Llama) |
| SOC2 compliance | Audit logging plus encryption at rest |
| Real-time collaboration | WebSocket plus conflict resolution |

## Case Study: Enterprise Knowledge Assistant

A legal firm built a knowledge assistant for 500 attorneys across 100,000 documents. The architecture combined Elasticsearch for BM25 search with Pinecone for dense retrieval, Cohere for reranking, and Claude Sonnet 4.6 for generation.

The system achieved 94 percent answer accuracy, 97 percent citation accuracy, 3.2 second p95 latency, $0.008 per query, and 4.3 out of 5 user satisfaction. The key was the hybrid search approach — BM25 captured exact legal terms that dense search missed, while dense search captured semantic similarity that keyword search missed.

## Key Takeaways

- Application architecture is layered: frontend, backend, LLM layer, data layer, memory layer, observability
- Streaming is mandatory for interactive applications — it is a UX and technical requirement
- The LLM layer must be abstracted, routed, and resilient — provider failures are inevitable
- Memory management (short-term and long-term) is a core architectural concern
- Observability from day one — tracing, metrics, cost tracking, quality scoring
- Enterprise applications require authentication, RBAC, audit logging, and compliance

## Further Reading

- "Designing Data-Intensive Applications" by Martin Kleppmann
- OpenTelemetry Documentation
- FastAPI and React streaming documentation
