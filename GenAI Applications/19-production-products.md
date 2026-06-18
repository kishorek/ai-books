# Chapter 19: Building Production GenAI Products

This chapter bridges technical architecture and product design. Building a GenAI product requires understanding user needs, designing for trust, measuring success, and iterating based on real usage data.

## Product Categories

The major GenAI product categories each have distinct architectural requirements:

**Customer support AI** needs intent classification, knowledge base integration, escalation logic, and SLA tracking. The key metric is resolution rate — what percentage of queries are handled without human intervention.

**Enterprise search** needs RAG pipelines, hybrid search, citation extraction, and permission filtering. The key metric is time-to-answer — how quickly users find the information they need.

**Coding assistants** need code-aware models, IDE integration, streaming, and security scanning. The key metric is acceptance rate — what percentage of suggested code is used.

**Research assistants** need multi-step reasoning, web search, and citation management. The key metric is completeness — how thoroughly the research covers the topic.

**Workflow automation** needs tool calling, state management, and audit logging. The key metric is automation rate — what percentage of workflows complete without human intervention.

## Engineering Concerns

### Scalability

Horizontal scaling through load balancing and multiple API instances. Caching for repeated queries (reduces LLM calls 20-40 percent). Model routing for cost optimization. Async processing for long-running tasks.

### Reliability

Circuit breakers prevent cascade failures. Retry with exponential backoff handles transient errors. Fallback chains provide redundancy. Health checks detect and replace unhealthy instances.

### Cost Optimization

Model routing is the biggest lever — 30-50 percent savings. Caching saves 20-40 percent. Prompt optimization reduces token count 10-30 percent. Self-hosting saves 50-80 percent for high-volume steady workloads.

### User Experience

Streaming is mandatory — users will not wait 10 seconds for a response. Source citations build trust. Confidence indicators help users judge reliability. Feedback loops enable continuous improvement.

## Adoption Metrics

The metrics that matter are not technical — they are business metrics: daily active users (are people using it?), queries per user (how deeply?), resolution rate (does it work?), return rate (do they come back?), and NPS (are they satisfied?).

A technically excellent product that users do not adopt is a failure. Design for adoption, not just accuracy.

## Key Takeaways

- Product design matters as much as technical architecture
- Measure what matters: resolution rate, time saved, user satisfaction
- Streaming and citations are UX requirements, not nice-to-haves
- Cost optimization should be a design constraint, not an afterthought
- Human-in-the-loop is essential for trust

## Further Reading

- "The Design of Everyday Things" by Don Norman
- Product-Led Growth resources
