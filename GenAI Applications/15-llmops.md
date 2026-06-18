# Chapter 15: LLMOps

LLMOps is the discipline of operating LLM applications in production — monitoring, observability, cost tracking, deployment, and incident response. Without LLMOps, you are flying blind.

## Monitoring

### Cost Monitoring

GenAI costs scale linearly with usage. Without monitoring, bills surprise you. Track cost per model, per user, per team, and per application. Set budget alerts at 80 and 95 percent thresholds. The most common surprise is token consumption from long-context applications — a few requests with 100K+ tokens can dominate the daily bill.

### Token Monitoring

Track input and output tokens per request. Monitor averages and outliers. Unusual spikes indicate bugs (infinite loops, context that grows without bound) or abuse (users crafting token-heavy prompts).

### Latency Monitoring

Track time-to-first-token (TTFT) and total response time. TTFT measures how quickly the user sees the first word — this is the perceived speed. Total response time includes the full generation. Both matter, but TTFT matters more for user experience.

### Error Monitoring

Track error rates by type: rate limits, authentication failures, model overloaded, timeout, and validation errors. Rate limit errors indicate you need to implement queuing or upgrade your tier. Authentication failures indicate credential issues. Model overloaded errors indicate provider capacity problems.

## Observability

### Distributed Tracing

Trace requests across your entire system — from API gateway through retrieval, reranking, LLM generation, and response. OpenTelemetry is the standard. Jaeger or Tempo for visualization.

Tracing reveals bottlenecks. If retrieval takes 2 seconds and generation takes 1 second, optimizing generation only saves 1 second. Optimizing retrieval saves 2 seconds.

### Structured Logging

Log every request with consistent fields: request ID, user ID, model, input tokens, output tokens, latency, cost, quality score. Structured logging enables debugging and analysis.

### Metrics

Key metrics to dashboard: request latency (p50, p95, p99), error rate, cost per request, token usage, quality score, and cache hit rate. Grafana with Prometheus is the standard stack.

## Operations

### Deployment

Container-based deployment with health checks. The application should expose a health endpoint that verifies LLM connectivity and vector database access. Rolling deployments with automatic rollback on health check failure.

### Versioning

Version your prompts, models, and configurations separately. A prompt change can affect quality as much as a model change. Track which version produced which output for debugging and reproducibility.

### Canary Deployments

Gradually shift traffic to new versions. Start with 10 percent, monitor for five minutes, increase to 50 percent, monitor again, then 100 percent. If error rate exceeds threshold or latency spikes, automatically rollback.

## Key Takeaways

- Cost monitoring is mandatory — GenAI costs scale linearly and can surprise you
- Tracing with OpenTelemetry gives visibility across your entire stack
- Canary deployments with automatic rollback prevent bad deployments from affecting all users
- Version your prompts and models — you need to know which version produced which output
- Token monitoring catches unusual patterns — spikes may indicate bugs or abuse

## Further Reading

- OpenTelemetry Documentation
- LangSmith and Arize Phoenix documentation
- Prometheus and Grafana documentation
