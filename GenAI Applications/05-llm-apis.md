# Chapter 5: LLM APIs and Model Providers

> **Last verified: June 2026.** API endpoints, pricing, and rate limits change frequently. Always check provider documentation before implementation.

Every GenAI application talks to a model provider through an API. The API design, pricing model, rate limits, and feature set directly affect your architecture. This chapter covers the major providers, their patterns, and practical integration strategies.

## The Major Providers

### OpenAI

OpenAI's Chat Completions API is the de facto standard — most other providers offer compatible interfaces. GPT-5.4 ($2.50/$15.00 per million tokens) is the default for most applications. GPT-5.4 mini ($0.75/$4.50) provides a cost-effective alternative. GPT-5.4 nano ($0.20/$1.25) is the cheapest option. All support one million token context windows.

OpenAI offers automatic prompt caching (repeated prefixes get 50 percent discount), a batch API (50 percent discount for non-real-time workloads), and a Responses API designed for agentic workflows that simplifies tool calling.

The structured output feature (`response_format`) achieves 99 percent schema adherence — essential for applications that need JSON output. Function calling is the most mature implementation in the industry.

### Anthropic

Claude uses a Messages API with key differences from OpenAI: the system prompt is a separate parameter rather than a message in the array. Claude Sonnet 4.6 ($3.00/$15.00) excels at coding, analysis, and structured output. Its JSON schema adherence is the best available.

The standout feature is prompt caching. By marking the system prompt with `cache_control`, subsequent requests reuse the cached prefix at 90 percent reduced cost and 2x speed. For applications with long system prompts (enterprise rules, domain context), this is a significant optimization.

### Google

Gemini offers both AI Studio (direct API) and Vertex AI (enterprise). Gemini 2.5 Pro ($1.25-$5.00/$5.00-$10.00) provides competitive pricing with one million token context windows. The pricing is tiered — under 128K tokens is cheaper than over. Gemini 2.5 Flash ($0.075-$0.30/$0.30-$0.60) offers exceptional cost-to-performance.

Vertex AI adds enterprise features: IAM, audit logging, VPC peering, and data residency. For enterprise deployments on GCP, Vertex AI is the path.

### DeepSeek

DeepSeek V3 ($0.27/$1.10) delivers frontier quality at a fraction of Western provider costs. DeepSeek R1 ($0.55/$2.19) adds reasoning capabilities. The API is OpenAI-compatible — migration is straightforward.

### Open Source: Ollama, vLLM, TGI

For self-hosted deployments, Ollama serves as a development tool (not production). vLLM is the production standard — it achieves 5-10x higher throughput than naive implementations through continuous batching and PagedAttention. TGI (Text Generation Inference) from Hugging Face is an alternative with strong Docker and Kubernetes integration.

## Building a Provider Abstraction

Provider abstraction is essential for avoiding lock-in, enabling A/B testing, and providing fallback during outages. The pattern is a simple interface with provider-specific implementations:

The interface defines three methods: `chat` (synchronous completion), `chat_stream` (streaming completion), and `count_tokens` (token counting). Each provider implements these methods using its native SDK.

The abstraction enables several critical capabilities: routing queries to the cheapest provider per task, falling back to alternative providers during outages, A/B testing across providers for quality comparison, and avoiding vendor lock-in.

## Rate Limiting and Resilience

All providers impose rate limits — typically 200-500 requests per minute on free tiers, scaling to 10,000+ on paid tiers. Design your system to queue requests when rate limited, distribute across multiple API keys, implement exponential backoff with jitter, and cache responses for repeated queries.

Circuit breakers prevent cascade failures. If a provider is down, the circuit breaker opens and routes to fallback providers instead of letting requests pile up and timeout.

## Provider Comparison

| Dimension | OpenAI | Anthropic | Google | DeepSeek | Open Source |
|-----------|--------|-----------|--------|----------|-------------|
| Best quality | GPT-5.5 | Claude Sonnet 4.6 | Gemini 2.5 Pro | DeepSeek V4 | Llama 4 Maverick |
| Best cost | GPT-5.4 nano | Haiku 3.5 | Gemini Flash | DeepSeek V3 | Self-hosted |
| Largest context | 1M | 1M (beta) | 1M | 1M (V4) | 10M (Scout) |
| Structured output | Excellent | Best | Good | Limited | Limited |
| Data sovereignty | No | No | No | No | Yes |

## Case Study: Multi-Provider Strategy

A SaaS company routed queries across three providers based on task requirements. Complex analysis went to Claude Sonnet 4.6, simple queries to GPT-5.4 nano, and high-volume workloads to DeepSeek V3. Monthly cost dropped from $150K to $65K (57 percent reduction), availability improved from 99.5 to 99.95 percent, and average latency decreased from 2.1 to 1.4 seconds.

The engineering investment in the abstraction layer paid for itself in the first month.

## Key Takeaways

- Provider abstraction is essential — build it early, not after you are locked in
- Rate limiting, retry logic, and fallback chains are production requirements
- Prompt caching (Anthropic) reduces cost 90 percent for repeated system prompts
- Multi-provider routing reduces cost 30-50 percent while improving availability
- vLLM is the standard for self-hosted serving — 5-10x higher throughput than naive implementations

## Further Reading

- OpenAI, Anthropic, Google, and DeepSeek API documentation
- vLLM and TGI documentation for self-hosted serving
