# Chapter 13: Structured Outputs and Reliability

LLMs produce text, but applications need structured data. JSON responses, consistent schemas, validated outputs — these are requirements for production systems. Structured outputs bridge the gap between probabilistic text generation and deterministic data processing.

## JSON Generation

The simplest approach is JSON mode — telling the model to respond in JSON only. OpenAI's `response_format={"type": "json_object"}` achieves this. But JSON mode ensures valid JSON, not correct JSON. The structure might not match your expected schema.

Schema validation with Pydantic adds the next layer. Define your expected output as a Pydantic model, use the model's structured output feature, and the response is validated against your schema. GPT-5.4 with `response_format` using a Pydantic model achieves 99 percent schema adherence.

Without structured output features, JSON generation succeeds only 85 percent of the time. The 14 percent failure rate means one in seven requests produces invalid output. In production, this is unacceptable.

## Reliability Patterns

### Retry with Backoff

The most basic reliability pattern: if structured output fails, retry with error context. Add the error message to the conversation so the model knows what went wrong. Implement exponential backoff to avoid overwhelming the provider.

Most failures are transient — the model generated slightly malformed JSON, or the schema validation caught a type error. One retry is usually sufficient. Three retries handle almost all cases.

### Multi-Model Fallback

If the primary model fails, try alternative models. GPT-5.4 is the primary, Claude Sonnet 4.6 is the fallback, GPT-5.4 mini is the last resort. Each model has different failure modes, so trying multiple models catches different types of errors.

### Validation Pipeline

Multi-layer validation catches errors that schema validation misses. Layer 1: JSON parsing (is it valid JSON?). Layer 2: Schema validation (does it match the Pydantic model?). Layer 3: Business rules (does the data make sense in context?). Layer 4: Quality checks (are fields complete and reasonable?).

## Confidence Scoring

Ask the model to assess its own confidence. Low-confidence outputs are flagged for human review. This is particularly important for high-stakes applications — financial data extraction, medical tripping, legal document processing.

The confidence assessment adds one additional model call per request, but it prevents bad data from flowing into downstream systems.

## Case Study: Financial Data Extraction

A fintech company extracted structured data from financial documents. Initial schema compliance was 85 percent, causing downstream failures. The fix: multi-model retry (primary GPT-5.4, fallback Claude Sonnet 4.6), Pydantic validation with business rules, confidence scoring for flagging uncertain extractions, and human review for low-confidence results.

Schema compliance improved to 99.5 percent, data accuracy improved from 92 to 98 percent, and manual review rate dropped from 15 to 2 percent. Processing cost increased 40 percent (from $0.05 to $0.07 per document), but the cost of bad data in downstream systems far exceeded this.

## Key Takeaways

- Structured output (response_format + Pydantic) achieves 99 percent schema adherence — always validate
- Multi-layer validation catches errors that schema validation misses
- Retry with fallback models is essential — no single model is 100 percent reliable
- Confidence scoring enables human-in-the-loop — flag low-confidence outputs for review
- The cost of validation is small compared to the cost of bad data in downstream systems

## Further Reading

- OpenAI Structured Outputs Guide
- Pydantic Documentation
- Instructor Library — structured output for LLMs
