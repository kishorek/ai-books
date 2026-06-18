# Chapter 14: Evaluation

You cannot improve what you cannot measure. GenAI evaluation is fundamentally different from traditional software testing — outputs are probabilistic, correctness is often subjective, and quality depends on context. This chapter covers evaluation frameworks, metrics, and testing strategies.

## LLM Evaluation Metrics

**Accuracy** measures the percentage of correct responses. Simple to compute for factual tasks, harder for open-ended generation.

**Hallucination rate** measures the frequency of fabricated information. Requires a fact-checking layer that verifies each claim against source documents. High hallucination rates indicate the model is not properly grounded in context.

**Relevance** measures how well responses address the query. An LLM-as-judge can score relevance on a 0-1 scale by comparing the response against the query and expected answer.

**Faithfulness** measures how well responses are grounded in provided context. The model should only make claims supported by the retrieved documents. Faithfulness scoring checks each claim against the context.

## Evaluation Frameworks

### RAGAS

RAGAS is the standard framework for RAG evaluation. It measures faithfulness (is the response grounded in context?), answer relevancy (does the response address the query?), context precision (are the retrieved documents relevant?), and context recall (are the relevant documents retrieved?).

### DeepEval

DeepEval provides broader LLM evaluation with metrics for faithfulness, relevancy, hallucination, and toxicity. It integrates well with CI/CD pipelines for automated evaluation.

### LangSmith

LangSmith provides observability and evaluation for LangChain applications. It traces requests, measures quality, and enables dataset-based evaluation.

## Testing Strategies

### Unit Testing

Test individual components: chunking produces expected output, embedding produces correct dimensions, vector search returns relevant results.

### Integration Testing

Test component interactions: end-to-end RAG pipeline produces coherent responses, tool calling returns expected results, streaming works correctly.

### Prompt Testing

Test prompt variations: compare quality across different system prompts, few-shot examples, and output format instructions. The best prompt is the one that produces the highest quality for your specific use case.

### Regression Testing

Ensure changes do not degrade quality. Maintain a golden dataset of questions with expected answers. Run it on every deployment. Alert if quality drops below baseline.

## The Evaluation Pipeline

The evaluation pipeline runs continuously. A golden dataset of 100-500 representative questions with expected answers is the foundation. On each deployment, the pipeline runs all questions, measures retrieval quality (precision, recall, MRR) and generation quality (faithfulness, relevancy, accuracy), compares against baseline scores, and alerts if any metric drops significantly.

This catches regressions before they reach users. The investment in building the evaluation pipeline pays for itself the first time it catches a quality degradation.

## Key Takeaways

- Evaluation is not optional — build evaluation pipelines before deploying to production
- RAGAS is the standard for RAG evaluation — faithfulness, relevancy, precision, recall
- LLM-as-judge is cost-effective but imperfect — combine with human evaluation for critical applications
- Regression testing prevents quality degradation — run golden dataset tests on every deployment
- Measure retrieval quality separately from generation quality

## Further Reading

- RAGAS and DeepEval documentation
- TruLens and LangSmith documentation
