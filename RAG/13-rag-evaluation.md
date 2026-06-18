# Chapter 13: RAG Evaluation

> **Last verified: June 2026.**

RAG evaluation is critical for enterprise adoption. Without rigorous evaluation, you cannot know if your system is working, improving, or degrading. This chapter covers evaluation frameworks, metrics, and testing strategies.

## Measuring Retrieval Quality

Retrieval quality is measured independently from generation quality. The four key metrics:

**Precision@K** — what fraction of retrieved documents are relevant. High precision means the model sees relevant information, not noise.

**Recall@K** — what fraction of relevant documents are retrieved. High recall means few relevant documents are missed.

**MRR** — where the first relevant document appears. Higher MRR means relevant results surface earlier.

**Hit Rate** — what percentage of queries retrieve at least one relevant document. This is the basic quality bar.

These metrics require a ground truth dataset: questions with known relevant documents. Creating this dataset is an investment, but it is essential for measuring and improving quality.

## Measuring Generation Quality

**Faithfulness** — is the response grounded in the retrieved context? The model should only make claims supported by the documents. Unfaithful responses are hallucinations.

**Relevancy** — does the response address the query? A response can be faithful (all claims are in the context) but irrelevant (it does not answer the question).

**Answer Correctness** — is the response factually correct? This requires comparison against expected answers.

## Evaluation Frameworks

### RAGAS

The standard framework for RAG evaluation. It measures faithfulness, answer relevancy, context precision, and context recall using LLM-as-judge. The output is a score for each metric, enabling quantitative comparison across model versions, retrieval strategies, and prompt changes.

### DeepEval

Broader LLM evaluation with metrics for faithfulness, relevancy, hallucination, and toxicity. Integrates with CI/CD for automated evaluation on every deployment.

### LangSmith

Observability and evaluation for LangChain applications. Traces requests, measures quality, and enables dataset-based evaluation.

## Building an Evaluation Pipeline

The evaluation pipeline runs continuously. A golden dataset of 100-500 representative questions with expected answers is the foundation. On each deployment, the pipeline runs all questions through the RAG system, measures retrieval and generation quality, compares against baseline scores, and alerts if any metric drops significantly.

This catches regressions before they reach users. The investment in building the pipeline pays for itself the first time it catches a quality degradation caused by a chunking change, embedding model upgrade, or prompt modification.

## Key Takeaways

- Build evaluation pipelines before deploying — measure quality from day one
- RAGAS is the standard for RAG evaluation
- Create ground truth datasets — they are essential for regression testing
- Measure retrieval quality separately from generation quality
- LLM-as-judge is cost-effective but imperfect — combine with human evaluation for critical applications

## Further Reading

- RAGAS and DeepEval documentation
