# Chapter 14: Evaluation

> "You cannot improve what you cannot measure. And in GenAI, measurement is not a one-time act—it is a continuous discipline that determines whether your system is getting better or silently degrading."

---

## Introduction

Traditional software testing has clear success criteria: a function either returns the correct output or it does not. A test passes or fails. A bug is reproducible. GenAI evaluation is fundamentally different. Outputs are probabilistic—running the same prompt twice may produce different (but equally valid) results. Correctness is often subjective—there is no single "right" answer for "write a marketing email." Quality depends on context—a response that is excellent for a casual user may be inadequate for an expert. And the ground truth itself may be ambiguous—different human evaluators may disagree on what constitutes a "good" response.

This chapter addresses the evaluation crisis in GenAI. Without rigorous evaluation, you are deploying systems blind—hoping they work, unable to prove they do, and unable to detect when they degrade. The cost of evaluation is small compared to the cost of deploying a system that hallucinates financial data, misclassifies medical symptoms, or generates offensive content.

The central thesis of this chapter is the **evaluation-as-infrastructure principle**: evaluation is not a phase in development—it is infrastructure that runs continuously, catches regressions, and provides the data needed to improve. Build your evaluation pipeline before you deploy to production, not after you discover quality problems.

We will examine the fundamental metrics for LLM evaluation (accuracy, hallucination rate, relevance, faithfulness). We will dissect the major evaluation frameworks (RAGAS, DeepEval, LangSmith) and their trade-offs. We will build testing strategies for unit, integration, prompt, and regression testing. We will walk through a full case study: building an evaluation pipeline for a RAG system that catches quality regressions before they reach users. And we will cover the hard-won lessons of production evaluation—LLM-as-judge calibration, dataset curation, and the metrics that actually matter.

### The Evaluation Spectrum

Evaluation approaches exist on a spectrum from automated to human:

| Approach | Cost | Accuracy | Speed | Scalability | Best For |
|----------|------|---------|-------|-------------|----------|
| **Automated metrics** | Low | Medium | Fast | High | Retrieval quality, schema compliance |
| **LLM-as-judge** | Medium | Medium-High | Medium | High | Relevance, faithfulness, tone |
| **Human evaluation** | High | High | Slow | Low | Nuance, creativity, safety |
| **A/B testing** | Medium | High | Slow | Medium | User preference, conversion |
| **Hybrid** | Medium | High | Medium | High | Production evaluation |

Most production systems use a hybrid approach: automated metrics for continuous monitoring, LLM-as-judge for quality scoring, and periodic human evaluation for calibration and edge cases.

---

## 14.1 LLM Evaluation Metrics

### 14.1.1 Accuracy Metrics

**Accuracy** measures the percentage of correct responses. Simple to compute for factual tasks, harder for open-ended generation.

```python
class AccuracyEvaluator:
    def __init__(self, llm):
        self.llm = llm

    def evaluate_exact_match(self, predicted: str, expected: str) -> bool:
        return predicted.strip().lower() == expected.strip().lower()

    def evaluate_semantic_match(self, predicted: str, expected: str) -> float:
        prompt = f"""Rate how semantically similar these two responses are.
Response 1: {predicted}
Response 2: {expected}

Rate from 0.0 (completely different) to 1.0 (semantically identical).
Return only the numeric score."""
        score = self.llm.generate(prompt)
        return float(score)

    def evaluate_factual_accuracy(self, response: str, source_documents: list[str]) -> float:
        prompt = f"""Verify each claim in the response against the source documents.

Response: {response}

Source documents:
{chr(10).join(source_documents)}

For each claim, determine if it is:
- SUPPORTED: directly stated or clearly implied in sources
- NOT VERIFIED: cannot be confirmed or denied by sources
- CONTRADICTED: contradicted by sources

Return JSON: {{"supported": N, "not_verified": N, "contradicted": N, "accuracy_score": 0.0-1.0}}"""
        result = self.llm.generate(prompt, response_format="json")
        return result["accuracy_score"]
```

### 14.1.2 Hallucination Detection

**Hallucination rate** measures the frequency of fabricated information. Requires a fact-checking layer that verifies each claim against source documents.

```python
class HallucinationDetector:
    def __init__(self, llm):
        self.llm = llm

    def detect_hallucinations(self, response: str,
                              context: list[str]) -> dict:
        prompt = f"""Analyze this response for hallucinations.

Response: {response}

Context documents:
{chr(10).join(context)}

Identify each factual claim in the response. For each claim:
1. Quote the exact claim
2. Determine if it is: SUPPORTED, NOT VERIFIED, or CONTRADICTED by the context
3. If CONTRADICTED, quote the contradicting passage

Return JSON:
{{
  "claims": [
    {{
      "claim": "...",
      "status": "SUPPORTED|NOT_VERIFIED|CONTRADICTED",
      "evidence": "..."
    }}
  ],
  "hallucination_rate": 0.0-1.0,
  "total_claims": N,
  "supported": N,
  "not_verified": N,
  "contradicted": N
}}"""
        return self.llm.generate(prompt, response_format="json")

    def calculate_hallucination_score(self, detection_result: dict) -> float:
        total = detection_result["total_claims"]
        if total == 0:
            return 0.0
        hallucinated = detection_result["contradicted"] + detection_result["not_verified"]
        return hallucinated / total
```

### 14.1.3 Relevance and Faithfulness

**Relevance** measures how well responses address the query. **Faithfulness** measures how well responses are grounded in provided context.

```python
class RelevanceEvaluator:
    def __init__(self, llm):
        self.llm = llm

    def evaluate_relevance(self, query: str, response: str) -> float:
        prompt = f"""Rate how well this response addresses the query.

Query: {query}
Response: {response}

Rate from 0.0 (completely irrelevant) to 1.0 (perfectly addresses the query).
Consider: Does the response answer the question? Is it on-topic? Is it complete?
Return only the numeric score."""
        return float(self.llm.generate(prompt))

    def evaluate_faithfulness(self, response: str, context: list[str]) -> dict:
        prompt = f"""Check if every claim in the response is supported by the context.

Response: {response}

Context:
{chr(10).join(context)}

For each sentence in the response:
1. Is it fully supported by the context? (faithful)
2. Is it partially supported? (partially faithful)
3. Is it unsupported or contradicted? (hallucination)

Return JSON:
{{
  "faithful_sentences": N,
  "partially_faithful": N,
  "hallucinated": N,
  "faithfulness_score": 0.0-1.0
}}"""
        return self.llm.generate(prompt, response_format="json")
```

### 14.1.4 Metrics Summary

| Metric | What It Measures | Range | Target |
|--------|-----------------|-------|--------|
| Accuracy | Correct responses / total | 0-1 | >0.90 |
| Hallucination rate | Fabricated claims / total claims | 0-1 | <0.05 |
| Relevance | Query-response alignment | 0-1 | >0.85 |
| Faithfulness | Context-grounded claims | 0-1 | >0.90 |
| Completeness | Coverage of expected answer | 0-1 | >0.80 |
| Coherence | Logical flow and clarity | 0-1 | >0.85 |

---

## 14.2 Evaluation Frameworks

### 14.2.1 RAGAS: RAG Evaluation

RAGAS is the standard framework for RAG evaluation. It measures four dimensions: faithfulness, answer relevancy, context precision, and context recall.

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset

# Prepare evaluation dataset
eval_data = {
    "question": [
        "What is the revenue growth rate?",
        "Who is the CEO?",
        "What are the main risk factors?"
    ],
    "answer": [
        "Revenue grew 12% year-over-year to $394 billion.",
        "Tim Cook is the CEO of Apple Inc.",
        "Main risks include supply chain disruption and regulatory changes."
    ],
    "contexts": [
        ["Revenue increased from $352B to $394B, representing 12% growth."],
        ["Tim Cook has served as CEO since August 2011."],
        ["Key risks: supply chain, regulatory, competition, currency fluctuations."]
    ],
    "ground_truth": [
        "12% YoY growth to $394B",
        "Tim Cook",
        "Supply chain, regulatory, competition, currency"
    ]
}

dataset = Dataset.from_dict(eval_data)

# Run evaluation
results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)

print(results)
# {'faithfulness': 0.95, 'answer_relevancy': 0.92,
#  'context_precision': 0.88, 'context_recall': 0.90}
```

### 14.2.2 DeepEval: Broader LLM Evaluation

DeepEval provides metrics beyond RAG, including hallucination, toxicity, bias, and summarization quality.

```python
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    HallucinationMetric,
    ToxicityMetric
)

# Create test cases
test_case = LLMTestCase(
    input="What are the Q4 financial results?",
    actual_output="Q4 revenue was $10.2B, up 15% from last year.",
    retrieval_context=["Q4 2024: Revenue $10.2B, +15% YoY"],
    expected_output="$10.2B revenue, 15% growth"
)

# Define metrics
faithfulness = FaithfulnessMetric(threshold=0.8)
relevancy = AnswerRelevancyMetric(threshold=0.7)
hallucination = HallucinationMetric(threshold=0.3)
toxicity = ToxicityMetric(threshold=0.5)

# Evaluate
results = evaluate(
    test_cases=[test_case],
    metrics=[faithfulness, relevancy, hallucination, toxicity]
)
```

### 14.2.3 LangSmith: Observability and Evaluation

LangSmith provides tracing, evaluation, and monitoring for LangChain applications.

```python
from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

# Define evaluation function
def correctness(run, example):
    """Compare LLM output to expected answer."""
    prediction = run.outputs.get("output", "")
    expected = example.outputs.get("answer", "")
    # Use LLM to judge correctness
    score = llm_judge(prediction, expected)
    return {"key": "correctness", "score": score}

# Run evaluation on dataset
results = evaluate(
    target=your_rag_chain,
    data="financial-qa-dataset",
    evaluators=[correctness],
    experiment_prefix="v2.1-evaluation"
)
```

### 14.2.4 Framework Comparison

| Feature | RAGAS | DeepEval | LangSmith |
|---------|-------|----------|-----------|
| Primary focus | RAG evaluation | General LLM evaluation | Observability + evaluation |
| Metrics | Faithfulness, relevance, context | Hallucination, toxicity, bias | Custom evaluators |
| Dataset support | HuggingFace datasets | Pytest-style | Built-in datasets |
| CI/CD integration | Limited | Good (pytest) | Good (API) |
| Cost | Free (open source) | Free (open source) | Free tier + paid |
| Best for | RAG pipeline evaluation | Comprehensive quality testing | Production monitoring |

---

## 14.3 Testing Strategies

### 14.3.1 Unit Testing

Test individual components: chunking produces expected output, embedding produces correct dimensions, vector search returns relevant results.

```python
import pytest

def test_chunking_produces_expected_output():
    chunker = SemanticChunker(chunk_size=500, overlap=50)
    chunks = chunker.chunk("Long document text...")
    assert len(chunks) > 0
    assert all(len(c.text) <= 600 for c in chunks)  # Allow some overshoot

def test_embedding_produces_correct_dimensions():
    embedder = OpenAIEmbeddings()
    embedding = embedder.embed_query("test text")
    assert len(embedding) == 1536  # OpenAI embedding dimension

def test_vector_search_returns_relevant_results():
    store = VectorStore(documents=test_docs)
    results = store.search("revenue growth", top_k=3)
    assert len(results) == 3
    assert any("revenue" in r.text.lower() for r in results)

def test_prompt_template_produces_valid_output():
    prompt = ChatPromptTemplate.from_template("Answer: {question}")
    formatted = prompt.format(question="What is revenue?")
    assert "What is revenue?" in formatted
```

### 14.3.2 Integration Testing

Test component interactions: end-to-end RAG pipeline produces coherent responses, tool calling returns expected results.

```python
def test_end_to_end_rag_pipeline():
    pipeline = RAGPipeline(vector_store=test_store, llm=test_llm)
    result = pipeline.query("What was the Q4 revenue?")
    assert result["answer"] != ""
    assert len(result["sources"]) > 0
    assert result["confidence"] > 0.5

def test_tool_calling_returns_expected_results():
    agent = ToolUsingAgent(tools=[mock_search_tool])
    result = agent.run("Search for latest revenue data")
    assert "search_results" in result

def test_streaming_works_correctly():
    pipeline = RAGPipeline(vector_store=test_store, llm=test_llm)
    chunks = list(pipeline.stream_query("Summarize the report"))
    assert len(chunks) > 0
    assert all(isinstance(c, str) for c in chunks)
```

### 14.3.3 Prompt Testing

Test prompt variations: compare quality across different system prompts, few-shot examples, and output format instructions.

```python
class PromptEvaluator:
    def __init__(self, llm, eval_dataset):
        self.llm = llm
        self.dataset = eval_dataset

    def evaluate_prompt(self, prompt_template: str,
                        num_samples: int = 50) -> dict:
        scores = {"relevance": [], "faithfulness": [], "accuracy": []}

        for sample in self.dataset[:num_samples]:
            response = self.llm.generate(
                prompt_template.format(**sample["inputs"])
            )

            scores["relevance"].append(
                self._score_relevance(response, sample["expected"])
            )
            scores["faithfulness"].append(
                self._score_faithfulness(response, sample["context"])
            )
            scores["accuracy"].append(
                self._score_accuracy(response, sample["expected"])
            )

        return {
            "avg_relevance": sum(scores["relevance"]) / len(scores["relevance"]),
            "avg_faithfulness": sum(scores["faithfulness"]) / len(scores["faithfulness"]),
            "avg_accuracy": sum(scores["accuracy"]) / len(scores["accuracy"]),
            "num_samples": num_samples
        }

    def compare_prompts(self, prompts: dict[str, str]) -> dict:
        results = {}
        for name, template in prompts.items():
            results[name] = self.evaluate_prompt(template)
        return results

# Example usage
prompts = {
    "baseline": "Answer this question: {question}\nContext: {context}",
    "structured": "Based on the context below, answer the question. Cite sources.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:",
    "cot": "Let me think step by step.\n\nContext: {context}\n\nQuestion: {question}\n\nStep-by-step answer:"
}

evaluator = PromptEvaluator(llm, eval_dataset)
results = evaluator.compare_prompts(prompts)
# {'baseline': {'avg_relevance': 0.78, ...},
#  'structured': {'avg_relevance': 0.85, ...},
#  'cot': {'avg_relevance': 0.82, ...}}
```

### 14.3.4 Regression Testing

Ensure changes do not degrade quality. Maintain a golden dataset and run it on every deployment.

```python
class RegressionTester:
    def __init__(self, pipeline, golden_dataset, threshold: float = 0.05):
        self.pipeline = pipeline
        self.dataset = golden_dataset
        self.threshold = threshold
        self.baseline_scores = None

    def set_baseline(self):
        self.baseline_scores = self._run_evaluation()
        return self.baseline_scores

    def check_regression(self) -> dict:
        current_scores = self._run_evaluation()
        regressions = {}

        for metric, current_value in current_scores.items():
            baseline_value = self.baseline_scores.get(metric, 0)
            if baseline_value > 0:
                change = (current_value - baseline_value) / baseline_value
                if change < -self.threshold:
                    regressions[metric] = {
                        "baseline": baseline_value,
                        "current": current_value,
                        "change": change
                    }

        return {
            "regressions": regressions,
            "passed": len(regressions) == 0,
            "current_scores": current_scores
        }

    def _run_evaluation(self) -> dict:
        scores = {"relevance": [], "faithfulness": [], "accuracy": []}
        for sample in self.dataset:
            result = self.pipeline.query(sample["question"])
            scores["relevance"].append(self._score_relevance(result, sample))
            scores["faithfulness"].append(self._score_faithfulness(result, sample))
            scores["accuracy"].append(self._score_accuracy(result, sample))
        return {k: sum(v)/len(v) for k, v in scores.items()}
```

---

## 14.4 The Evaluation Pipeline

### 14.4.1 Architecture

```mermaid
graph TD
    A[Golden Dataset<br/>100-500 questions] --> B[Evaluation Runner]
    B --> C[Retrieval Quality<br/>Precision, Recall, MRR]
    B --> D[Generation Quality<br/>Faithfulness, Relevance]
    B --> E[End-to-End Quality<br/>Accuracy, Completeness]
    C --> F[Score Aggregator]
    D --> F
    E --> F
    F --> G{Regression Check}
    G -->|Pass| H[Deploy]
    G -->|Fail| I[Alert + Block Deploy]
    I --> J[Investigate]
    J --> K[Fix]
    K --> B
```

### 14.4.2 Continuous Evaluation Pipeline

```python
class EvaluationPipeline:
    def __init__(self, pipeline, golden_dataset: list[dict]):
        self.pipeline = pipeline
        self.dataset = golden_dataset
        self.metrics = {
            "retrieval_precision": RetrievalPrecisionEvaluator(),
            "retrieval_recall": RetrievalRecallEvaluator(),
            "faithfulness": FaithfulnessEvaluator(llm),
            "relevance": RelevanceEvaluator(llm),
            "accuracy": AccuracyEvaluator(llm)
        }

    def run_full_evaluation(self) -> dict:
        results = {metric: [] for metric in self.metrics}

        for sample in self.dataset:
            # Run pipeline
            output = self.pipeline.query(sample["question"])

            # Evaluate retrieval quality
            if "sources" in output:
                results["retrieval_precision"].append(
                    self.metrics["retrieval_precision"].evaluate(
                        output["sources"], sample.get("expected_sources", [])
                    )
                )
                results["retrieval_recall"].append(
                    self.metrics["retrieval_recall"].evaluate(
                        output["sources"], sample.get("expected_sources", [])
                    )
                )

            # Evaluate generation quality
            results["faithfulness"].append(
                self.metrics["faithfulness"].evaluate(
                    output["answer"], output.get("context", [])
                )
            )
            results["relevance"].append(
                self.metrics["relevance"].evaluate(
                    sample["question"], output["answer"]
                )
            )
            results["accuracy"].append(
                self.metrics["accuracy"].evaluate(
                    output["answer"], sample.get("expected_answer", "")
                )
            )

        # Aggregate scores
        return {
            metric: {
                "mean": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
                "p50": sorted(scores)[len(scores)//2],
                "p95": sorted(scores)[int(len(scores)*0.95)]
            }
            for metric, scores in results.items()
        }

    def run_ci_check(self, baseline: dict, threshold: float = 0.05) -> dict:
        current = self.run_full_evaluation()
        regressions = []

        for metric in current:
            if metric in baseline:
                change = (current[metric]["mean"] - baseline[metric]["mean"]) / baseline[metric]["mean"]
                if change < -threshold:
                    regressions.append({
                        "metric": metric,
                        "baseline": baseline[metric]["mean"],
                        "current": current[metric]["mean"],
                        "change_pct": change * 100
                    })

        return {
            "passed": len(regressions) == 0,
            "regressions": regressions,
            "current_scores": current
        }
```

### 14.4.3 Golden Dataset Management

```python
class GoldenDatasetManager:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.dataset = self._load()

    def _load(self) -> list[dict]:
        with open(self.filepath) as f:
            return json.load(f)

    def add_sample(self, question: str, expected_answer: str,
                   expected_sources: list[str] = None,
                   category: str = "general"):
        sample = {
            "id": str(uuid.uuid4()),
            "question": question,
            "expected_answer": expected_answer,
            "expected_sources": expected_sources or [],
            "category": category,
            "added_date": datetime.utcnow().isoformat(),
            "difficulty": "medium"
        }
        self.dataset.append(sample)
        self._save()

    def get_samples_by_category(self, category: str) -> list[dict]:
        return [s for s in self.dataset if s["category"] == category]

    def get_samples_by_difficulty(self, difficulty: str) -> list[dict]:
        return [s for s in self.dataset if s["difficulty"] == difficulty]

    def validate_dataset(self) -> dict:
        issues = []
        for sample in self.dataset:
            if not sample.get("question"):
                issues.append(f"Sample {sample['id']} missing question")
            if not sample.get("expected_answer"):
                issues.append(f"Sample {sample['id']} missing expected_answer")
        return {"valid": len(issues) == 0, "issues": issues}

    def _save(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.dataset, f, indent=2)
```

---

## 14.5 LLM-as-Judge

### 14.5.1 Using LLMs for Evaluation

LLM-as-judge uses a powerful model to evaluate the outputs of your application. It is cost-effective for large-scale evaluation but imperfect—LLMs have their own biases and blind spots.

```python
class LLMJudge:
    def __init__(self, judge_model: str = "gpt-4o"):
        self.client = OpenAI()
        self.judge_model = judge_model

    def judge_relevance(self, question: str, answer: str,
                        expected: str = None) -> dict:
        prompt = f"""You are an expert evaluator. Rate this Q&A pair.

Question: {question}
Answer: {answer}
{f'Expected: {expected}' if expected else ''}

Evaluate on these criteria (0-1 scale):
1. Relevance: Does the answer address the question?
2. Completeness: Does the answer cover all important aspects?
3. Accuracy: Is the information correct?
4. Clarity: Is the answer well-structured and clear?

Return JSON: {{"relevance": 0.0, "completeness": 0.0, "accuracy": 0.0, "clarity": 0.0, "overall": 0.0, "feedback": "..."}}"""

        response = self.client.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)

    def judge_faithfulness(self, answer: str,
                           context: list[str]) -> dict:
        prompt = f"""Evaluate if this answer is faithful to the provided context.

Answer: {answer}

Context:
{chr(10).join(context)}

For each claim in the answer:
1. Is it supported by the context?
2. Is it partially supported?
3. Is it unsupported (hallucination)?

Return JSON: {{"faithful_claims": N, "partial_claims": N, "hallucinated_claims": N, "faithfulness_score": 0.0-1.0}}"""

        response = self.client.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)

    def pairwise_comparison(self, question: str, answer_a: str,
                            answer_b: str) -> dict:
        prompt = f"""Compare these two answers to the same question.

Question: {question}
Answer A: {answer_a}
Answer B: {answer_b}

Which answer is better? Consider relevance, accuracy, completeness, and clarity.
Return JSON: {{"winner": "A" or "B" or "tie", "reasoning": "...", "scores": {{"A": 0.0-1.0, "B": 0.0-1.0}}}}"""

        response = self.client.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
```

### 14.5.2 LLM-as-Judge Limitations

| Limitation | Description | Mitigation |
|-----------|-------------|------------|
| Position bias | Prefers first answer in pairwise comparison | Randomize answer order |
| Verbosity bias | Longer answers rated higher | Normalize by length |
| Self-preference | Same model judges its own outputs | Use different model as judge |
| Inconsistency | Different runs produce different scores | Average multiple evaluations |
| Cost | Evaluating 1000 samples costs $5-50 | Use smaller judge models for screening |

---

## 14.6 Case Study: RAG Evaluation Pipeline

### 14.6.1 Problem Statement

A financial research firm built a RAG system for querying SEC filings. The system works well in testing but produces inconsistent quality in production. Some queries return perfectly grounded answers; others hallucinate financial figures. The firm needs a continuous evaluation pipeline that catches quality regressions before they affect users.

**Requirements:**
- Catch quality regressions within 1 hour of deployment
- Evaluate 500+ representative queries
- Measure retrieval quality, generation quality, and end-to-end accuracy
- Block deployments if any metric drops below threshold
- Provide actionable diagnostics for quality issues

### 14.6.2 Implementation

```python
class RAGEvaluationPipeline:
    def __init__(self, rag_pipeline, dataset_path: str):
        self.pipeline = rag_pipeline
        self.dataset = self._load_dataset(dataset_path)
        self.judge = LLMJudge()
        self.baseline = None

    def run_evaluation(self) -> dict:
        results = {
            "retrieval_precision": [],
            "retrieval_recall": [],
            "faithfulness": [],
            "relevance": [],
            "accuracy": []
        }

        for sample in self.dataset:
            output = self.pipeline.query(sample["question"])

            # Retrieval quality
            if "sources" in output and sample.get("expected_sources"):
                precision = self._calculate_precision(
                    output["sources"], sample["expected_sources"]
                )
                recall = self._calculate_recall(
                    output["sources"], sample["expected_sources"]
                )
                results["retrieval_precision"].append(precision)
                results["retrieval_recall"].append(recall)

            # Generation quality
            faithfulness = self.judge.judge_faithfulness(
                output["answer"], output.get("context", [])
            )
            results["faithfulness"].append(faithfulness["faithfulness_score"])

            relevance = self.judge.judge_relevance(
                sample["question"], output["answer"],
                sample.get("expected_answer")
            )
            results["relevance"].append(relevance["overall"])

            # Accuracy
            if sample.get("expected_answer"):
                accuracy = self._compare_answers(
                    output["answer"], sample["expected_answer"]
                )
                results["accuracy"].append(accuracy)

        return self._aggregate_results(results)

    def check_deployment_readiness(self, threshold: float = 0.05) -> dict:
        if not self.baseline:
            self.baseline = self.run_evaluation()

        current = self.run_evaluation()
        issues = []

        for metric in current:
            if metric in self.baseline:
                baseline_val = self.baseline[metric]["mean"]
                current_val = current[metric]["mean"]
                if baseline_val > 0:
                    change = (current_val - baseline_val) / baseline_val
                    if change < -threshold:
                        issues.append({
                            "metric": metric,
                            "baseline": baseline_val,
                            "current": current_val,
                            "change": f"{change*100:.1f}%"
                        })

        return {
            "ready": len(issues) == 0,
            "issues": issues,
            "current_scores": current
        }

    def _calculate_precision(self, retrieved: list, expected: list) -> float:
        if not retrieved:
            return 0.0
        relevant = sum(1 for r in retrieved if r in expected)
        return relevant / len(retrieved)

    def _calculate_recall(self, retrieved: list, expected: list) -> float:
        if not expected:
            return 1.0
        relevant = sum(1 for r in retrieved if r in expected)
        return relevant / len(expected)

    def _compare_answers(self, predicted: str, expected: str) -> float:
        judge_result = self.judge.judge_relevance(
            "", predicted, expected
        )
        return judge_result["accuracy"]

    def _aggregate_results(self, results: dict) -> dict:
        aggregated = {}
        for metric, scores in results.items():
            if scores:
                aggregated[metric] = {
                    "mean": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "std": (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)) ** 0.5,
                    "count": len(scores)
                }
            else:
                aggregated[metric] = {"mean": 0, "count": 0}
        return aggregated
```

### 14.6.3 CI/CD Integration

```yaml
# .github/workflows/eval.yml
name: Evaluation Pipeline
on:
  pull_request:
    paths:
      - 'src/**'
      - 'prompts/**'

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Evaluation
        run: python -m evaluation.run --dataset golden_dataset.json --threshold 0.05
      - name: Check Results
        run: |
          python -c "
          import json
          with open('eval_results.json') as f:
              results = json.load(f)
          if not results['ready']:
              print('Evaluation FAILED:')
              for issue in results['issues']:
                  print(f'  {issue[\"metric\"]}: {issue[\"change\"]} regression')
              exit(1)
          print('Evaluation PASSED')
          "
```

---

## 14.7 Testing Evaluation Systems

### 14.7.1 Unit Testing Evaluators

```python
def test_relevance_evaluator():
    evaluator = RelevanceEvaluator(mock_llm)
    score = evaluator.evaluate_relevance(
        "What is revenue?", "Revenue was $10 billion."
    )
    assert 0.8 <= score <= 1.0

def test_faithfulness_evaluator():
    evaluator = FaithfulnessEvaluator(mock_llm)
    result = evaluator.evaluate_faithfulness(
        "Revenue grew 12%",
        ["Revenue increased from $352B to $394B, representing 12% growth."]
    )
    assert result["faithfulness_score"] > 0.9

def test_hallucination_detector():
    detector = HallucinationDetector(mock_llm)
    result = detector.detect_hallucinations(
        "Revenue was $500 billion",
        ["Revenue was $394 billion"]
    )
    assert result["hallucination_rate"] > 0.5
```

### 14.7.2 Integration Testing

```python
def test_full_evaluation_pipeline():
    pipeline = RAGEvaluationPipeline(mock_rag, "test_dataset.json")
    results = pipeline.run_evaluation()
    assert "faithfulness" in results
    assert results["faithfulness"]["mean"] > 0.7

def test_deployment_readiness_check():
    pipeline = RAGEvaluationPipeline(mock_rag, "test_dataset.json")
    pipeline.baseline = pipeline.run_evaluation()
    readiness = pipeline.check_deployment_readiness()
    assert readiness["ready"] in [True, False]
    assert "issues" in readiness
```

### 14.7.3 Evaluation Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Evaluation coverage | >90% of failure modes | Golden dataset includes edge cases |
| Evaluator accuracy | >85% agreement with human | Periodic human calibration |
| Evaluation latency | <5 minutes for 500 samples | Pipeline execution time |
| Regression detection rate | >95% of real regressions caught | A/B testing validation |
| False positive rate | <5% | Evaluations that block valid deploys |

---

## 14.8 LLM-as-Judge Bias Mitigation at Scale

Using an LLM to evaluate LLM outputs is cost-effective and scalable, but it introduces systematic biases that degrade evaluation reliability. When production pipelines depend on LLM judges for quality gates, unmitigated bias produces silent failures: valid responses get rejected, poor responses pass quality checks, and evaluation scores drift away from human judgment. This section covers detection, quantification, and mitigation of the three dominant bias classes in LLM-as-judge systems, plus a calibration framework for measuring judge reliability.

### 14.8.1 Bias Taxonomy

| Bias Type | Symptom | Severity | Detection Method | Mitigation Strategy |
|-----------|---------|----------|-----------------|---------------------|
| **Position bias** | First (or last) answer in pairwise comparison scores higher | High | Swap answer order, measure score delta | Randomize order, average over both orderings |
| **Length bias** | Longer responses score higher regardless of quality | Medium | Correlate scores with response length | Normalize by length, train adjustment model |
| **Self-preference bias** | Judge LLM favors outputs from same model family | High | Compare cross-model vs. same-model judging | Use different model as judge |
| **Verbosity bias** | More detailed responses score higher than concise correct ones | Medium | Annotate conciseness, check correlation | Prompt instructs judge to penalize unnecessary length |
| **Familiarity bias** | Judge scores known/famous content higher | Low | Test with obscure vs. common topics | Diversify evaluation dataset |
| **Authority bias** | Judge defers to confident-sounding responses | Medium | Compare hedged vs. assertive phrasing | Explicit prompt instructions to ignore tone |

### 14.8.2 Position Bias Detection and Mitigation

Position bias is the most documented bias in pairwise LLM evaluation. The judge assigns higher scores to the first answer presented, regardless of quality. This bias is consistent across most LLM families and can skew pairwise comparison win rates by 10-20%.

```python
import random
from typing import List, Tuple, Dict


class PositionBiasDetector:
    """Detect position bias in pairwise LLM comparisons."""

    def __init__(self, judge_llm):
        self.judge = judge_llm

    def measure_bias(
        self,
        pairs: List[Tuple[str, str]],
        num_samples: int = 100,
    ) -> Dict:
        """
        Measure position bias by comparing scores when answer
        order is preserved vs. reversed.

        Args:
            pairs: List of (answer_a, answer_b) tuples to compare.
            num_samples: Number of random pairs to evaluate.

        Returns:
            Dict with bias magnitude and statistical significance.
        """
        random.seed(42)
        sampled = random.sample(pairs, min(num_samples, len(pairs)))

        forward_wins = 0  # first answer wins in original order
        reverse_wins = 0  # first answer wins in reversed order
        score_deltas_forward = []
        score_deltas_reverse = []

        for a, b in sampled:
            # Original order: A first, B second
            score_a_forward, score_b_forward = self._judge_pair(a, b)
            if score_a_forward > score_b_forward:
                forward_wins += 1
            score_deltas_forward.append(score_a_forward - score_b_forward)

            # Reversed order: B first, A second
            score_a_reverse, score_b_reverse = self._judge_pair(b, a)
            if score_a_reverse > score_b_reverse:
                reverse_wins += 1
            score_deltas_reverse.append(score_b_reverse - score_a_reverse)

        total = len(sampled)
        bias_magnitude = abs(forward_wins - reverse_wins) / total
        avg_delta_forward = sum(score_deltas_forward) / len(score_deltas_forward)
        avg_delta_reverse = sum(score_deltas_reverse) / len(score_deltas_reverse)

        return {
            "bias_magnitude": bias_magnitude,
            "forward_first_advantage": avg_delta_forward,
            "reverse_first_advantage": avg_delta_reverse,
            "sample_size": total,
            "is_significant": bias_magnitude > 0.1,
        }

    def _judge_pair(self, first: str, second: str) -> Tuple[float, float]:
        prompt = f"""Compare these two responses to the same query.
Evaluate quality, accuracy, and helpfulness.

Response 1:
{first}

Response 2:
{second}

Rate each response from 1-10. Return ONLY JSON:
{{"response_1_score": <int>, "response_2_score": <int>}}"""
        result = self.judge.generate(prompt)
        return self._parse_scores(result)

    def _parse_scores(self, response: str) -> Tuple[float, float]:
        import json
        try:
            data = json.loads(response)
            return data["response_1_score"], data["response_2_score"]
        except (json.JSONDecodeError, KeyError):
            return 5.0, 5.0
```

**Mitigation: order averaging.** Evaluate every pair twice—once in each order—and average the scores. This eliminates first-position advantage at the cost of 2x evaluation calls.

```python
class OrderAveragingJudge:
    """Debias pairwise comparisons by averaging over both orderings."""

    def __init__(self, judge_llm):
        self.judge = judge_llm

    def compare(self, answer_a: str, answer_b: str) -> Dict:
        """
        Compare two answers with position bias mitigation.

        Returns debiased scores and the raw per-order results.
        """
        # Forward order: A first, B second
        score_a_f, score_b_f = self._evaluate_pair(answer_a, answer_b)

        # Reverse order: B first, A second
        score_a_r, score_b_r = self._evaluate_pair(answer_b, answer_a)

        # Average scores for each answer across both positions
        debiased_a = (score_a_f + score_a_r) / 2
        debiased_b = (score_b_f + score_b_r) / 2

        return {
            "answer_a_score": debiased_a,
            "answer_b_score": debiased_b,
            "winner": "a" if debiased_a > debiased_b else "b",
            "confidence": abs(debiased_a - debiased_b) / 10,
            "raw": {
                "forward": {"a": score_a_f, "b": score_b_f},
                "reverse": {"a": score_a_r, "b": score_b_r},
            },
        }

    def _evaluate_pair(self, first: str, second: str) -> Tuple[float, float]:
        prompt = f"""Compare these two responses. Rate each 1-10.

Response 1:
{first}

Response 2:
{second}

Return ONLY JSON:
{{"response_1_score": <int>, "response_2_score": <int>}}"""
        result = self.judge.generate(prompt)
        import json
        data = json.loads(result)
        return data["response_1_score"], data["response_2_score"]
```

### 14.8.3 Length Bias Detection and Mitigation

LLM judges tend to conflate response length with quality. A 500-word answer that repeats itself scores higher than a 100-word answer that directly addresses the query. This bias is particularly damaging for applications that value conciseness—customer support, code generation, and summarization.

```python
import numpy as np
from typing import List, Dict


class LengthBiasDetector:
    """Detect correlation between response length and judge scores."""

    def __init__(self, judge_llm):
        self.judge = judge_llm

    def measure_bias(
        self,
        samples: List[Dict],
    ) -> Dict:
        """
        Measure length bias by correlating response length with judge scores.

        Args:
            samples: List of {"response": str, "reference_quality": float}
                     where reference_quality is the human-rated quality score.

        Returns:
            Dict with correlation coefficient and length-effect size.
        """
        lengths = []
        judge_scores = []
        human_scores = []

        for sample in samples:
            response = sample["response"]
            length = len(response.split())
            score = self._score_response(response)
            human = sample["reference_quality"]

            lengths.append(length)
            judge_scores.append(score)
            human_scores.append(human)

        # Pearson correlation between length and judge score
        length_correlation = np.corrcoef(lengths, judge_scores)[0, 1]

        # Partial correlation: length vs. judge score, controlling for quality
        quality_residuals = np.array(human_scores) - np.mean(human_scores)
        judge_residuals = np.array(judge_scores) - np.mean(judge_scores)
        length_arr = np.array(lengths) - np.mean(lengths)

        # Regress out quality, measure residual length correlation
        from numpy.linalg import lstsq
        beta_judge, _, _, _ = lstsq(
            quality_residuals.reshape(-1, 1), judge_residuals, rcond=None
        )
        judge_adjusted = judge_residuals - beta_judge * quality_residuals

        residual_correlation = np.corrcoef(length_arr, judge_adjusted)[0, 1]

        return {
            "length_vs_judge_correlation": round(float(length_correlation), 3),
            "length_vs_quality_residual": round(float(residual_correlation), 3),
            "sample_size": len(samples),
            "is_biased": abs(length_correlation) > 0.3,
        }

    def _score_response(self, response: str) -> float:
        prompt = f"""Rate the quality of this response from 1-10.

Response:
{response}

Return ONLY: {{"score": <int>}}"""
        import json
        result = self.judge.generate(prompt)
        return json.loads(result)["score"]
```

**Mitigation: length-normalized scoring.** Two approaches work in practice. First, include explicit length instructions in the judge prompt. Second, train a length-adjustment model that corrects scores based on observed length effects.

```python
class LengthNormalizedJudge:
    """Judge with length bias mitigation."""

    def __init__(self, judge_llm, length_penalty_weight: float = 0.02):
        self.judge = judge_llm
        self.length_penalty = length_penalty_weight

    def score(self, response: str, reference: str = None) -> Dict:
        """Score a response with length normalization."""
        raw_score = self._raw_judge_score(response)
        word_count = len(response.split())

        # Normalize: penalize responses significantly longer than reference
        if reference:
            ref_length = len(reference.split())
            length_ratio = word_count / max(ref_length, 1)
            # Penalty grows quadratically for responses >2x reference length
            penalty = max(0, (length_ratio - 1.0) ** 2) * self.length_penalty
        else:
            # Absolute length penalty for responses over 300 words
            penalty = max(0, (word_count - 300) / 1000) * self.length_penalty

        normalized_score = max(1.0, raw_score * (1 - penalty))

        return {
            "raw_score": raw_score,
            "normalized_score": round(normalized_score, 2),
            "word_count": word_count,
            "penalty_applied": round(penalty, 4),
        }

    def _raw_judge_score(self, response: str) -> float:
        prompt = f"""Rate this response from 1-10. Do NOT give higher scores
for longer responses. Only assess quality and accuracy.

Response:
{response}

Return ONLY: {{"score": <int>}}"""
        import json
        result = self.judge.generate(prompt)
        return json.loads(result)["score"]
```

### 14.8.4 Self-Preference Bias

Self-preference bias occurs when an LLM judge rates outputs from its own model family higher than outputs from other models. Research shows GPT-4 rates GPT-4 outputs 10-15% higher than equivalent Claude outputs, and vice versa. The mitigation is straightforward: never use a model to evaluate its own outputs.

```python
class CrossModelJudge:
    """Mitigate self-preference bias by using a different model as judge."""

    def __init__(self, judge_llm, target_model_name: str):
        """
        Args:
            judge_llm: The LLM used for judging.
            target_model_name: Name of the model being evaluated.
                               Must differ from judge model.
        """
        self.judge = judge_llm
        self.target_model = target_model_name
        self._validate_cross_model()

    def _validate_cross_model(self):
        judge_name = getattr(self.judge, "model_name", "unknown")
        if judge_name == self.target_model:
            raise ValueError(
                f"Judge model '{judge_name}' matches target model "
                f"'{self.target_model}'. Use a different model as judge "
                f"to avoid self-preference bias."
            )

    def evaluate(self, prompt: str, response: str) -> Dict:
        """Evaluate a response using a cross-model judge."""
        eval_prompt = f"""Rate this AI response from 1-10 on quality,
accuracy, and helpfulness.

Original Prompt:
{prompt}

AI Response:
{response}

Return ONLY JSON:
{{"score": <int>, "reasoning": "<brief explanation>"}}"""
        import json
        result = self.judge.generate(eval_prompt)
        parsed = json.loads(result)

        return {
            "score": parsed["score"],
            "reasoning": parsed.get("reasoning", ""),
            "judge_model": getattr(self.judge, "model_name", "unknown"),
            "target_model": self.target_model,
            "cross_model": True,
        }
```

Production systems should rotate judges across model families and track which judge evaluated which response. When comparing across model generations, use at least two independent judges and average their scores.

```python
class RotatingJudgePool:
    """Pool of cross-model judges for unbiased evaluation."""

    def __init__(self, judges: Dict[str, object], target_model: str):
        self.judges = judges  # {"gpt4": llm, "claude": llm, ...}
        self.target_model = target_model
        self._remove_self_judges()

    def _remove_self_judges(self):
        if self.target_model in self.judges:
            del self.judges[self.target_model]

    def evaluate(self, prompt: str, response: str, num_judges: int = 2) -> Dict:
        """Evaluate with multiple judges and average results."""
        import random
        selected = random.sample(
            list(self.judges.keys()),
            min(num_judges, len(self.judges)),
        )

        scores = []
        for judge_name in selected:
            judge = self.judges[judge_name]
            evaluator = CrossModelJudge(judge, self.target_model)
            result = evaluator.evaluate(prompt, response)
            scores.append(result["score"])

        return {
            "final_score": sum(scores) / len(scores),
            "individual_scores": dict(zip(selected, scores)),
            "num_judges": len(selected),
            "score_variance": np.var(scores) if len(scores) > 1 else 0,
        }
```

### 14.8.5 Calibration Framework

A judge is only useful if its scores correlate with human judgment. Calibration measures this correlation and quantifies the expected error. The key metric is **Expected Calibration Error (ECE)**: the average gap between the judge's predicted confidence and its actual accuracy across score buckets.

```python
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class CalibrationResult:
    ece: float  # Expected Calibration Error
    accuracy_by_bin: Dict[str, float]
    confidence_by_bin: Dict[str, float]
    calibration_curve: List[Tuple[float, float]]
    sample_size: int


class JudgeCalibrationFramework:
    """Calibrate LLM judge scores against human ratings."""

    def __init__(self, num_bins: int = 10):
        self.num_bins = num_bins

    def calibrate(
        self,
        judge_scores: List[float],
        human_scores: List[float],
    ) -> CalibrationResult:
        """
        Compute calibration metrics for judge scores.

        Args:
            judge_scores: Scores produced by the LLM judge (1-10 scale).
            human_scores: Ground-truth human ratings (1-10 scale).

        Returns:
            CalibrationResult with ECE, bin-level accuracy, and curve.
        """
        judge_arr = np.array(judge_scores)
        human_arr = np.array(human_scores)

        # Normalize to 0-1 range for binning
        judge_norm = (judge_arr - 1) / 9
        human_norm = (human_arr - 1) / 9

        # Define bins: [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]
        bin_edges = np.linspace(0, 1, self.num_bins + 1)
        bin_indices = np.digitize(judge_norm, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)

        # Compute per-bin metrics
        bin_counts = []
        bin_accuracies = []
        bin_confidences = []
        calibration_curve = []

        for b in range(self.num_bins):
            mask = bin_indices == b
            count = mask.sum()
            bin_counts.append(count)

            if count == 0:
                bin_accuracies.append(0)
                bin_confidences.append(0)
                continue

            bin_judge = judge_norm[mask]
            bin_human = human_norm[mask]

            # Accuracy: fraction of judge scores within 0.1 of human score
            accuracy = np.mean(np.abs(bin_judge - bin_human) < 0.1)
            confidence = bin_judge.mean()

            bin_accuracies.append(float(accuracy))
            bin_confidences.append(float(confidence))
            calibration_curve.append((float(confidence), float(accuracy)))

        # ECE: weighted average of |confidence - accuracy| across bins
        total = sum(bin_counts)
        ece = 0
        for b in range(self.num_bins):
            if bin_counts[b] > 0:
                weight = bin_counts[b] / total
                ece += weight * abs(bin_confidences[b] - bin_accuracies[b])

        # Per-bin labels
        accuracy_by_bin = {}
        confidence_by_bin = {}
        for b in range(self.num_bins):
            label = f"{bin_edges[b]:.1f}-{bin_edges[b+1]:.1f}"
            accuracy_by_bin[label] = bin_accuracies[b]
            confidence_by_bin[label] = bin_confidences[b]

        return CalibrationResult(
            ece=round(float(ece), 4),
            accuracy_by_bin=accuracy_by_bin,
            confidence_by_bin=confidence_by_bin,
            calibration_curve=calibration_curve,
            sample_size=len(judge_scores),
        )

    def correlation_report(
        self,
        judge_scores: List[float],
        human_scores: List[float],
    ) -> Dict:
        """Compute correlation metrics between judge and human scores."""
        judge_arr = np.array(judge_scores)
        human_arr = np.array(human_scores)

        pearson_r = float(np.corrcoef(judge_arr, human_arr)[0, 1])

        # Spearman rank correlation
        from scipy.stats import spearmanr
        spearman_r, spearman_p = spearmanr(judge_arr, human_arr)

        # Mean absolute error
        mae = float(np.mean(np.abs(judge_arr - human_arr)))

        # Percentage within 1 point of human score
        within_one = float(np.mean(np.abs(judge_arr - human_arr) <= 1.0))

        return {
            "pearson_correlation": round(pearson_r, 3),
            "spearman_correlation": round(float(spearman_r), 3),
            "spearman_p_value": float(spearman_p),
            "mean_absolute_error": round(mae, 3),
            "within_one_point_pct": round(within_one * 100, 1),
            "sample_size": len(judge_scores),
        }

    def should_trust_judge(self, calibration_result: CalibrationResult) -> Dict:
        """Determine if the judge is reliable enough for production use."""
        report = self.correlation_report(
            list(calibration_result.confidence_by_bin.values()),
            list(calibration_result.accuracy_by_bin.values()),
        )

        trustworthy = (
            calibration_result.ece < 0.15
            and report["pearson_correlation"] > 0.7
        )

        return {
            "trustworthy": trustworthy,
            "ece": calibration_result.ece,
            "pearson_r": report["pearson_correlation"],
            "recommendation": (
                "Judge is calibrated for production use"
                if trustworthy
                else "Recalibrate or supplement with human evaluation"
            ),
        }
```

**Calibration workflow.** Run the calibration framework quarterly or after any judge model update. Maintain a gold-standard dataset of 200-500 samples with human ratings. Track ECE over time—if it rises above 0.15, the judge needs recalibration or replacement.

```python
# Calibration pipeline
calibrator = JudgeCalibrationFramework(num_bins=10)

# Load gold-standard dataset with human ratings
judge_scores = load_judge_scores("judge_output.json")
human_scores = load_human_ratings("human_ratings.json")

# Compute calibration
calibration = calibrator.calibrate(judge_scores, human_scores)
correlation = calibrator.correlation_report(judge_scores, human_scores)
verdict = calibrator.should_trust_judge(calibration)

print(f"ECE: {calibration.ece}")
print(f"Pearson r: {correlation['pearson_correlation']}")
print(f"Trustworthy: {verdict['trustworthy']}")
print(f"Recommendation: {verdict['recommendation']}")

# If ECE > 0.15, investigate per-bin errors
if calibration.ece > 0.15:
    for bin_label, acc in calibration.accuracy_by_bin.items():
        conf = calibration.confidence_by_bin[bin_label]
        gap = abs(conf - acc)
        if gap > 0.2:
            print(f"  Problem bin {bin_label}: conf={conf:.2f}, acc={acc:.2f}")
```

### 14.8.6 Production Bias Monitoring

In production, bias monitoring runs continuously alongside evaluation. The `BiasAwareJudge` class integrates all mitigation strategies into a single interface and logs bias metrics for ongoing monitoring.

```python
class BiasAwareJudge:
    """
    Production-grade LLM judge with integrated bias mitigation.

    Combines order averaging, length normalization, cross-model
    judging, and calibration into a single evaluation interface.
    """

    def __init__(
        self,
        judge_pool: Dict[str, object],
        target_model: str,
        calibration_data: Dict = None,
        length_penalty_weight: float = 0.02,
    ):
        self.pool = RotatingJudgePool(judge_pool, target_model)
        self.length_normalizer = LengthNormalizedJudge(
            next(iter(judge_pool.values())),
            length_penalty_weight,
        )
        self.calibrator = JudgeCalibrationFramework()
        self.calibration_data = calibration_data
        self.metrics_log = []

    def evaluate(self, prompt: str, response: str, reference: str = None) -> Dict:
        """
        Evaluate a response with full bias mitigation.

        Steps:
        1. Cross-model judging (self-preference mitigation)
        2. Length normalization (length bias mitigation)
        3. Log metrics for monitoring
        """
        # Cross-model evaluation
        cross_result = self.pool.evaluate(prompt, response, num_judges=2)

        # Length normalization
        length_result = self.length_normalizer.score(response, reference)

        # Weighted combination
        final_score = (
            0.7 * cross_result["final_score"]
            + 0.3 * length_result["normalized_score"]
        )

        result = {
            "final_score": round(final_score, 2),
            "cross_model_score": cross_result["final_score"],
            "length_normalized_score": length_result["normalized_score"],
            "judges_used": list(cross_result["individual_scores"].keys()),
            "score_variance": cross_result["score_variance"],
            "length_penalty": length_result["penalty_applied"],
        }

        # Log for monitoring
        self.metrics_log.append({
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "result": result,
        })

        return result

    def batch_evaluate(self, evaluations: List[Dict]) -> List[Dict]:
        """Evaluate multiple samples and return aggregated metrics."""
        results = []
        for eval_item in evaluations:
            result = self.evaluate(
                prompt=eval_item["prompt"],
                response=eval_item["response"],
                reference=eval_item.get("reference"),
            )
            results.append(result)

        scores = [r["final_score"] for r in results]
        return {
            "individual_results": results,
            "aggregate": {
                "mean_score": round(sum(scores) / len(scores), 2),
                "min_score": min(scores),
                "max_score": max(scores),
                "std_dev": round(float(np.std(scores)), 3),
                "sample_count": len(scores),
            },
        }

    def get_bias_report(self) -> Dict:
        """Generate a bias monitoring report from logged evaluations."""
        if not self.metrics_log:
            return {"status": "no_data"}

        variances = [m["result"]["score_variance"] for m in self.metrics_log]
        penalties = [m["result"]["length_penalty"] for m in self.metrics_log]

        return {
            "total_evaluations": len(self.metrics_log),
            "avg_score_variance": round(float(np.mean(variances)), 4),
            "high_variance_rate": round(
                float(np.mean([v > 1.0 for v in variances])), 3
            ),
            "avg_length_penalty": round(float(np.mean(penalties)), 4),
            "calibration_status": (
                "calibrated" if self.calibration_data else "uncalibrated"
            ),
        }
```

### 14.8.7 Bias Mitigation Summary

The following production checklist summarizes the key mitigation strategies and their impact on evaluation reliability:

| Strategy | Bias Addressed | Cost Impact | Reliability Gain | Implementation Effort |
|----------|---------------|-------------|------------------|----------------------|
| Order averaging | Position bias | 2x evaluation calls | +15-20% accuracy | Low |
| Length normalization | Length bias | Negligible | +5-10% accuracy | Low |
| Cross-model judging | Self-preference | Moderate (multiple models) | +10-15% accuracy | Medium |
| Judge rotation pool | All biases | Moderate | +20-25% accuracy | Medium |
| Calibration framework | All biases | Quarterly human effort | Baseline measurement | Medium |
| Continuous monitoring | Drift | Ongoing compute | Catch degradation early | Low |

---

## 14.9 Key Takeaways

1. **Evaluation is not optional—build evaluation pipelines before deploying to production.** A golden dataset of 100-500 representative questions with expected answers is the foundation. Run it on every deployment. Alert if any metric drops below baseline. The investment in evaluation infrastructure pays for itself the first time it catches a quality regression.

2. **RAGAS is the standard for RAG evaluation—faithfulness, relevancy, precision, recall.** Faithfulness measures context grounding. Relevancy measures query alignment. Precision measures retrieval accuracy. Recall measures retrieval completeness. These four metrics cover the most common RAG failure modes.

3. **LLM-as-judge is cost-effective but imperfect—combine with human evaluation for critical applications.** LLM judges have biases (position, verbosity, self-preference) that require mitigation. Use different models as judges, average multiple evaluations, and periodically calibrate against human ratings.

4. **Regression testing prevents quality degradation—run golden dataset tests on every deployment.** Maintain a baseline of quality scores. Compare current scores against baseline on every change. Block deployments if any metric drops more than 5% below baseline. This catches subtle regressions that unit tests miss.

5. **Measure retrieval quality separately from generation quality.** A perfect generator cannot compensate for irrelevant retrieved documents. A perfect retriever cannot compensate for a poor generator. Measure both independently to identify which component needs improvement.

6. **Prompt testing is evaluation too—compare prompt variations systematically.** Use the same golden dataset to evaluate different prompt templates. Measure relevance, faithfulness, and accuracy for each. The best prompt is the one that produces the highest quality for your specific use case, not the one that looks most impressive in demos.

7. **Evaluation datasets must be representative and maintained.** Add new test cases for every production failure. Remove outdated cases. Balance across categories and difficulty levels. A stale evaluation dataset gives false confidence.

8. **Automated evaluation catches quantitative issues; human evaluation catches qualitative issues.** Use automated metrics for continuous monitoring. Use human evaluation for nuanced assessment—tone, creativity, appropriateness, and edge cases that automated metrics miss.

9. **Evaluate at multiple levels: component, pipeline, and end-to-end.** Unit test individual components. Integration test the pipeline. End-to-end test with real queries. Each level catches different classes of failure.

10. **Make evaluation results visible and actionable.** Dashboards showing quality trends over time. Alerts when metrics degrade. Diagnostic information for investigating failures. Evaluation that is not visible does not prevent regressions.

---

## 14.10 Further Reading

- **RAGAS Documentation** (docs.ragas.io) — Official documentation for RAG evaluation metrics, dataset preparation, and integration with CI/CD pipelines.

- **DeepEval Documentation** (docs.confident-ai.com) — Comprehensive guide to LLM evaluation metrics, test case creation, and pytest integration.

- **LangSmith Documentation** (docs.smith.langchain.com) — Official guide for tracing, evaluation, and monitoring of LangChain applications.

- **"Evaluating LLM Systems" by Hamel Husain (2024)** — Practical guide to building evaluation systems for production LLM applications.

- **"Building Effective RAG Systems" by Anthropic (2024)** — Covers evaluation patterns for RAG systems including faithfulness and relevance measurement.

- **"Human Evaluation of Language Models" by Liang et al. (2023)** — Research on human evaluation methodologies for LLMs, including inter-annotator agreement and evaluation frameworks.

- **"A Survey on Evaluation of Large Language Models" by Chang et al. (2023)** — Comprehensive survey covering automated metrics, human evaluation, and benchmark datasets.

- **"TruLens Documentation"** (truLens.org) — Evaluation and tracking for LLM experiments with feedback functions and dashboards.

- **"Site Reliability Engineering" by Google** — Chapters on SLIs, SLOs, and error budgets provide the framework for setting evaluation targets.

- **"Designing Data-Intensive Applications" by Martin Kleppmann** — Chapters on testing and monitoring provide patterns applicable to evaluation pipeline design.

- **HuggingFace Evaluate Library** (huggingface.co/docs/evaluate) — Library for computing evaluation metrics with support for custom metrics and distributed evaluation.
