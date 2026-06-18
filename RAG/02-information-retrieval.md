# Chapter 2: Information Retrieval Fundamentals

> "RAG is not a new invention. It is information retrieval — a 60-year-old discipline — applied to a new generation of language models."

---

**Last verified: June 2026.**

---

## Introduction

RAG is built on information retrieval. Every time you embed a query, search a vector store, and return ranked results, you are performing information retrieval. Understanding IR fundamentals — TF-IDF, BM25, inverted indexes, relevance scoring, evaluation — is essential for building production RAG systems. Most RAG implementations fail not because of the LLM, but because of poor retrieval.

The field of information retrieval predates modern AI by decades. The core problems — matching queries to documents, ranking results by relevance, evaluating retrieval quality — were studied extensively before anyone imagined using neural networks for search. The mathematical foundations of TF-IDF date to the 1970s. BM25 was published in 1994. Inverted indexes have been the backbone of search engines since the 1990s.

This matters because the classical IR techniques that underpin modern search are not obsolete — they are essential. Dense (semantic) retrieval using embedding models captures meaning, but it struggles with exact term matching. Sparse (keyword) retrieval using BM25 captures exact terms, but it struggles with meaning. The production standard is hybrid search: combining both approaches. To build effective hybrid search, you must understand both the classical and modern techniques.

This chapter covers the IR fundamentals that every RAG architect needs. We will dissect the BM25 scoring function, explain inverted indexes at the data structure level, compare dense and sparse retrieval approaches, and establish the evaluation metrics that measure retrieval quality. The chapter is technical and mathematical — these are the foundations that everything else in this book builds upon.

The central thesis of this chapter is that **hybrid search is not a nice-to-have — it is the production standard**. Dense and sparse retrieval have complementary strengths and weaknesses. Understanding these strengths and weaknesses, and knowing how to combine them, is the single most important skill for a retrieval engineer.

---

## 2.1 Classical Information Retrieval

### 2.1.1 The Retrieval Problem

Information retrieval is the problem of finding documents in a collection that are relevant to a user's query. This sounds simple, but it is deceptively complex. Consider a query like "contract termination for breach." A relevant document might use different terminology: "agreement cancellation due to non-performance," "early termination rights upon material breach," or "remedies following default." The retrieval system must recognize that these different formulations describe the same concept.

The two fundamental approaches to this problem are:

1. **Sparse retrieval (lexical matching)**: Match based on shared terms between query and document. Fast, exact, but misses semantic similarity.
2. **Dense retrieval (semantic matching)**: Match based on meaning, using learned embeddings. Captures semantic similarity but may miss exact terms.

Neither approach alone is sufficient. The production standard is to combine both.

### 2.1.2 Term Frequency-Inverse Document Frequency (TF-IDF)

TF-IDF was the standard weighting scheme for information retrieval for decades and still underpins modern search. The core insight is simple but powerful: a term is important to a document if it appears frequently in that document but rarely across the collection.

**Term Frequency (TF)** measures how often a term appears in a document:

```
TF(t, d) = count(t in d) / total_terms(d)
```

**Inverse Document Frequency (IDF)** measures how rare a term is across the collection:

```
IDF(t) = log(N / df(t))
```

Where N is the total number of documents and df(t) is the number of documents containing term t.

**TF-IDF score** combines these:

```
TF-IDF(t, d) = TF(t, d) * IDF(t)
```

Consider a collection of 10,000 legal contracts. The term "the" appears in almost every document — IDF is near zero, so it contributes almost nothing to the score. The term "arbitration" appears in only 200 documents — IDF is high, so it contributes significantly. A document that frequently uses "arbitration" scores high on that term, while a document that rarely uses it scores low.

```python
import math
from collections import Counter

def compute_tf(document: list[str]) -> dict[str, float]:
    """Compute term frequency for a document."""
    term_counts = Counter(document)
    total = len(document)
    return {term: count / total for term, count in term_counts.items()}

def compute_idf(documents: list[list[str]]) -> dict[str, float]:
    """Compute inverse document frequency across a corpus."""
    n_docs = len(documents)
    doc_freq = Counter()
    for doc in documents:
        unique_terms = set(doc)
        for term in unique_terms:
            doc_freq[term] += 1
    return {
        term: math.log(n_docs / freq)
        for term, freq in doc_freq.items()
    }

def compute_tfidf(
    document: list[str],
    idf: dict[str, float]
) -> dict[str, float]:
    """Compute TF-IDF scores for a document."""
    tf = compute_tf(document)
    return {term: tf_val * idf.get(term, 0) for term, tf_val in tf.items()}
```

### 2.1.3 BM25: The Industry Standard

BM25 (Best Match 25) is the industry standard for sparse retrieval. Published by Robertson et al. in 1994, it improves on TF-IDF with two critical innovations:

**Logarithmic term saturation**: In TF-IDF, term frequency is linear — a term appearing 10 times scores 10x more than a term appearing once. In practice, the difference between 1 and 5 occurrences is much more important than the difference between 95 and 100. BM25 applies logarithmic saturation:

```
tf_sat = (tf * (k1 + 1)) / (tf + k1)
```

The parameter k1 controls saturation. Typical values are 1.2-2.0. Higher k1 means more weight given to frequent terms.

**Document length normalization**: Longer documents naturally contain more terms, giving them an unfair advantage in TF-IDF. BM25 normalizes by document length:

```
norm = (1 - b + b * (dl / avgdl))
```

The parameter b controls normalization strength. b=0 means no normalization. b=1 means full normalization. Typical values are 0.75.

The full BM25 scoring function:

```
BM25(q, d) = IDF(q) * SUM(tf_sat(t, d) / norm(d))
```

```python
def bm25_score(
    query_terms: list[str],
    document_terms: list[str],
    idf: dict[str, float],
    k1: float = 1.5,
    b: float = 0.75,
    avg_doc_length: float = 100.0
) -> float:
    """Compute BM25 score for a query against a document."""
    doc_length = len(document_terms)
    term_freq = Counter(document_terms)
    score = 0.0

    for term in query_terms:
        if term not in idf:
            continue
        tf = term_freq.get(term, 0)
        tf_sat = (tf * (k1 + 1)) / (tf + k1 * (
            1 - b + b * doc_length / avg_doc_length
        ))
        score += idf[term] * tf_sat

    return score
```

The key insight for RAG architects: BM25 excels at exact term matching. A search for "Q4 2024 revenue" will find documents containing that exact phrase. Dense (semantic) search might miss it if the embedding does not capture the precise terminology. This is why hybrid search — combining BM25 with dense retrieval — consistently outperforms either approach.

### 2.1.4 BM25 Parameter Tuning

BM25 has two parameters that affect retrieval quality:

| Parameter | Range | Effect | Typical Value |
|-----------|-------|--------|---------------|
| k1 | 0.2-2.0 | Controls term frequency saturation. Higher = more weight on frequent terms. | 1.2-1.5 |
| b | 0.0-1.0 | Controls document length normalization. Higher = stronger normalization. | 0.75 |

**When to increase k1**: Documents with many term repetitions benefit from higher k1 (technical documents, code).

**When to decrease k1**: Collections with short, uniform documents benefit from lower k1 (tweets, chat messages).

**When to increase b**: Collections with highly variable document lengths benefit from higher b (mixed collections of short and long documents).

**When to decrease b**: Collections with uniform document lengths benefit from lower b (standardized forms).

Tuning BM25 parameters on your specific dataset and queries consistently improves retrieval quality by 5-15% over default settings.

---

## 2.2 Inverted Indexes

### 2.2.1 The Data Structure

An inverted index is the data structure that makes fast search possible. Without it, every query would require scanning every document — O(n) per query. With an inverted index, every query is O(1) per term lookup.

The inverted index maps each term to the list of documents containing it (the "posting list"):

```
Term -> [(doc_id, term_frequency, positions), ...]

"arbitration" -> [(doc_001, 3, [12, 45, 89]), (doc_042, 1, [7]), ...]
"termination" -> [(doc_001, 2, [3, 67]), (doc_015, 4, [1, 23, 45, 78]), ...]
"breach"      -> [(doc_001, 1, [90]), (doc_042, 2, [34, 56]), ...]
```

Searching for "arbitration termination" is fast: look up "arbitration" (get 1,000 documents), look up "termination" (get 5,000 documents), intersect the lists (find documents containing both), and rank by score. The entire operation takes milliseconds, regardless of corpus size.

### 2.2.2 Index Construction

Building an inverted index is a one-time cost that pays off at every query:

```python
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class PostingEntry:
    doc_id: str
    term_frequency: int
    positions: list[int] = field(default_factory=list)

class InvertedIndex:
    def __init__(self):
        self.index: dict[str, list[PostingEntry]] = defaultdict(list)
        self.doc_lengths: dict[str, int] = {}
        self.total_docs: int = 0
        self.total_term_frequency: dict[str, int] = defaultdict(int)

    def add_document(self, doc_id: str, terms: list[str]):
        """Index a document."""
        self.total_docs += 1
        self.doc_lengths[doc_id] = len(terms)

        term_positions = defaultdict(list)
        for pos, term in enumerate(terms):
            term_positions[term].append(pos)

        for term, positions in term_positions.items():
            tf = len(positions)
            self.index[term].append(PostingEntry(
                doc_id=doc_id,
                term_frequency=tf,
                positions=positions
            ))
            self.total_term_frequency[term] += tf

    def search(self, query_terms: list[str]) -> dict[str, float]:
        """Search the index using BM25 scoring."""
        scores = defaultdict(float)
        avg_doc_length = (
            sum(self.doc_lengths.values()) / self.total_docs
            if self.total_docs > 0 else 0
        )

        for term in query_terms:
            if term not in self.index:
                continue
            idf = math.log(
                (self.total_docs + 1) /
                (len(self.index[term]) + 1)
            ) + 1
            for entry in self.index[term]:
                dl = self.doc_lengths[entry.doc_id]
                tf_sat = (entry.term_frequency * (1.5 + 1)) / (
                    entry.term_frequency + 1.5 * (
                        1 - 0.75 + 0.75 * dl / avg_doc_length
                    )
                )
                scores[entry.doc_id] += idf * tf_sat

        return dict(sorted(
            scores.items(), key=lambda x: x[1], reverse=True
        ))
```

### 2.2.3 Index Optimizations

Production inverted indexes use several optimizations:

| Optimization | What It Does | Trade-off |
|-------------|-------------|-----------|
| **Compression** | Reduces index size (PForDelta, VByte) | Slightly slower decompression |
| **Skip lists** | Enables fast intersection of posting lists | More memory |
| **Field-level indexing** | Indexes title, body, etc. separately | More complex queries |
| **Positional indexing** | Stores term positions for phrase queries | More memory |
| **Forward index** | Maps doc_id to terms for document updates | More memory, faster updates |

Elasticsearch and OpenSearch use these optimizations internally. Understanding them helps when debugging retrieval quality issues or optimizing for specific use cases.

---

## 2.3 Dense Retrieval

### 2.3.1 How Dense Retrieval Works

Dense retrieval uses embedding models to convert text into dense vectors (typically 768-1536 dimensions). Similarity between query and document is measured by cosine similarity or inner product of their vectors.

The embedding model is trained to map semantically similar text to nearby points in vector space. "Vehicle" and "car" have similar embeddings because they appear in similar contexts during training. This is something BM25 cannot do — BM25 treats "vehicle" and "car" as completely different terms.

```python
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class DenseRetriever:
    def __init__(self, embedding_model):
        self.model = embedding_model
        self.documents: dict[str, np.ndarray] = {}

    def index(self, doc_id: str, text: str):
        """Embed and store a document."""
        self.documents[doc_id] = self.model.embed(text)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Find most similar documents to query."""
        query_embedding = self.model.embed(query)
        scores = {
            doc_id: cosine_similarity(query_embedding, doc_emb)
            for doc_id, doc_emb in self.documents.items()
        }
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
```

### 2.3.2 Dense Retrieval Strengths and Weaknesses

| Strength | Example |
|----------|---------|
| Semantic matching | "vehicle" matches "car" |
| Synonym handling | "big" matches "large" |
| Conceptual understanding | "how to fix login problem" matches "authentication failure resolution" |

| Weakness | Example |
|----------|---------|
| Exact term matching | "Q4 2024 revenue" may not match "fourth quarter 2024 earnings" |
| Rare terms | Domain-specific jargon may not be well-represented in training data |
| Numerical precision | "123 Main Street" vs. "123 Main St" may not match |
| Out-of-vocabulary | New terms not in training data produce poor embeddings |

### 2.3.3 Embedding Model Selection

The embedding model is the foundation of dense retrieval quality. The following table compares common embedding models:

| Model | Dimensions | Max Tokens | MTEB Score | Cost per 1M Tokens | Best For |
|-------|-----------|------------|------------|--------------------|----------|
| OpenAI text-embedding-3-large | 3072 | 8191 | 64.6 | $0.13 | General purpose |
| OpenAI text-embedding-3-small | 1536 | 8191 | 62.3 | $0.02 | Cost-sensitive |
| Cohere Embed v3 | 1024 | 512 | 64.5 | $0.10 | Multi-lingual, enterprise |
| BGE-large-en-v1.5 | 1024 | 512 | 64.2 | Self-hosted | Open source |
| E5-large-v2 | 1024 | 512 | 62.8 | Self-hosted | Open source |
| GTE-large | 1024 | 8192 | 63.1 | Self-hosted | Long documents |

The choice depends on your constraints:
- **Cost-sensitive**: OpenAI text-embedding-3-small or self-hosted models
- **Enterprise**: Cohere Embed v3 (strong on legal/medical text)
- **Long documents**: GTE-large (8192 token context)
- **Open source**: BGE-large or E5-large

---

## 2.4 Hybrid Search

### 2.4.1 Why Hybrid Search Wins

The fundamental insight behind hybrid search is that dense and sparse retrieval have complementary strengths:

| Query Type | Dense Retrieval | Sparse Retrieval | Hybrid |
|-----------|----------------|-----------------|--------|
| Semantic: "how to fix login problem" | Strong | Weak (no exact match) | Strong |
| Exact: "Q4 2024 revenue" | Weak (may miss exact phrase) | Strong | Strong |
| Technical: "Section 7.2(b)" | Weak (numbers in embedding) | Strong (exact match) | Strong |
| Conceptual: "vendor risk management" | Strong | Moderate | Strong |

Hybrid search combines both approaches, typically using Reciprocal Rank Fusion (RRF) to merge ranked lists.

### 2.4.2 Reciprocal Rank Fusion

RRF is the most common technique for combining ranked lists. It assigns scores based on rank position:

```
RRF_score(d) = SUM(1 / (k + rank_i(d)))
```

Where k is a constant (typically 60) that controls how much weight is given to lower-ranked results.

```python
def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60
) -> list[tuple[str, float]]:
    """Merge multiple ranked lists using RRF."""
    scores: dict[str, float] = defaultdict(float)
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list):
            scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

RRF is robust because it does not require score normalization. Dense retrieval scores (cosine similarity) and sparse retrieval scores (BM25) are on different scales. RRF uses only rank positions, making it agnostic to score scales.

### 2.4.3 Weighted Hybrid Search

A more sophisticated approach assigns different weights to dense and sparse results:

```python
def weighted_hybrid_search(
    dense_results: list[tuple[str, float]],
    sparse_results: list[tuple[str, float]],
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
    score_normalization: str = "minmax"
) -> list[tuple[str, float]]:
    """Combine dense and sparse results with weights."""
    scores: dict[str, float] = defaultdict(float)

    # Normalize scores to [0, 1]
    dense_scores = normalize_scores(
        [s for _, s in dense_results], score_normalization
    )
    sparse_scores = normalize_scores(
        [s for _, s in sparse_results], score_normalization
    )

    for (doc_id, _), norm_score in zip(dense_results, dense_scores):
        scores[doc_id] += dense_weight * norm_score
    for (doc_id, _), norm_score in zip(sparse_results, sparse_scores):
        scores[doc_id] += sparse_weight * norm_score

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def normalize_scores(
    scores: list[float], method: str = "minmax"
) -> list[float]:
    """Normalize scores to [0, 1]."""
    if method == "minmax":
        min_s, max_s = min(scores), max(scores)
        range_s = max_s - min_s if max_s != min_s else 1.0
        return [(s - min_s) / range_s for s in scores]
    elif method == "zscore":
        mean_s = sum(scores) / len(scores)
        std_s = (sum((s - mean_s) ** 2 for s in scores) / len(scores)) ** 0.5
        return [(s - mean_s) / (std_s or 1.0) for s in scores]
    return scores
```

### 2.4.4 Adaptive Hybrid Search

The optimal weighting between dense and sparse retrieval varies by query. Short, specific queries benefit from sparse retrieval. Long, conceptual queries benefit from dense retrieval. Adaptive weighting adjusts the balance based on query characteristics.

| Query Characteristic | Dense Weight | Sparse Weight | Rationale |
|---------------------|-------------|--------------|-----------|
| Short query (<5 words) | 0.3 | 0.7 | Specific terms dominate |
| Long query (>10 words) | 0.7 | 0.3 | Semantic meaning dominates |
| Contains exact terms (codes, IDs) | 0.2 | 0.8 | Exact matching critical |
| Conceptual query | 0.8 | 0.2 | Semantic matching critical |
| Mixed query | 0.5 | 0.5 | Balanced approach |

---

## 2.5 Query Expansion and Rewriting

### 2.5.1 The Query-Document Vocabulary Mismatch

Users do not always search with the terms that appear in documents. A user might search "how to fix login problem" when the document says "authentication failure resolution." Query expansion adds related terms to bridge this gap. Query rewriting uses an LLM to reformulate the query for better retrieval.

Both techniques improve recall — they help find relevant documents that exact term matching would miss. The cost is additional latency (one LLM call for rewriting) and potential noise (expansion might add irrelevant terms).

### 2.5.2 Query Rewriting with LLMs

```python
REWRITE_PROMPT = """You are a search query reformulator. Rewrite the following
user query to improve document retrieval. The rewritten query should:
1. Use formal/technical language that appears in legal documents
2. Include synonyms and related terms
3. Preserve the user's intent

User query: {query}

Rewritten query:"""

async def rewrite_query(query: str, llm) -> str:
    """Rewrite query for better retrieval."""
    response = await llm.generate(
        REWRITE_PROMPT.format(query=query)
    )
    return response.strip()
```

### 2.5.3 Query Expansion Techniques

| Technique | How It Works | Quality Impact | Cost Impact |
|-----------|-------------|----------------|-------------|
| **Synonym expansion** | Add synonyms from thesaurus or word embeddings | +5-10% recall | Negligible |
| **LLM expansion** | Use LLM to generate related terms | +10-20% recall | +1 LLM call |
| **Pseudo-relevance feedback** | Use top-retrieved documents to expand query | +10-15% recall | +1 retrieval pass |
| **Hyde** | Generate a hypothetical answer, use it as query | +15-25% recall | +1 LLM call |

### 2.5.4 Query Decomposition

Complex queries can be decomposed into sub-queries, each targeting a specific aspect of the information need:

```python
DECOMPOSE_PROMPT = """Break down the following complex query into 2-4 simpler
sub-queries that together address the full information need.

Query: {query}

Sub-queries (one per line):"""

async def decompose_query(query: str, llm) -> list[str]:
    """Decompose a complex query into sub-queries."""
    response = await llm.generate(DECOMPOSE_PROMPT.format(query=query))
    return [q.strip() for q in response.strip().split("\n") if q.strip()]
```

| Query Type | Decomposition Strategy | Example |
|-----------|----------------------|---------|
| Multi-entity | Split by entity | "Contracts with Acme and Beta" becomes two queries |
| Multi-aspect | Split by aspect | "Pricing and termination terms" becomes two queries |
| Comparative | Split by side | "Compare US and EU contracts" becomes two queries |
| Temporal | Split by time | "Changes since 2023" becomes two queries |

---

## 2.6 Retrieval Metrics

### 2.6.1 Precision and Recall

Precision measures how many retrieved documents are relevant. Recall measures how many relevant documents are retrieved. These are the fundamental metrics of retrieval quality.

| Metric | Formula | Target | What It Tells You |
|--------|---------|--------|-------------------|
| Precision@K | Relevant in top-K / K | >80% | Model sees relevant information |
| Recall@K | Relevant in top-K / Total relevant | >90% | Few relevant docs missed |
| F1@K | Harmonic mean of P@K and R@K | >85% | Balanced quality |

### 2.6.2 Ranking Quality Metrics

**MRR (Mean Reciprocal Rank)** measures where the first relevant document appears:

```
MRR = (1/Q) * SUM(1/rank_i)
```

Where Q is the number of queries and rank_i is the rank of the first relevant document for query i. Target: >0.8.

**NDCG (Normalized Discounted Cumulative Gain)** considers the position of relevant results. Higher-ranked relevant results contribute more:

```
DCG@K = SUM((2^relevance_i - 1) / log2(i + 1))
NDCG@K = DCG@K / IDCG@K
```

Where IDCG is the ideal DCG (perfect ranking). Target: >0.85.

### 2.6.3 Evaluation Framework

```python
from dataclasses import dataclass

@dataclass
class RetrievalMetrics:
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg: float
    hit_rate: float

def evaluate_retrieval(
    queries: list[str],
    retrieved_docs: list[list[str]],
    relevant_docs: list[set[str]],
    k: int = 5
) -> RetrievalMetrics:
    """Evaluate retrieval quality across a set of queries."""
    precisions = []
    recalls = []
    reciprocal_ranks = []
    dcgs = []
    idcgs = []
    hits = 0

    for query, retrieved, relevant in zip(
        queries, retrieved_docs, relevant_docs
    ):
        retrieved_at_k = retrieved[:k]
        relevant_retrieved = len(set(retrieved_at_k) & relevant)
        precisions.append(relevant_retrieved / k)
        recalls.append(
            relevant_retrieved / len(relevant) if relevant else 0
        )
        for rank, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
        dcg = sum(
            (1.0 if retrieved_at_k[i] in relevant else 0.0) /
            math.log2(i + 2)
            for i in range(len(retrieved_at_k))
        )
        ideal_dcg = sum(
            1.0 / math.log2(i + 2)
            for i in range(min(len(relevant), k))
        )
        dcgs.append(dcg)
        idcgs.append(ideal_dcg if ideal_dcg > 0 else 1.0)
        if any(doc_id in relevant for doc_id in retrieved_at_k):
            hits += 1

    n = len(queries)
    return RetrievalMetrics(
        precision_at_k=sum(precisions) / n,
        recall_at_k=sum(recalls) / n,
        mrr=sum(reciprocal_ranks) / n,
        ndcg=sum(d / i for d, i in zip(dcgs, idcgs)) / n,
        hit_rate=hits / n
    )
```

### 2.6.4 Metrics Selection Guide

| Metric | When to Use | What It Tells You | Limitation |
|--------|------------|-------------------|------------|
| **Precision@K** | When irrelevant results are costly | How clean the results are | Does not capture ranking |
| **Recall@K** | When missing relevant docs is costly | How complete the results are | Requires knowing all relevant docs |
| **MRR** | When the first result matters most | How quickly users find answers | Ignores results after first relevant |
| **NDCG** | When ranking quality matters | Overall ranking quality | Complex to compute and interpret |
| **Hit Rate** | When any relevant result is sufficient | Basic quality bar | Does not measure ranking quality |

---

## 2.7 Search System Architecture

### 2.7.1 Elasticsearch and OpenSearch

The production standard for hybrid search. Both support BM25 (via inverted indexes) and dense retrieval (via dense_vector fields) with Reciprocal Rank Fusion to combine results.

The architectural advantage of Elasticsearch is that it handles both sparse and dense search in a single system, eliminating the need for a separate vector database in many cases.

```json
{
  "query": {
    "bool": {
      "should": [
        {
          "match": {
            "content": {
              "query": "contract termination breach",
              "boost": 0.5
            }
          }
        },
        {
          "knn": {
            "field": "content_embedding",
            "query_vector": [0.1, 0.2],
            "k": 10,
            "boost": 0.5
          }
        }
      ]
    }
  }
}
```

### 2.7.2 Comparison: Search Systems

| System | BM25 | Dense | Hybrid | Cost | Best For |
|--------|------|-------|--------|------|----------|
| **Elasticsearch** | Native | dense_vector | RRF built-in | Self-hosted or cloud | Production hybrid search |
| **OpenSearch** | Native | k-NN plugin | Manual fusion | Self-hosted | Open-source alternative |
| **Weaviate** | Via BM25 module | Native | Built-in | Cloud or self-hosted | Vector-first with BM25 |
| **Pinecone** | No | Native | No | Cloud | Pure vector search |
| **Chroma** | No | Native | No | Self-hosted | Prototyping |
| **Qdrant** | No | Native | Manual | Self-hosted | High-performance vector |

For production hybrid search, Elasticsearch or OpenSearch is the standard. For vector-first applications with hybrid needs, Weaviate is a strong choice. For pure vector search, Pinecone or Qdrant are performant options.

---

## 2.8 Case Study: Hybrid Search Optimization

### 2.8.1 Problem Statement

A legal research firm's RAG system uses only dense retrieval (Cohere Embed v3) for searching 100,000 contracts. The system achieves 62% precision@5 — below the 90% target. The primary failure mode: queries with exact terms (section numbers, dates, specific clause references) do not find the right documents.

### 2.8.2 Diagnosis

Analyzing failed queries reveals the pattern:

| Query Type | Dense Precision@5 | Root Cause |
|-----------|------------------|------------|
| Semantic queries | 78% | Acceptable |
| Exact term queries | 31% | Dense retrieval misses exact matches |
| Mixed queries | 55% | Partial match |
| **Overall** | **62%** | |

The fix: add BM25 search and combine with RRF.

### 2.8.3 Implementation

```python
from elasticsearch import Elasticsearch

class HybridLegalSearcher:
    def __init__(self):
        self.es = Elasticsearch("localhost:9200")
        self.dense_client = initialize_weaviate()
        self.sparse_index = "contracts_bm25"

    def search(
        self,
        query: str,
        top_k: int = 20
    ) -> list[dict]:
        """Hybrid search combining dense and sparse retrieval."""
        # Dense search via Weaviate
        dense_response = self.dense_client.query.near_text(
            query,
            limit=top_k
        )
        dense_results = [
            (obj.uuid, obj.metadata.distance)
            for obj in dense_response.objects
        ]

        # Sparse search via Elasticsearch
        sparse_response = self.es.search(
            index=self.sparse_index,
            query={"match": {"content": {"query": query}}},
            size=top_k
        )
        sparse_results = [
            (hit["_id"], hit["_score"])
            for hit in sparse_response["hits"]["hits"]
        ]

        # Combine with RRF
        combined = self._rrf_fusion(dense_results, sparse_results)
        return combined[:top_k]

    def _rrf_fusion(
        self,
        dense: list[tuple[str, float]],
        sparse: list[tuple[str, float]],
        k: int = 60
    ) -> list[tuple[str, float]]:
        scores = {}
        for rank, (doc_id, _) in enumerate(dense):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        for rank, (doc_id, _) in enumerate(sparse):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### 2.8.4 Cost Analysis

| Component | Before (Dense Only) | After (Hybrid) | Change |
|-----------|---------------------|----------------|--------|
| Embedding (query) | $0.0001 | $0.0001 | Same |
| Vector search | $0.00005 | $0.00005 | Same |
| BM25 search | N/A | $0.00002 | +$0.00002 |
| Reranking | $0.001 | $0.001 | Same |
| LLM generation | $0.025 | $0.025 | Same |
| **Total** | **$0.026** | **$0.026** | +$0.00002 |

The cost increase is negligible ($0.02/month at 10K queries/day). The precision improvement is 28 percentage points (62% to 90%). This is the highest-ROI change in the system.

### 2.8.5 Results

| Metric | Dense Only | Hybrid | Improvement |
|--------|-----------|--------|-------------|
| Precision@5 | 62% | 90% | +28% |
| Recall@5 | 71% | 93% | +22% |
| MRR | 0.68 | 0.89 | +0.21 |
| Hit Rate | 82% | 97% | +15% |
| Latency (p50) | 1.2s | 1.5s | +0.3s |
| Cost per query | $0.026 | $0.026 | +$0.00002 |

The latency increase of 0.3s is within the 2-second SLA. The precision improvement justifies the additional complexity.

---

## 2.9 Testing Information Retrieval

### 2.9.1 Unit Testing BM25

```python
import pytest

@pytest.fixture
def bm25_index():
    index = InvertedIndex()
    index.add_document("doc1", ["contract", "termination", "breach"])
    index.add_document("doc2", ["agreement", "termination", "notice"])
    index.add_document("doc3", ["contract", "payment", "terms"])
    return index

def test_bm25_finds_exact_match(bm25_index):
    results = bm25_index.search(["termination"])
    assert "doc1" in results
    assert "doc2" in results
    assert "doc3" not in results

def test_bm25_ranks_exact_match_higher(bm25_index):
    results = bm25_index.search(["contract", "termination"])
    assert results.index("doc1") < results.index("doc2")

def test_bm25_handles_empty_query(bm25_index):
    results = bm25_index.search([])
    assert results == {}
```

### 2.9.2 Unit Testing Hybrid Search

```python
def test_rrf_fusion_combines_results():
    dense_results = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
    sparse_results = [("doc2", 5.0), ("doc4", 4.0), ("doc1", 3.0)]

    merged = reciprocal_rank_fusion([dense_results, sparse_results])
    doc_ids = [doc_id for doc_id, _ in merged]

    assert doc_ids[0] in ("doc1", "doc2")
    assert doc_ids[1] in ("doc1", "doc2")

def test_rrf_handles_single_list():
    results = [("doc1", 0.9), ("doc2", 0.8)]
    merged = reciprocal_rank_fusion([results])
    assert len(merged) == 2
    assert merged[0][0] == "doc1"
```

### 2.9.3 Integration Testing Retrieval Quality

```python
def test_retrieval_quality_on_golden_set():
    evaluator = RetrievalEvaluator("legal_contracts_eval.jsonl")
    searcher = HybridLegalSearcher()
    metrics = evaluator.evaluate(searcher, k=5)

    assert metrics.precision_at_k > 0.80
    assert metrics.recall_at_k > 0.90
    assert metrics.mrr > 0.80
    assert metrics.hit_rate > 0.95

def test_retrieval_consistency():
    searcher = HybridLegalSearcher()
    query = "contract termination for breach"
    results_1 = searcher.search(query, top_k=10)
    results_2 = searcher.search(query, top_k=10)
    assert [r["doc_id"] for r in results_1] == [
        r["doc_id"] for r in results_2
    ]
```

---

## 2.10 Key Takeaways

1. **BM25 is the industry standard for sparse retrieval.** Understand its scoring function (term frequency saturation + document length normalization) and tune its parameters (k1, b) on your specific data.

2. **Inverted indexes enable fast search.** They are the data structure behind every production search system. Understanding them helps when debugging retrieval quality issues.

3. **Dense retrieval captures meaning but misses exact terms.** Embedding models map semantic similarity, but they struggle with exact term matching, numerical precision, and rare terminology.

4. **Hybrid search (dense + sparse) is the production standard.** It consistently outperforms either approach alone. Use RRF for robust fusion or weighted fusion for adaptive control.

5. **Query rewriting and expansion improve recall 10-20%.** LLM-based query rewriting bridges the vocabulary gap between how users search and how documents are written.

6. **Query decomposition handles complex queries.** Break multi-part queries into sub-queries, retrieve for each, and synthesize the results.

7. **Retrieval metrics measure quality independently from generation.** Use Precision@K, Recall@K, MRR, and NDCG to diagnose retrieval problems before blaming the LLM.

8. **Elasticsearch/OpenSearch are production standards for hybrid search.** They handle BM25 and dense retrieval in a single system. Weaviate is a strong alternative for vector-first architectures.

9. **Embedding model selection is a critical architectural decision.** Match the model to your domain (legal, medical, financial), document length, and cost constraints.

10. **The highest-ROI improvement is usually adding hybrid search.** Dense-only retrieval leaves precision on the table. Adding BM25 with RRF fusion typically improves precision by 15-30% for negligible cost.

---

## 2.11 Further Reading

- **"Introduction to Information Retrieval" by Manning, Raghavan, and Schutze** — The definitive textbook on information retrieval. Covers TF-IDF, BM25, inverted indexes, and evaluation in depth.

- **"The Probabilistic Relevance Framework: BM25 and Beyond" by Robertson and Zaragoza (2009)** — The authoritative paper on BM25 theory and variants.

- **"Okapi at TREC-3" by Robertson et al. (1994)** — The original BM25 paper. Essential reading for understanding the theoretical foundation.

- **"Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods" by Cormack et al. (2009)** — The RRF paper. Explains why RRF is robust across different retrieval systems.

- **"ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT" by Khattab and Zaharia (2020)** — Late interaction retrieval that combines the benefits of dense and sparse approaches.

- **Elasticsearch Documentation** (elastic.co/guide/en/elasticsearch/reference/current) — Complete reference for BM25 tuning, hybrid search implementation, and index optimization.

- **OpenSearch Documentation** (opensearch.org/docs/latest) — Open-source alternative to Elasticsearch with k-NN plugin for dense retrieval.

- **Weaviate Documentation** (weaviate.io/developers/weaviate) — Vector database with native hybrid search support.

- **"Pretrained Transformers for Text Ranking: BERT and Beyond" by Nogueira and Cho (2020)** — Survey of neural ranking models, relevant to understanding dense retrieval and reranking.

- **BEIR Benchmark** (github.com/beir-cellar/beir) — Heterogeneous benchmark for evaluating retrieval systems. Essential for comparing embedding models and retrieval approaches.
