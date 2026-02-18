# 📦 RAG Chunking Methods — Multi-Document Comparison

A research notebook for testing and comparing **4 chunking strategies** on a multi-document corpus, designed as the foundational layer for a full Hybrid RAG pipeline (BM25 + Semantic + LLaMA).

---

## 🧠 What is RAG Chunking and Why Does It Matter?

Retrieval-Augmented Generation (RAG) works in two stages: first **retrieve** relevant passages from a corpus, then **generate** an answer using an LLM. The chunking step happens before any of that — it decides how to split your raw documents into storable, searchable units.

Chunking quality directly determines retrieval quality. A bad chunking strategy can cut sentences mid-thought, mix unrelated topics into one chunk, or produce chunks so small they lack context for the LLM. The goal of this notebook is to empirically compare different approaches so you can make an informed choice for your own RAG system.

---

## 🗂️ Project Structure

```
rag_chunking_multidoc.ipynb   ← Main notebook
README.md                     ← This file
chunk_distributions.png       ← Auto-generated: chunk length histograms
chunks_per_doc.png            ← Auto-generated: chunks per document bar chart
```

---

## 📚 Documents

The notebook fetches **8 long Wikipedia articles** by default, forming a multi-document AI/ML corpus:

| # | Article |
|---|---------|
| 0 | Artificial Intelligence |
| 1 | Machine Learning |
| 2 | Natural Language Processing |
| 3 | Computer Vision |
| 4 | Deep Learning |
| 5 | Reinforcement Learning |
| 6 | Transformer (deep learning architecture) |
| 7 | Large Language Model |

You can add or remove any Wikipedia article by editing the `WIKI_TITLES` list in **Cell 5**. The notebook will fetch, clean, and combine them into a unified corpus automatically.

---

## ✂️ Chunking Methods

### 1. Overlap Chunking
The simplest approach. Splits text into fixed-size character windows with a configurable overlap so context is not lost at boundaries.

```
[----chunk 1----]
           [----chunk 2----]
                      [----chunk 3----]
```

**Config:**
```python
CHUNK_SIZE = 600   # characters per chunk
OVERLAP    = 150   # shared characters between consecutive chunks
```

**Best for:** Uniform dense text (legal documents, technical papers), baseline comparisons, fast prototyping.

**Weakness:** Completely blind to semantics — boundaries can cut mid-sentence or mid-thought.

---

### 2. Semantic Chunking
Embeds every sentence and starts a new chunk wherever the **cosine similarity between consecutive sentence embeddings drops** below a threshold — indicating a topic shift.

```
[high sim][high sim][high sim] → SPLIT ← [high sim][high sim] → SPLIT
         Chunk 1                                   Chunk 2
```

**Config:**
```python
SIM_THRESHOLD       = 0.75   # lower = more splits
min_chunk_sentences = 3
max_chunk_sentences = 20
```

**Best for:** Documents that jump between topics (Wikipedia articles, reports, books). Produces the most topically coherent chunks.

**Weakness:** Slower than overlap (requires encoding all sentences). Sensitive to threshold tuning.

---

### 3. Hierarchical Chunking
Creates **two levels** of chunks:

- **Parent chunks** (~1000 chars): Large, used for generation context
- **Child chunks** (~200 chars): Small, sentence-based, used for precise retrieval

Each child carries a `parent_id` reference. At retrieval time, you search on children but feed the parent to the LLM — getting both high precision and rich context.

```
Parent 0 [large context]
  ├── Child 0 [precise retrieval unit]
  ├── Child 1 [precise retrieval unit]
  └── Child 2 [precise retrieval unit]

Parent 1 [large context]
  ├── Child 3
  └── Child 4
```

> ⚠️ **Important implementation note:** Children are built by accumulating **full sentences** up to `child_size`, not by character slicing. This prevents mid-word cuts like `"y of programs..."` that would appear with naive character splitting.

**Config:**
```python
PARENT_SIZE = 1000   # chars per parent
CHILD_SIZE  = 200    # target chars per child (sentence-aligned)
```

**Best for:** Production RAG systems. Separates retrieval precision from generation context. Works especially well with hybrid BM25 + semantic retrieval.

---

### 4. Context-Aware / LLM Chunking
Passes text segments to an LLM and asks it to identify **natural semantic split points** — section starts, topic transitions, argument boundaries.

**With Ollama (LLaMA 3) — recommended:**
```python
OLLAMA_MODEL = "llama3"   # or llama3.1, mistral, phi3
```

The LLM receives numbered sentences and returns a JSON array of split indices:
```
Input:  [0] Sentence one. [1] Sentence two. [2] New topic starts here...
Output: [0, 2, 7, 14]   ← chunk boundaries
```

**Automatic fallback (no Ollama needed):** If Ollama is not running, the notebook automatically switches to a heuristic splitter that uses paragraph boundaries + structural signals (heading-like sentences, question/exclamation endings).

**Best for:** High-value documents where quality matters more than speed. Produces the most semantically meaningful boundaries.

**Weakness:** Expensive — calls an LLM just for chunking, before any retrieval happens. Not ideal for large corpora.

---

## 📊 What Gets Measured

Every chunking method reports:

| Metric | Description |
|--------|-------------|
| `# Chunks` | Total chunks produced across all documents |
| `# Docs` | Number of source documents covered |
| `Avg Chars` | Average characters per chunk |
| `Min/Max Chars` | Shortest and longest chunk |

Plus two visualizations:
- **Chunk length distribution histograms** — shows how consistent/variable chunk sizes are per method
- **Chunks per document bar chart** — shows if any document is dominating the corpus

---

## 🔍 Chunk Metadata

Every chunk (except overlap, which uses plain strings internally) is stored as a Python dict with full provenance:

```python
{
    "text"      : "Machine learning is the study of...",
    "doc_id"    : 1,
    "doc_title" : "Machine learning",
    "chunk_index": 4       # position within the source document
}
```

This means query results always show you **which document the answer came from**, not just the chunk text.

---

## 🔎 Interactive Query Test

The final cell lets you search across all documents and compare retrieval quality:

```python
QUERY = "What are the main applications of machine learning?"
TOP_K = 3
```

Results show score, chunk ID, and source document for every method:

```
────────────────────────────────────────────────
  [Semantic]  Top-3 results
────────────────────────────────────────────────
  #1 score=0.590 | chunk_id=37 | doc='Machine learning'
  Classifiers and statistical learning methods...

  #2 score=0.589 | chunk_id=17 | doc='Artificial intelligence'
  There are several kinds of machine learning...
```

---

## 🚀 Getting Started

### Option 1 — Google Colab (recommended)

1. Open the notebook in Colab
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Run all cells — documents are fetched automatically

### Option 2 — Local

```bash
pip install sentence-transformers transformers nltk requests scikit-learn pandas matplotlib
```

```bash
jupyter notebook rag_chunking_multidoc.ipynb
```

### Option 3 — With LLaMA (for context-aware chunking)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start server and pull model
ollama serve &
ollama pull llama3
```

Then run the notebook — it auto-detects Ollama and switches to LLM chunking.

> **Note on TPU:** Ollama does not support Google TPU. If running on Colab TPU, the notebook automatically falls back to heuristic context-aware chunking. For the embedding model, use `TinyLlama/TinyLlama-1.1B-Chat-v1.0` or `google/gemma-2b-it` (requires HuggingFace token) as TPU-compatible alternatives.

---

## ⚙️ Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `WIKI_TITLES` | 8 articles | List of Wikipedia articles to fetch |
| `CHUNK_SIZE` | 600 | Overlap chunking: chars per chunk |
| `OVERLAP` | 150 | Overlap chunking: shared chars |
| `SIM_THRESHOLD` | 0.75 | Semantic chunking: cosine sim split threshold |
| `PARENT_SIZE` | 1000 | Hierarchical: chars per parent chunk |
| `CHILD_SIZE` | 200 | Hierarchical: target chars per child |
| `OLLAMA_MODEL` | `llama3` | LLM model for context-aware chunking |
| `QUERY` | (example) | Search query for the retrieval test |
| `TOP_K` | 3 | Number of results to return per method |

---

## 📈 Experimental Results (AI/ML Wikipedia Corpus)

Tested on the default 8-document corpus:

| Method | Chunks | Avg Chars | Content Quality | Retrieval Score |
|--------|--------|-----------|----------------|-----------------|
| Overlap | ~1800 | 600 | ⚠️ Mid-word cuts | 0.581 |
| Semantic | ~420 | variable | ✅ Coherent topics | 0.590 |
| Hierarchical (children) | ~3200 | 200 | ✅ Clean sentences | 0.607* |
| Context-Aware | ~380 | variable | ✅ Natural boundaries | 0.592 |

> *Hierarchical scores highest on cosine similarity due to higher term density in smaller chunks. However, **semantic chunking produces the cleanest readable content** — the score difference is partially an artifact of chunk size, not true semantic quality. For production use, hierarchical is recommended because it decouples retrieval precision (children) from generation context (parents).

---

## 🗺️ Roadmap — Next Notebooks

This notebook covers **Stage 1: Chunking**. The full pipeline will include:

- **Stage 2:** BM25 sparse retrieval + semantic dense retrieval
- **Stage 3:** Hybrid fusion via Reciprocal Rank Fusion (RRF)
- **Stage 4:** LLaMA generation on top of retrieved chunks
- **Stage 5:** Evaluation metrics (faithfulness, relevance, completeness)

---

## 🛠️ Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `sentence-transformers` | latest | Sentence embeddings (all-MiniLM-L6-v2) |
| `nltk` | latest | Sentence tokenization |
| `scikit-learn` | latest | Cosine similarity |
| `requests` | latest | Wikipedia API fetch |
| `pandas` | latest | Summary tables |
| `matplotlib` | latest | Visualizations |
| `ollama` | optional | LLaMA-based context-aware chunking |

---
