# ✂️ Chunking Methods Comparison — Multi-Document

A hands-on notebook for comparing **4 text chunking strategies** across multiple documents to find out which one produces the best chunks for downstream use. This is a **pure chunking comparison** — no retrieval pipeline, no LLM generation, just a focused study of how different splitting methods behave on real long-form text.

---

## 🧠 Why Compare Chunking Methods?

When you split a document, the quality of those splits affects everything that comes after — search precision, context completeness, and answer quality. But there is no universally best method. Each strategy makes different tradeoffs:

- Does it preserve complete sentences and thoughts?
- Does it respect topic boundaries?
- Is it fast enough for large corpora?
- Does it give you consistent or variable chunk sizes?

This notebook lets you run all 4 methods on the same corpus and see the differences side by side — chunk counts, size distributions, content quality, and how each method responds to the same search query.

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

| doc_id | Article |
|--------|---------|
| 0 | Artificial Intelligence |
| 1 | Machine Learning |
| 2 | Natural Language Processing |
| 3 | Computer Vision |
| 4 | Deep Learning |
| 5 | Reinforcement Learning |
| 6 | Transformer (deep learning architecture) |
| 7 | Large Language Model |

You can add or remove any Wikipedia article by editing `WIKI_TITLES` in Cell 5. The notebook fetches, cleans, and combines them into a unified corpus automatically.

---

## ✂️ Chunking Methods

### 1. Overlap Chunking

The simplest approach. Splits text into fixed-size character windows with a configurable overlap so context is not lost at boundaries.

```
[----chunk 1----]
           [----chunk 2----]
                      [----chunk 3----]
```
## 📝 License

MIT — free to use, modify, and build on.


**Config:**
```python
CHUNK_SIZE = 600   # characters per chunk
OVERLAP    = 150   # shared characters between consecutive chunks
```

**Strength:** Fast, simple, predictable chunk sizes.

**Weakness:** Completely blind to semantics — cuts mid-sentence or mid-word if the character limit lands there.

---

### 2. Semantic Chunking

Embeds every sentence and starts a new chunk wherever the **cosine similarity between consecutive sentence embeddings drops** below a threshold — indicating a topic shift.

```
[high sim][high sim][high sim] → SPLIT ← [high sim][high sim] → SPLIT
         Chunk 1                                   Chunk 2
```

**Config:**
```python
SIM_THRESHOLD       = 0.75   # lower = more splits, higher = fewer splits
min_chunk_sentences = 3
max_chunk_sentences = 20
```

**Strength:** Produces topically coherent chunks that respect natural topic boundaries. Best content quality.

**Weakness:** Slower than overlap — needs to encode every sentence before any splitting happens. Sensitive to threshold tuning.

---

### 3. Hierarchical Chunking

Creates **two levels** of chunks from the same text:

- **Parent chunks** (~1000 chars): Large, preserve full context
- **Child chunks** (~200 chars): Small, sentence-aligned, precise units

Each child carries a `parent_id` so you can always trace back to the full context it came from.

```
## 📝 License

MIT — free to use, modify, and build on.

Parent 0 [large context block]
  ├── Child 0 [sentence-aligned unit]
  ├── Child 1 [sentence-aligned unit]
  └── Child 2 [sentence-aligned unit]

Parent 1 [large context block]
  ├── Child 3
  └── Child 4
```

> ⚠️ **Implementation note:** Children are built by accumulating **full sentences** up to `child_size`, not raw character slicing. This prevents mid-word cuts like `"y of programs..."` that appear with naive character splitting.

**Config:**
```python
PARENT_SIZE = 1000   # chars per parent
CHILD_SIZE  = 200    # target chars per child (sentence-aligned)
```

**Strength:** Best balance between granularity and context. Children score highest in cosine similarity tests due to high term density per chunk.

**Weakness:** Produces significantly more chunks than other methods — higher memory usage at scale.

---

### 4. Context-Aware / LLM Chunking

Passes text segments to an LLM and asks it to identify **natural split points** — section starts, topic transitions, argument shifts.

**With Ollama (LLaMA 3):**
```python
OLLAMA_MODEL = "llama3"   # or llama3.1, mistral, phi3
```

The LLM receives numbered sentences and returns a JSON array of split indices:
```
Input:  [0] Sentence one. [1] Sentence two. [2] New topic starts here...
Output: [0, 2, 7, 14]   ← indices where new chunks begin
```

**Automatic fallback:** If Ollama is not running, the notebook switches to a heuristic splitter using paragraph boundaries + structural signals. No GPU or LLM required.

**Strength:** Most semantically meaningful boundaries — the model actually understands the text.

**Weakness:** Slowest method by far. Not practical for large corpora.

---

## 📊 What Gets Measured

Every chunking method reports:

| Metric | Description |
|--------|-------------|
| `# Chunks` | Total chunks produced across all documents |
| `# Docs` | Number of source documents covered |
| `Avg Chars` | Average characters per chunk |
| `Min / Max Chars` | Shortest and longest chunk |

Plus two auto-generated visualizations:
- **Chunk length distribution histograms** — how consistent or variable chunk sizes are per method
- **Chunks per document bar chart** — whether certain documents produce disproportionately more chunks

---

## 🔍 Chunk Metadata

Every chunk is stored as a Python dict carrying its source document:

```python
{
    "text"        : "Machine learning is the study of...",
    "doc_id"      : 1,
    "doc_title"   : "Machine learning",
    "chunk_index" : 4
}
```

This makes it easy to filter, group, or trace any chunk back to its origin document.

---

## 🔎 Query Similarity Test

The final cell runs a **cosine similarity search** across all chunk sets for a given query. This is a proxy test to visually compare which chunking method surfaces the most relevant content for the same question — not a full retrieval system.

```python
QUERY = "What are the main applications of machine learning?"
TOP_K = 3
```

Output shows score, chunk ID, and source document per method:

```
────────────────────────────────────────────────────────
  [Semantic]  Top-3 results
────────────────────────────────────────────────────────
  #1 score=0.590 | chunk_id=37 | doc='Machine learning'
  Classifiers and statistical learning methods...

  #2 score=0.589 | chunk_id=17 | doc='Artificial intelligence'
  There are several kinds of machine learning...
```

---

## 📈 Results (AI/ML Wikipedia Corpus)

Tested on the default 8-document corpus with the query *"What are the main applications of machine learning?"*:

| Method | Total Chunks | Avg Chars | Content Quality | Cosine Score |
|--------|-------------|-----------|----------------|-------------|
| Overlap | ~1800 | 600 | ⚠️ Mid-word cuts | 0.581 |
| Semantic | ~420 | variable | ✅ Coherent topics | 0.590 |
| Hierarchical (children) | ~3200 | 200 | ✅ Clean sentences | 0.607 |
| Context-Aware | ~380 | variable | ✅ Natural boundaries | 0.592 |

**Key finding:** Hierarchical children score highest numerically because smaller, denser chunks have higher cosine similarity to a focused query. However, semantic chunking produces the most readable and coherent content — the score gap is partly a chunk-size artifact. Overlap consistently performs worst due to broken boundaries.

---

## 📝 License

MIT — free to use, modify, and build on.

## 🚀 Getting Started

### Google Colab (recommended)

1. Open the notebook in Colab
2. Set runtime to **GPU** — Runtime → Change runtime type → T4 GPU
3. Run all cells — documents are fetched automatically

### Local

```bash
pip install sentence-transformers transformers nltk requests scikit-learn pandas matplotlib
jupyter notebook rag_chunking_multidoc.ipynb
```

### With LLaMA (context-aware chunking)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull llama3
```

The notebook auto-detects Ollama and switches to LLM chunking. If not found, falls back to heuristic automatically.

> **TPU note:** Ollama does not support Google Colab TPU. On TPU the notebook falls back to heuristic chunking automatically. For the embedding model, `TinyLlama/TinyLlama-1.1B-Chat-v1.0` or `google/gemma-2b-it` (HuggingFace token required) are compatible alternatives.

---

## ⚙️ Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `WIKI_TITLES` | 8 articles | Wikipedia articles to fetch as the corpus |
| `CHUNK_SIZE` | 600 | Overlap: characters per chunk |
| `OVERLAP` | 150 | Overlap: shared characters between chunks |
| `SIM_THRESHOLD` | 0.75 | Semantic: cosine similarity split threshold |
| `PARENT_SIZE` | 1000 | Hierarchical: characters per parent |
| `CHILD_SIZE` | 200 | Hierarchical: target characters per child |
| `OLLAMA_MODEL` | `llama3` | LLM model for context-aware chunking |
| `QUERY` | (example) | Query used in the similarity test |
| `TOP_K` | 3 | Number of results shown per method |

---

## 🛠️ Dependencies

| Package | Purpose |
|---------|---------|
| `sentence-transformers` | Sentence embeddings (all-MiniLM-L6-v2) |
| `nltk` | Sentence tokenization |
| `scikit-learn` | Cosine similarity |
| `requests` | Wikipedia API |
| `pandas` | Summary tables |
| `matplotlib` | Visualizations |
| `ollama` | *(optional)* LLM-based context-aware chunking |

---
