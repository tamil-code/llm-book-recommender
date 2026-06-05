---
title: LLM Book Recommender
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
---

<!-- The YAML block above configures the Hugging Face Space. GitHub renders it as a small table; it is required for Spaces to detect the Gradio SDK. -->

# LLM Book Recommender

A semantic, LLM-powered book recommendation system built on top of the publicly available [`dylanjcastillo/7k-books-with-metadata`](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata) Kaggle dataset.

Given a free-form natural-language query (e.g. *"give me some self motivation books"* or *"a sci-fi novel about time travel"*), the system returns the most semantically relevant books from a curated catalog of ~7,000 titles. It also uses a zero-shot classifier to enrich the dataset with simplified Fiction / Nonfiction categories.

**Live demo:** https://huggingface.co/spaces/<your-username>/llm-book-recommender (replace with your Space URL after deploying)

**Tech stack:** Gradio, LangChain, Chroma, sentence-transformers (`all-MiniLM-L6-v2`), pandas.

---

## Overview

The project is a single, end-to-end research notebook (`main.ipynb`) that walks through:

1. **Data acquisition** — downloads the 7K books dataset from Kaggle Hub.
2. **Exploratory data analysis** — null-value heatmaps, correlation matrices, and category distributions using `pandas`, `matplotlib`, and `seaborn`.
3. **Data cleaning** — drops rows with missing critical fields (`description`, `published_year`, `num_pages`, `average_rating`) and keeps only books whose description has more than 25 words. The cleaned catalog is written to `cleaned_books.csv`.
4. **Tagged descriptions** — every cleaned description is prefixed with its `isbn13` and written to `tagged_description.txt`. The ISBN prefix is what later lets the vector search map a result document back to the row in the catalog.
5. **Embeddings + vector store** — descriptions are split into one document per line, embedded with the `sentence-transformers/all-MiniLM-L6-v2` model via LangChain's `HuggingFaceEmbeddings`, and indexed into a Chroma vector database.
6. **Semantic retrieval** — a `retrieve_semantic_search_results(query, top_k)` helper takes a natural-language query, runs a similarity search against Chroma, parses the leading ISBN from each match, and returns the matching rows of the original catalog.
7. **Zero-shot category classification** — uses the `facebook/bart-large-mnli` model via 🤗 `transformers` to classify books with missing `simple_categories` into `Fiction` / `Nonfiction`, plus a sanity-check accuracy loop over 300 known-Fiction samples.

A sample 384-dimensional embedding vector produced by `all-MiniLM-L6-v2` is checked in at `sample_embedding.txt` for reference.

### Files in this repo

| File | Purpose |
|---|---|
| `app.py` | Gradio web app — the entry point for the live demo / Hugging Face Space. |
| `app/retrieval.py` | Loads the persisted Chroma index plus the catalog and exposes `search(query, top_k)`. |
| `scripts/build_index.py` | One-time script that embeds the descriptions and writes the `chroma_db/` index. |
| `chroma_db/` | Persisted Chroma vector store (committed so the app starts without re-embedding). |
| `main.ipynb` | The full end-to-end notebook (EDA → cleaning → embeddings → semantic search → classification). |
| `requirements.txt` | Slim runtime dependencies for the app / Space. |
| `requirements-notebook.txt` | Full pinned dependencies for running `main.ipynb`. |
| `cleaned_books.csv` | Output of the cleaning step — the curated catalog used as the source of truth for results. |
| `tagged_description.txt` | One line per book, formatted as `<isbn13> <description>`. This is what gets embedded. |
| `sample_embedding.txt` | A sample serialized embedding vector, useful for debugging dimensionality. |
| `venv/` | Local virtual environment (not committed in clean checkouts). |

---

## How it works

```text
                 ┌─────────────────────────────┐
                 │ Kaggle: 7k Books w/ Metadata│
                 └──────────────┬──────────────┘
                                │ kagglehub.dataset_download
                                ▼
                       ┌────────────────┐
                       │  pandas EDA &  │
                       │  cleaning      │
                       └───────┬────────┘
                               │
              writes           ▼
        ┌─────────────► cleaned_books.csv
        │                      │
        │                      ▼
        │           tagged_description.txt
        │            (isbn13 + description per line)
        │                      │
        │                      ▼
        │      LangChain TextLoader + CharacterTextSplitter
        │                      │
        │                      ▼
        │     HuggingFaceEmbeddings (all-MiniLM-L6-v2)
        │                      │
        │                      ▼
        │              Chroma vector DB
        │                      │
        │   user query ───────►│ similarity_search(query, k)
        │                      ▼
        │           top-k matching documents
        │                      │
        │  parse leading isbn13│
        └──────────────────────┘
                               │
                               ▼
              filter cleaned_books by isbn13
                               │
                               ▼
                    Recommended books
```

In short:

- The **`isbn13` is embedded into the document itself** (as the first token on each line) so that after a similarity search we can split off the ISBN and look the book up in the cleaned dataframe.
- **Chroma** is created in-memory in the notebook via `Chroma.from_documents(...)` — there is no persistent on-disk vector store by default, so the index is rebuilt every time the notebook is re-run.
- The **zero-shot classification** step is independent of retrieval; it's used to back-fill `simple_categories` for books whose original Google Books category doesn't map cleanly to `Fiction` / `Children's Fiction` / `Nonfiction` / `Children's Nonfiction`.

---

## Getting started (development)

### 1. Prerequisites

- **Python 3.10+** (the dependency set, e.g. `numpy==2.2.4`, `torch==2.6.0`, requires a reasonably recent Python).
- **Git**.
- A **Kaggle account** plus an API token (`~/.kaggle/kaggle.json`) so `kagglehub` can download the dataset. Follow the [Kaggle API auth guide](https://www.kaggle.com/docs/api) if you haven't set this up.
- Roughly **2–3 GB of free disk** for the HuggingFace model cache (`all-MiniLM-L6-v2` and `facebook/bart-large-mnli`).
- macOS / Linux / Windows are all supported. On Apple Silicon, `torch` 2.6 will use MPS automatically where applicable.

### 2. Clone & enter the project

```bash
git clone <your-fork-url>
cd "4 - LLM book recommender"
```

### 3. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# .\venv\Scripts\activate       # Windows PowerShell
```

### 4. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> The `requirements.txt` is pinned to the exact versions the notebook was developed against. If you want a leaner install for just the retrieval pipeline, the minimum is roughly: `pandas`, `numpy`, `matplotlib`, `seaborn`, `kagglehub`, `langchain`, `langchain-community`, `langchain-chroma`, `chromadb`, `sentence-transformers`, `transformers`, `torch`, `tqdm`.

### 5. Configure Kaggle credentials

Place your Kaggle API token at:

- macOS / Linux: `~/.kaggle/kaggle.json`
- Windows: `%USERPROFILE%\.kaggle\kaggle.json`

and make sure it's readable only by you:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

### 6. Launch the notebook

```bash
jupyter lab          # or: jupyter notebook
```

Open `main.ipynb` and run the cells top-to-bottom. The first run will:

- Download the dataset via `kagglehub` (cached under `~/.cache/kagglehub`).
- Download the HuggingFace models on first use (cached under `~/.cache/huggingface`).
- Rebuild the Chroma index in memory.

Subsequent runs reuse all the caches and are much faster.

### 7. Try a query

Inside the notebook, after the Chroma index has been built, call:

```python
retrieve_semantic_search_results("give me some self motivation books", top_k=5)
```

You'll get back a dataframe slice of `cleaned_books` containing the 5 most semantically relevant books for that query.

---

## Running the web app (Gradio)

The notebook is for exploration; the shippable product is a small Gradio app that wraps the same semantic search. It reads a **persisted** Chroma index so it starts quickly instead of re-embedding the catalog on every launch.

### 1. Install the slim runtime dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` is intentionally minimal (Gradio, LangChain, Chroma, sentence-transformers, torch). For the full notebook environment use `requirements-notebook.txt` instead.

### 2. Build the vector index (one time)

```bash
python scripts/build_index.py
```

This reads `tagged_description.txt`, embeds each description with `all-MiniLM-L6-v2`, and writes the index to `chroma_db/`. It takes about a minute on CPU for ~5,000 books. The `chroma_db/` directory is committed to the repo, so you only need this step if you change the catalog.

### 3. Launch the app

```bash
python app.py
```

Open the printed local URL (default http://127.0.0.1:7860), type a query such as *"books about building good habits"*, and you'll get ranked book cards with cover art, ratings, and a similarity score.

---

## Deploying to Hugging Face Spaces

The repo is laid out so it works as a Gradio Space with no extra configuration: `app.py` is the entry point, `requirements.txt` holds the runtime deps, and the YAML block at the top of this README configures the Space.

1. Push the project to GitHub (the `venv/` is git-ignored; `chroma_db/`, `cleaned_books.csv`, and `tagged_description.txt` are committed).
2. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space), choose the **Gradio** SDK, and link this GitHub repo (or upload the files).
3. Confirm `app.py`, `requirements.txt`, `chroma_db/`, and `cleaned_books.csv` are present on the Space branch.
4. Wait for the build to finish, then copy the public URL into the **Live demo** link near the top of this README and onto your resume.

`cpu-basic` (the free tier) is enough for MiniLM over ~5,000 vectors. The largest committed file is the ~37 MB Chroma SQLite store, which is well under GitHub's 100 MB limit, so Git LFS is not required.

---

## Development tips

- **Don't re-download every time.** `kagglehub` caches the dataset; once it's downloaded you can comment out the download cell while iterating.
- **Rebuilding the vector index is the slow step.** Consider passing `persist_directory="./chroma_db"` to `Chroma.from_documents(...)` and using `Chroma(persist_directory=..., embedding_function=...)` on subsequent runs to avoid re-embedding ~7K descriptions every time.
- **CPU vs GPU.** `sentence-transformers` will auto-use CUDA if available, otherwise CPU. On Apple Silicon the MiniLM model is fast enough on CPU; the `bart-large-mnli` zero-shot loop is much slower and benefits significantly from a GPU.
- **Linting.** `ruff` is already pinned in `requirements.txt`. Run `ruff check .` to lint Python cells you've exported.
- **Convert notebook to script** for diffs / reviews: `jupyter nbconvert --to script main.ipynb`.

---

## Possible next steps

The retrieval pipeline now ships as a deployed Gradio app. Natural extensions:

- **Add emotion / tone tags** (e.g. joyful, suspenseful, contemplative) using another zero-shot pass and let users filter on them.
- **Swap the embedding model** for a more powerful one (e.g. `BAAI/bge-large-en-v1.5`) and compare retrieval quality.
- **Add re-ranking** with a cross-encoder (`sentence-transformers/ms-marco-MiniLM-L-6-v2`) after the initial Chroma similarity search.

---

## License

No license file is currently included. The underlying book metadata comes from the Kaggle dataset linked above and is subject to its own terms of use.
