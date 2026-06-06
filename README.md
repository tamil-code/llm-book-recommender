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

# LLM Book Recommender

Semantic book search over ~7,000 titles — describe what you want in plain English and get ranked recommendations.

**Live demo → [huggingface.co/spaces/tamilbharathiaishitter/llm-book-recommender](https://huggingface.co/spaces/tamilbharathiaishitter/llm-book-recommender)**

<video src="docs/LLM Book Recommender.mp4" controls width="100%" style="max-width: 960px; border-radius: 12px;"></video>

## How it works

- Book descriptions are embedded with `all-MiniLM-L6-v2` and indexed in Chroma.
- A natural-language query runs similarity search; results map back to catalog rows via ISBN.
- A Gradio app serves the live demo on Hugging Face Spaces.

## Quick start

```bash
git clone https://github.com/tamil-code/llm-book-recommender.git
cd llm-book-recommender
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-app.txt
python app.py
```

Open http://127.0.0.1:7860 and try a query like *"cozy mystery set in a small town"*.

**Tech stack:** Gradio, LangChain, Chroma, sentence-transformers, pandas.
