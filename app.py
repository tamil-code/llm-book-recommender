"""Gradio web app for the semantic book recommender.

Entry point for Hugging Face Spaces (``app_file: app.py``). Type a natural
language query and get back the most semantically similar books, rendered as
cards with cover art, ratings, and a short description.
"""

from __future__ import annotations

import html

import gradio as gr

from app.retrieval import search, warm_up

PLACEHOLDER_COVER = "https://placehold.co/128x193?text=No+Cover"

EXAMPLE_QUERIES = [
    "give me some self motivation books",
    "a sci-fi novel about space exploration",
    "cozy mystery set in a small town",
    "books about building good habits",
    "epic fantasy with dragons and magic",
    "true stories of survival and adventure",
]


def _stars(rating) -> str:
    try:
        value = float(rating)
    except (TypeError, ValueError):
        return ""
    full = int(round(value))
    return "★" * full + "☆" * (5 - full)


def _book_card(book: dict) -> str:
    title = html.escape(book["title"] or "Untitled")
    authors = html.escape(book["authors"] or "Unknown author")
    cover = html.escape(book["thumbnail"] or PLACEHOLDER_COVER)
    category = html.escape(book["categories"] or "")
    description = html.escape((book["description"] or "")[:320])
    if book["description"] and len(book["description"]) > 320:
        description += "…"

    rating = book["average_rating"]
    stars = _stars(rating)
    rating_text = f"{stars} {rating}" if stars else ""

    meta_bits = []
    if rating_text:
        meta_bits.append(rating_text)
    if book["published_year"]:
        meta_bits.append(str(book["published_year"]))
    if book["num_pages"]:
        meta_bits.append(f"{book['num_pages']} pages")
    meta_line = html.escape("  ·  ".join(meta_bits))

    return f"""
    <div style="display:flex;gap:16px;padding:16px;border:1px solid var(--border-color-primary,#e5e7eb);
                border-radius:12px;margin-bottom:14px;background:var(--background-fill-secondary,#fff);">
      <img src="{cover}" alt="cover" referrerpolicy="no-referrer"
           onerror="this.src='{PLACEHOLDER_COVER}'"
           style="width:96px;height:145px;object-fit:cover;border-radius:6px;flex-shrink:0;"/>
      <div style="flex:1;min-width:0;">
        <div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start;">
          <h3 style="margin:0;font-size:1.05rem;line-height:1.3;">{title}</h3>
          <span style="font-size:0.72rem;white-space:nowrap;padding:2px 8px;border-radius:999px;
                       background:var(--color-accent-soft,#eef2ff);color:var(--body-text-color,#374151);">
            score {book['score']}
          </span>
        </div>
        <div style="color:var(--body-text-color-subdued,#6b7280);font-size:0.9rem;margin:2px 0 6px;">{authors}</div>
        <div style="font-size:0.8rem;color:var(--body-text-color-subdued,#6b7280);margin-bottom:8px;">
          {meta_line}{('  ·  ' + category) if category else ''}
        </div>
        <p style="margin:0;font-size:0.88rem;line-height:1.45;">{description}</p>
      </div>
    </div>
    """


def recommend(query: str, top_k: int) -> str:
    books = search(query, top_k=int(top_k))
    if not query or not query.strip():
        return "<p style='padding:12px;'>Enter a description of what you feel like reading.</p>"
    if not books:
        return "<p style='padding:12px;'>No matching books found. Try a different phrasing.</p>"
    header = f"<p style='margin:0 0 12px;color:var(--body-text-color-subdued,#6b7280);'>Top {len(books)} matches</p>"
    return header + "".join(_book_card(book) for book in books)


def build_ui() -> gr.Blocks:
    with gr.Blocks(theme=gr.themes.Soft(), title="LLM Book Recommender") as demo:
        gr.Markdown(
            """
            # 📚 LLM Book Recommender
            Semantic search over 5,000+ books using sentence embeddings and a Chroma vector store.
            Describe a mood, topic, or theme in plain English — the ranking is by meaning, not keywords.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                query = gr.Textbox(
                    label="What do you feel like reading?",
                    placeholder="e.g. cozy mystery set in a small town",
                    lines=2,
                )
                top_k = gr.Slider(
                    minimum=3, maximum=15, value=8, step=1, label="Number of results"
                )
                search_btn = gr.Button("Search", variant="primary")
                gr.Examples(examples=[[q] for q in EXAMPLE_QUERIES], inputs=[query])
            with gr.Column(scale=2):
                results = gr.HTML(
                    "<p style='padding:12px;'>Results will appear here.</p>"
                )

        search_btn.click(recommend, inputs=[query, top_k], outputs=results)
        query.submit(recommend, inputs=[query, top_k], outputs=results)
    return demo


warm_up()
demo = build_ui()

if __name__ == "__main__":
    demo.launch()
