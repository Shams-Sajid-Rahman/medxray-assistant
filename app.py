"""
Gradio web app: upload a chest X-ray -> prediction, Grad-CAM heatmap,
grounded summary. Run locally with:  python app.py
"""
import os
import time
from collections import defaultdict, deque
from pathlib import Path
from threading import Lock

import gradio as gr

from src.inference import analyze
from src.retrieval import build_index, retrieve
from src.summary import DISCLAIMER, summarize

ROOT = Path(__file__).resolve().parent

# ---------------- Per-visitor request limit ----------------
MAX_REQUESTS = int(os.environ.get("MAX_REQUESTS", 10))       # requests...
WINDOW_SECONDS = int(os.environ.get("WINDOW_SECONDS", 3600))  # ...per hour
_history = defaultdict(deque)   # visitor IP -> times of recent requests
_lock = Lock()


def _visitor_id(request):
    if request is None:
        return "local"
    # On Hugging Face the real visitor IP is the first entry of x-forwarded-for
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _allowed(visitor):
    """Sliding window: allow at most MAX_REQUESTS in the last WINDOW_SECONDS."""
    now = time.time()
    with _lock:
        times = _history[visitor]
        while times and now - times[0] > WINDOW_SECONDS:
            times.popleft()
        if len(times) >= MAX_REQUESTS:
            wait_min = int((WINDOW_SECONDS - (now - times[0])) / 60) + 1
            return False, wait_min
        times.append(now)
        return True, 0


# ---------------- Main pipeline ----------------
def run(image, request: gr.Request):
    if image is None:
        raise gr.Error("Please upload a chest X-ray image first.")

    ok, wait_min = _allowed(_visitor_id(request))
    if not ok:
        raise gr.Error(f"Request limit reached ({MAX_REQUESTS} per hour). "
                       f"Please try again in about {wait_min} minutes.")

    label, confidence, p_pneu, overlay = analyze(image)

    query = ("pneumonia findings on a pediatric chest X-ray" if label == "PNEUMONIA"
             else "normal pediatric chest X-ray and false positives")
    chunks = retrieve(query, k=3)
    summary, source = summarize(label, confidence, p_pneu, chunks)

    result = {"PNEUMONIA": p_pneu, "NORMAL": 1 - p_pneu}
    refs = "\n".join(f"- [{c['title']}]({c['source']})" if c["source"]
                     else f"- {c['title']}" for c in chunks)
    summary_md = (f"{summary}\n\n**References used:**\n{refs}\n\n"
                  f"<sub>Summary by: {'Gemini' if source == 'gemini' else 'template (LLM unavailable)'}</sub>")
    return result, overlay, summary_md


# ---------------- Layout ----------------
examples = sorted(str(p) for p in (ROOT / "samples").rglob("*")
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[:6]

with gr.Blocks(title="Chest X-ray Assistant") as demo:
    gr.Markdown(
        "# Chest X-ray Assistant\n"
        "DenseNet121 pneumonia classifier with Grad-CAM heatmap and a short "
        "summary grounded in reference notes (RAG).\n\n"
        f"**⚠️ {DISCLAIMER}** Trained only on children's chest X-rays "
        "(Kermany et al., single hospital)."
    )
    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="pil", label="Chest X-ray", height=360)
            button = gr.Button("Analyze", variant="primary")
            if examples:
                gr.Examples(examples=examples, inputs=image_in, label="Sample X-rays")
        with gr.Column():
            label_out = gr.Label(label="Prediction", num_top_classes=2)
            heatmap_out = gr.Image(label="Grad-CAM heatmap (regions that influenced the model)",
                                   height=360)
    summary_out = gr.Markdown()

    button.click(run, inputs=image_in, outputs=[label_out, heatmap_out, summary_out])


if __name__ == "__main__":
    n_notes, n_chunks = build_index()          # build RAG index once at startup
    print(f"RAG index ready: {n_notes} notes, {n_chunks} chunks")
    demo.queue(default_concurrency_limit=2)    # CPU only: handle 2 requests at a time
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
