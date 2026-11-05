"""
Short, grounded summary of a prediction.

summarize(label, confidence, pneumonia_prob, chunks) -> (text, source)
  source is "gemini" if the LLM answered, or "template" if it failed.

Only the label, the probabilities and the reference text are sent to Gemini.
The X-ray image is NEVER sent.
"""
import os

from google import genai
from google.genai import types

# Models are tried in this order. If one is busy (503), out of free quota (429)
# or not available (404), the next one is tried. Override without touching code
# with GEMINI_MODELS, e.g.  gemini-3.5-flash-lite,gemini-3.1-flash-lite
# Current models: https://ai.google.dev/gemini-api/docs/models
MODELS = [m.strip() for m in os.environ.get(
    "GEMINI_MODELS",
    "gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-3.8-flash",
).split(",") if m.strip()]
TIMEOUT_MS = 15_000   # per model, so the visitor never waits too long

DISCLAIMER = ("This is an AI demo for learning purposes, not a medical diagnosis. "
              "Always consult a qualified doctor.")

SYSTEM_PROMPT = """You write short explanations for a chest X-ray classification DEMO.
Rules:
- Use ONLY the reference notes given. Do not add medical facts that are not in them.
- Write 3-5 plain sentences for a non-expert reader.
- Say what the model predicted and how confident it was, in words and as a percentage.
- Mention one relevant limitation from the notes (for example: trained only on
  children's X-rays from one hospital, or tends to flag healthy X-rays as pneumonia).
- Never give a diagnosis, treatment or medication advice.
- Do not write a disclaimer; the app adds one."""

_client = None


def _get_client():
    global _client
    if _client is None:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        _client = genai.Client(api_key=key,
                               http_options=types.HttpOptions(timeout=TIMEOUT_MS))
    return _client


def _format_notes(chunks):
    return "\n\n".join(f"[{i+1}] {c['title']}\n{c['text']}" for i, c in enumerate(chunks))


def template_summary(label, confidence, pneumonia_prob, chunks):
    """Used when the API key is missing, the quota is used up, or the API fails."""
    if label == "PNEUMONIA":
        first = (f"The model flagged this X-ray as showing signs consistent with "
                 f"pneumonia, with {confidence:.0%} confidence.")
    else:
        first = (f"The model did not find signs of pneumonia in this X-ray "
                 f"({confidence:.0%} confidence that it looks normal).")
    titles = ", ".join(sorted({c["title"] for c in chunks})) or "none"
    return (f"{first} The heatmap shows which image regions most influenced the "
            f"prediction; it is not a map of disease. This model was trained only on "
            f"children's X-rays from a single source and tends to flag some healthy "
            f"X-rays as pneumonia. Related reference notes: {titles}.")


def summarize(label, confidence, pneumonia_prob, chunks):
    user_prompt = (
        f"Model prediction: {label}\n"
        f"Confidence in this prediction: {confidence:.1%}\n"
        f"Probability of pneumonia: {pneumonia_prob:.1%}\n\n"
        f"Reference notes:\n{_format_notes(chunks)}"
    )
    try:
        client = _get_client()
    except Exception as e:
        print(f"[summary] No Gemini client, using template: {e}")
        return template_summary(label, confidence, pneumonia_prob, chunks), "template"

    for model in MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.2,
                ),
            )
            text = (response.text or "").strip()
            if not text:
                raise RuntimeError("Empty response")
            print(f"[summary] Answered by {model}")
            return text, "gemini"
        except Exception as e:   # busy, quota, timeout... -> try the next model
            print(f"[summary] {model} failed: {type(e).__name__}: {str(e)[:120]}")

    print("[summary] All Gemini models failed, using template")
    return template_summary(label, confidence, pneumonia_prob, chunks), "template"


if __name__ == "__main__":
    fake_chunks = [{"title": "Test note", "source": "", "text": "Pneumonia can appear as white patches."}]
    text, source = summarize("PNEUMONIA", 0.93, 0.93, fake_chunks)
    print(f"Source: {source}\n\n{text}")
