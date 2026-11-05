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

# Change the model without touching code: set GEMINI_MODEL in your environment
# (or in Hugging Face Space "Variables"). Check current free models at
# https://ai.google.dev/gemini-api/docs/models
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
TIMEOUT_MS = 20_000

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
        response = _get_client().models.generate_content(
            model=MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
            ),
        )
        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("Empty response from Gemini")
        return text, "gemini"
    except Exception as e:   # any failure -> safe template, app keeps working
        print(f"[summary] Gemini failed, using template: {type(e).__name__}: {e}")
        return template_summary(label, confidence, pneumonia_prob, chunks), "template"


if __name__ == "__main__":
    fake_chunks = [{"title": "Test note", "source": "", "text": "Pneumonia can appear as white patches."}]
    text, source = summarize("PNEUMONIA", 0.93, 0.93, fake_chunks)
    print(f"Source: {source}\n\n{text}")
