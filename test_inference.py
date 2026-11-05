"""Run analyze() on every image in samples/ and save the heatmaps to outputs/."""
import time
from pathlib import Path

from PIL import Image

from src.inference import analyze

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

images = sorted(p for p in SAMPLES.rglob("*")
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
if not images:
    raise SystemExit(f"No images found in {SAMPLES}")

print(f"Found {len(images)} images. The first one is slower (model loading).\n")
print(f"{'file':40} {'folder':12} {'label':10} {'conf':>6} {'p(pneu)':>8} {'sec':>5}")

for path in images:
    start = time.perf_counter()
    label, conf, p_pneu, overlay = analyze(path)
    secs = time.perf_counter() - start

    Image.fromarray(overlay).save(OUT / f"{path.stem}_gradcam.png")
    folder = path.parent.name if path.parent != SAMPLES else "-"
    print(f"{path.name[:40]:40} {folder[:12]:12} {label:10} {conf:6.3f} {p_pneu:8.3f} {secs:5.1f}")

print(f"\nHeatmaps saved to: {OUT}")
