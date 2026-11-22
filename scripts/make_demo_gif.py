"""Record docs/figures/demo.gif: load the example in the web UI, verify it, and walk through the verdicts.

Needs a running service (`make serve`) and Playwright with a local Chrome (`pip install playwright`; uses the installed
Google Chrome, no browser download).
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
URL = "http://localhost:8060/"
WIDTH, HEIGHT, OUT_WIDTH = 1240, 860, 900


def shot(page, frames: list, seconds: float) -> None:
    img = Image.open(io.BytesIO(page.screenshot())).convert("RGB")
    img = img.resize((OUT_WIDTH, round(img.height * OUT_WIDTH / img.width)), Image.LANCZOS)
    frames.append((img, int(seconds * 1000)))


def main() -> None:
    frames: list = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT}, color_scheme="dark")
        page.goto(URL)
        shot(page, frames, 1.5)
        page.click("#example")
        shot(page, frames, 2.0)
        page.click("#go")
        page.wait_for_timeout(150)
        shot(page, frames, 1.0)
        page.wait_for_selector(".claim", timeout=120000)
        page.evaluate("window.scrollTo(0, document.querySelector('#result').offsetTop - 20)")
        shot(page, frames, 2.5)
        claims = page.query_selector_all(".claim")
        for c in claims:
            c.query_selector("summary").click()
            c.scroll_into_view_if_needed()
            page.evaluate("(el) => window.scrollTo(0, el.getBoundingClientRect().top + window.scrollY - 140)", c)
            page.wait_for_timeout(200)
            shot(page, frames, 2.6)
        browser.close()
    imgs, durations = zip(*frames, strict=True)
    palette = imgs[-1].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    quantized = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in imgs]
    out = ROOT / "docs" / "figures" / "demo.gif"
    quantized[0].save(out, save_all=True, append_images=quantized[1:], duration=list(durations), loop=0, optimize=True)
    print(f"{out.relative_to(ROOT)}: {len(frames)} frames, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    t0 = time.time()
    main()
    sys.exit(0)
