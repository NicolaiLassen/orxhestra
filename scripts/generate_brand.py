"""Generate the orxhestra brand system.

One source of truth for every logo asset.

The mark
--------
A **vesica-piscis X**: two pointed lens shapes on opposite diagonals,
each bounded by two circular arcs.  Aspect ratio (length : width) is
``3.0`` — slightly thinner than φ², slightly fatter than φ³.

The two lenses are different colours (mint signal under, paper cream
on top).  The cream lens is *also* used as a knockout mask against the
mint lens — slightly enlarged from the canvas centre — so where the
two strokes cross, a thin empty gap separates them.  This is the
defining touch of the mark.

Geometry recipe (per stroke)
----------------------------
- Endpoints: opposite corners (after a ``U/φ³`` corner padding).
- Half-length ``L = (U − 2·pad) · √2 / 2``.
- Aspect ``A = 3.0``.
- Arc radius     ``r = L · (A² + 1) / (2A)``.
- Arc centre dist ``c = L · (A² − 1) / (2A)``.
- Lens width     ``2·(r − c)``.

Knockout
--------
The cream lens, scaled by ``KNOCKOUT_SCALE`` (default ``1.08``) from
the canvas centre, is painted black inside an SVG ``<mask>`` that the
mint lens references — punching a slightly-larger cream-shaped hole.

Run
---
    python scripts/generate_brand.py

Writes ``web/brand/*.svg``, the preview page, and refreshes Mintlify-
consumable copies under ``docs/images/`` plus a GitHub-raw-friendly
copy under ``assets/``.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from textwrap import dedent

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

INK = "#0F0E13"      # warm near-black base
PAPER = "#F5F2EB"    # warm cream — primary mark stroke
SIGNAL = "#3FE0A8"   # mint-teal accent — secondary mark stroke
WHISPER = "#6B6872"  # muted text (not used in marks)
LINE = "#1E1C24"     # subtle dividers (not used in marks)

# ---------------------------------------------------------------------------
# Geometry constants
# ---------------------------------------------------------------------------

PHI = (1 + 5 ** 0.5) / 2
PHI2 = PHI ** 2
PHI3 = PHI ** 3
PHI4 = PHI ** 4

ASPECT = 3.0           # lens length : width
KNOCKOUT_SCALE = 1.08  # how much larger the knockout is than the cream lens
PAD_RATIO = 1 / PHI3   # corner padding (≈0.236 of canvas)

WORDMARK = "orxhestra"

# Paths
ROOT = Path(__file__).resolve().parent.parent
BRAND = ROOT / "web" / "brand"
DOCS_IMAGES = ROOT / "docs" / "images"
ASSETS = ROOT / "assets"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fmt(*vals: float) -> str:
    return " ".join(f"{v:.3f}".rstrip("0").rstrip(".") for v in vals)


Point = tuple[float, float]


def _lens_path(p1: Point, p2: Point, r: float) -> str:
    """Vesica-piscis lens between two points, bounded by two arcs (radius r).

    Both arcs use sweep=0; SVG's automatic centre-selection picks the
    two mirror-image circles that share the chord, producing a true
    pointed-tip lens.
    """
    return (
        f"M{fmt(p1[0])} {fmt(p1[1])} "
        f"A{fmt(r)} {fmt(r)} 0 0 0 {fmt(p2[0])} {fmt(p2[1])} "
        f"A{fmt(r)} {fmt(r)} 0 0 0 {fmt(p1[0])} {fmt(p1[1])} Z"
    )


def vesica_x_paths(
    size: float,
    *,
    offset: Point = (0, 0),
    aspect: float = ASPECT,
    pad_ratio: float = PAD_RATIO,
) -> tuple[str, str, float, Point]:
    """Compute the two diagonal lens paths for a vesica-X.

    Returns ``(main_path, anti_path, arc_radius, centre)`` where:

    * ``main_path`` is the lens on the TL→BR diagonal.
    * ``anti_path`` is the lens on the TR→BL diagonal.
    * ``arc_radius`` is the radius of every arc in both paths.
    * ``centre`` is the canvas centre, used as the knockout origin.
    """
    ox, oy = offset
    pad = size * pad_ratio
    tl = (ox + pad, oy + pad)
    br = (ox + size - pad, oy + size - pad)
    tr = (ox + size - pad, oy + pad)
    bl = (ox + pad, oy + size - pad)
    centre = (ox + size / 2, oy + size / 2)

    L_half = math.hypot(centre[0] - tl[0], centre[1] - tl[1])
    r = L_half * (aspect ** 2 + 1) / (2 * aspect)

    return _lens_path(tl, br, r), _lens_path(tr, bl, r), r, centre


def svg(view_box: str, body: str, defs: str = "", title: str = "orxhestra") -> str:
    defs_block = f"<defs>{defs}</defs>" if defs else ""
    return (
        f'<svg viewBox="{view_box}" xmlns="http://www.w3.org/2000/svg">'
        f"<title>{title}</title>"
        f"{defs_block}"
        f"{body}"
        f"</svg>"
    )


def _knockout_mask(
    mask_id: str,
    canvas_w: float,
    canvas_h: float,
    knockout_path: str,
    centre: Point,
    scale: float = KNOCKOUT_SCALE,
) -> str:
    """A mask: white everywhere, with `knockout_path` painted black —
    scaled by `scale` from `centre` so the hole is slightly larger
    than the original."""
    cx, cy = centre
    transform = f"translate({fmt(cx)} {fmt(cy)}) scale({fmt(scale)}) translate({fmt(-cx)} {fmt(-cy)})"
    return (
        f'<mask id="{mask_id}" maskUnits="userSpaceOnUse" '
        f'x="0" y="0" width="{fmt(canvas_w)}" height="{fmt(canvas_h)}">'
        f'<rect x="0" y="0" width="{fmt(canvas_w)}" height="{fmt(canvas_h)}" fill="white"/>'
        f'<path d="{knockout_path}" fill="black" transform="{transform}"/>'
        f"</mask>"
    )


def _mark_body(
    main_path: str,
    anti_path: str,
    centre: Point,
    canvas_w: float,
    canvas_h: float,
    color_main: str,
    color_anti: str,
    knockout: bool,
    mask_id: str,
) -> tuple[str, str]:
    """Compose the two-lens body (with optional knockout) + defs.

    Returns ``(body_svg, defs_svg)``.
    """
    if knockout:
        defs = _knockout_mask(mask_id, canvas_w, canvas_h, anti_path, centre)
        body = (
            f'<path d="{main_path}" fill="{color_main}" mask="url(#{mask_id})"/>'
            f'<path d="{anti_path}" fill="{color_anti}"/>'
        )
    else:
        defs = ""
        body = (
            f'<path d="{main_path}" fill="{color_main}"/>'
            f'<path d="{anti_path}" fill="{color_anti}"/>'
        )
    return body, defs


# ---------------------------------------------------------------------------
# Marks
# ---------------------------------------------------------------------------


def mark(
    size: float = 120,
    color_main: str = SIGNAL,
    color_anti: str = PAPER,
    knockout: bool = True,
    mask_id: str = "knockout",
) -> str:
    """Primary mark — vesica-X with knockout intersection.

    Mint lens on the main diagonal sits *behind* and is masked by an
    enlarged version of the cream anti-diagonal lens; the cream lens
    is then painted on top at original size.  The difference between
    the two reveals as a thin empty gap at the crossing.
    """
    main_path, anti_path, _r, centre = vesica_x_paths(size)
    body, defs = _mark_body(
        main_path, anti_path, centre,
        canvas_w=size, canvas_h=size,
        color_main=color_main, color_anti=color_anti,
        knockout=knockout, mask_id=mask_id,
    )
    return svg(f"0 0 {fmt(size)} {fmt(size)}", body, defs)


def mark_light(size: float = 120) -> str:
    """Light-background variant — pure mint, both lenses + white gap.

    Used on white surfaces (PyPI, GitHub light mode, plain README hosts).
    The knockout still cuts a white gap at the intersection so the X
    reads as two crossed strokes.
    """
    return mark(
        size=size, color_main=SIGNAL, color_anti=SIGNAL,
        knockout=True, mask_id="light-knockout",
    )


def mark_black(size: float = 120) -> str:
    """Pure-ink mark — full black, no mint accent.

    For B&W contexts, print, or any surface where the mint accent is
    inappropriate. Keeps the white knockout gap at the intersection.
    """
    return mark(
        size=size, color_main=INK, color_anti=INK,
        knockout=True, mask_id="black-knockout",
    )


def favicon_light(size: float = 32) -> str:
    """Favicon for light backgrounds — no tile, pure mint mark on transparent.

    The knockout reveals the white background through the gap so the
    two crossed strokes read distinctly even at 16×16.
    """
    main_path, anti_path, _r, centre = vesica_x_paths(size)
    body, defs = _mark_body(
        main_path, anti_path, centre,
        canvas_w=size, canvas_h=size,
        color_main=SIGNAL, color_anti=SIGNAL,
        knockout=True, mask_id="fav-light-knockout",
    )
    return svg(f"0 0 {fmt(size)} {fmt(size)}", body, defs)


def favicon(
    size: float = 32,
    bg: str = INK,
    color_main: str = SIGNAL,
    color_anti: str = PAPER,
) -> str:
    """Favicon — vesica-X inside a rounded-square ink tile.

    Identical geometry to the primary mark.
    """
    main_path, anti_path, _r, centre = vesica_x_paths(size)
    rx = size / PHI3
    body_paths, defs = _mark_body(
        main_path, anti_path, centre,
        canvas_w=size, canvas_h=size,
        color_main=color_main, color_anti=color_anti,
        knockout=True, mask_id="fav-knockout",
    )
    body = (
        f'<rect width="{fmt(size)}" height="{fmt(size)}" '
        f'rx="{fmt(rx)}" fill="{bg}"/>'
        f"{body_paths}"
    )
    return svg(f"0 0 {fmt(size)} {fmt(size)}", body, defs)


# ---------------------------------------------------------------------------
# Wordmark + lockups
# ---------------------------------------------------------------------------

# Sum of Geist Medium glyph advances for "orxhestra", expressed in em.
# Per-glyph widths (approx, in em):
#   o: 0.575  r: 0.395  x: 0.510  h: 0.580  e: 0.555
#   s: 0.515  t: 0.380  r: 0.395  a: 0.560
# Total = 4.465 em.  Updated from the prior 4.85 estimate which was
# leaving a ~12px right-side overhang on the lockup viewBox.
EM_TO_WIDTH = 4.465
LETTER_SPACING = -0.015
CAP_HEIGHT_RATIO = 0.72


def _text_metrics(font_size: float) -> tuple[float, float]:
    text_w = (
        EM_TO_WIDTH * font_size
        + LETTER_SPACING * font_size * (len(WORDMARK) - 1)
    )
    cap_h = CAP_HEIGHT_RATIO * font_size
    return text_w, cap_h


def _wordmark_text(
    x: float, baseline_y: float, font_size: float, color: str
) -> str:
    return (
        f'<text x="{fmt(x)}" y="{fmt(baseline_y)}" '
        f'font-family="Geist, Inter, -apple-system, sans-serif" '
        f'font-weight="500" font-size="{fmt(font_size)}" '
        f'letter-spacing="{LETTER_SPACING}em" '
        f'fill="{color}">{WORDMARK}</text>'
    )


def wordmark(font_size: float = 64, color: str = PAPER) -> str:
    text_w, cap_h = _text_metrics(font_size)
    pad = cap_h / PHI2
    view_w = text_w + 2 * pad
    view_h = cap_h + 2 * pad
    body = (
        f'<text x="{fmt(pad)}" y="{fmt(pad + cap_h)}" '
        f'font-family="Geist, Inter, -apple-system, sans-serif" '
        f'font-weight="500" font-size="{fmt(font_size)}" '
        f'letter-spacing="{LETTER_SPACING}em" '
        f'fill="{color}">{WORDMARK}</text>'
    )
    return svg(f"0 0 {fmt(view_w)} {fmt(view_h)}", body)


def lockup(
    font_size: float = 28,
    color: str = PAPER,
    accent: str = SIGNAL,
) -> str:
    """Horizontal lockup: vesica-X mark + wordmark side by side.

    Perfectly-spaced ratios — every visible gap equals ``lens_pad``:

    * mark height = ``cap_h · φ^1.5``   (mark ÷ cap_h ≈ 2.058)
    * lens_pad    = ``mark_size / φ³``  (the natural mark-canvas pad,
                                          used uniformly for every margin)
    * code gap    = ``0``               (visible mark-to-text gap = lens_pad)
    * right pad   = ``lens_pad``        (matches visible left margin)
    """
    cap_h = font_size * CAP_HEIGHT_RATIO
    mark_size = cap_h * (PHI ** 1.5)
    lens_pad = mark_size / PHI3

    total_h = mark_size
    baseline_y = (total_h + cap_h) / 2

    main_path, anti_path, _r, centre = vesica_x_paths(mark_size)
    text_w, _ = _text_metrics(font_size)
    total_w = mark_size + text_w + lens_pad

    body_paths, defs = _mark_body(
        main_path, anti_path, centre,
        canvas_w=total_w, canvas_h=total_h,
        color_main=accent, color_anti=color,
        knockout=True, mask_id="lockup-knockout",
    )
    body = body_paths + _wordmark_text(mark_size, baseline_y, font_size, color)
    return svg(f"0 0 {fmt(total_w)} {fmt(total_h)}", body, defs)


def lockup_stacked(
    mark_size: float = 200,
    color: str = PAPER,
    accent: str = SIGNAL,
) -> str:
    """Stacked lockup: mark above wordmark.  Used in hero / OG card.

    Perfectly-spaced ratios — every visible gap equals ``lens_pad``:

    * cap height = ``mark_size / φ³``   (mark : cap = φ³ ≈ 4.236)
    * lens_pad   = ``mark_size / φ³``   (the natural mark-canvas pad,
                                          used uniformly for every margin)
    * code gap   = ``0``                (visible mark-to-text gap = lens_pad)
    * top pad    = ``0``                (mark-canvas's own internal pad
                                          provides the visible top margin)
    * bottom pad = ``lens_pad``         (matches visible top)
    * left/right = ``lens_pad``         (uniform clear-space)
    """
    cap_h = mark_size / PHI3
    font_size = cap_h / CAP_HEIGHT_RATIO
    lens_pad = mark_size / PHI3
    text_w, _ = _text_metrics(font_size)

    # Visible top margin comes from the mark canvas's internal lens pad —
    # no extra top padding needed.  Bottom + left + right get explicit pad.
    total_w = max(mark_size, text_w) + 2 * lens_pad
    total_h = mark_size + cap_h + lens_pad

    mark_dx = (total_w - mark_size) / 2
    text_x = (total_w - text_w) / 2
    baseline_y = mark_size + cap_h

    main_path, anti_path, _r, centre = vesica_x_paths(
        mark_size, offset=(mark_dx, 0)
    )

    body_paths, defs = _mark_body(
        main_path, anti_path, centre,
        canvas_w=total_w, canvas_h=total_h,
        color_main=accent, color_anti=color,
        knockout=True, mask_id="stacked-knockout",
    )
    body = body_paths + _wordmark_text(text_x, baseline_y, font_size, color)
    return svg(f"0 0 {fmt(total_w)} {fmt(total_h)}", body, defs)


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------


def preview_html() -> str:
    return dedent(
        f"""\
        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="utf-8">
          <title>orxhestra — brand</title>
          <link rel="icon" type="image/svg+xml" href="favicon.svg">
          <link href="https://fonts.googleapis.com/css2?family=Geist:wght@300..700&family=Inter:wght@400..600&display=swap" rel="stylesheet">
          <style>
            :root {{
              --ink: {INK};
              --paper: {PAPER};
              --signal: {SIGNAL};
              --whisper: {WHISPER};
              --line: {LINE};
            }}
            * {{ box-sizing: border-box; }}
            body {{
              margin: 0;
              padding: 56px 32px 120px;
              background: var(--ink);
              color: var(--paper);
              font-family: 'Geist', 'Inter', system-ui, sans-serif;
              -webkit-font-smoothing: antialiased;
            }}
            header {{ max-width: 1200px; margin: 0 auto 64px; }}
            h1 {{ font-weight: 500; font-size: 36px; letter-spacing: -0.025em; margin: 0 0 12px; }}
            header p {{ color: var(--whisper); margin: 0; max-width: 64ch; line-height: 1.6; font-size: 15px; }}
            section {{ max-width: 1200px; margin: 0 auto 48px; }}
            h2 {{ font-weight: 500; font-size: 18px; letter-spacing: -0.01em; margin: 0 0 20px; }}
            .swatches {{ display: flex; gap: 12px; margin-top: 28px; flex-wrap: wrap; }}
            .sw {{ display: flex; align-items: center; gap: 10px; font-size: 13px; color: var(--whisper); }}
            .sw::before {{
              content: ""; width: 22px; height: 22px; border-radius: 6px;
              border: 1px solid var(--line); background: var(--c);
            }}
            .grid {{
              display: grid; gap: 20px;
              grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            }}
            .tile {{
              aspect-ratio: 1;
              display: grid; place-items: center;
              border: 1px solid var(--line);
              border-radius: 16px;
              padding: 48px;
              background:
                radial-gradient(ellipse 70% 50% at 50% 40%, rgba(63,224,168,0.05), transparent 70%),
                var(--ink);
            }}
            .tile img {{ width: 100%; height: 100%; object-fit: contain; }}
            figure {{ margin: 0; }}
            figcaption {{ margin-top: 12px; color: var(--whisper); font-size: 13px; }}
            figcaption strong {{ color: var(--paper); font-weight: 500; }}
            .panel {{
              border: 1px solid var(--line); border-radius: 18px;
              padding: 56px; background: var(--ink);
              display: grid; place-items: center;
            }}
            .panel img {{ max-width: min(92%, 560px); }}
            .panel.paper {{ background: #FFFFFF; }}
            .tile.paper {{ background: #FFFFFF !important; }}
            .zoom {{
              display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;
              margin-top: 16px;
            }}
            .zoom .tile {{ aspect-ratio: 1; padding: 18px; }}
            .zoom .tile img {{ width: 100%; height: 100%; }}
            .zoom-label {{ font-size: 12px; color: var(--whisper); margin-top: 8px; text-align: center; }}
          </style>
        </head>
        <body>
          <header>
            <h1>orxhestra · brand</h1>
            <p>
              Vesica-piscis X.  Each diagonal stroke is a pointed lens bounded
              by two circular arcs (aspect ratio L : W = {ASPECT}).  At the centre,
              the cream lens punches a slightly-enlarged knockout from the mint
              lens, leaving a thin empty gap that defines the crossing.  All
              dimensions derive from a single base unit <code>U</code> via powers
              of <code>φ</code>.
            </p>
            <div class="swatches">
              <span class="sw" style="--c: {INK}">ink · {INK}</span>
              <span class="sw" style="--c: {PAPER}">paper · {PAPER}</span>
              <span class="sw" style="--c: {SIGNAL}">signal · {SIGNAL}</span>
              <span class="sw" style="--c: {WHISPER}">whisper · {WHISPER}</span>
              <span class="sw" style="--c: {LINE}">line · {LINE}</span>
            </div>
          </header>

          <section>
            <h2>1 — Dark (primary)</h2>
            <p style="color: var(--whisper); margin: 0 0 20px; font-size: 14px; line-height: 1.6;">
              Mint main lens + cream anti lens on ink. Used on the site, OG card, GitHub dark mode.
            </p>
            <div class="grid">
              <figure>
                <div class="tile"><img src="icon.svg"></div>
                <figcaption><strong>icon.svg</strong></figcaption>
              </figure>
              <figure>
                <div class="tile"><img src="favicon.svg"></div>
                <figcaption><strong>favicon.svg</strong></figcaption>
              </figure>
            </div>
            <div style="height: 12px;"></div>
            <div class="panel"><img src="lockup.svg"></div>
            <div style="height: 12px;"></div>
            <div class="panel"><img src="lockup_stacked.svg"></div>
          </section>

          <section>
            <h2>2 — Light (white background)</h2>
            <p style="color: var(--whisper); margin: 0 0 20px; font-size: 14px; line-height: 1.6;">
              Pure mint <code>#3FE0A8</code> — both lenses + wordmark text — on white.  Used on PyPI, GitHub light mode, plain README hosts.
            </p>
            <div class="grid">
              <figure>
                <div class="tile paper"><img src="icon_light.svg"></div>
                <figcaption><strong>icon_light.svg</strong></figcaption>
              </figure>
              <figure>
                <div class="tile paper"><img src="favicon_light.svg"></div>
                <figcaption><strong>favicon_light.svg</strong></figcaption>
              </figure>
            </div>
            <div style="height: 12px;"></div>
            <div class="panel paper"><img src="lockup_light.svg"></div>
            <div style="height: 12px;"></div>
            <div class="panel paper"><img src="lockup_stacked_light.svg"></div>
          </section>

          <section>
            <h2>3 — Black (full mono)</h2>
            <p style="color: var(--whisper); margin: 0 0 20px; font-size: 14px; line-height: 1.6;">
              Pure ink everywhere, no mint accent. For B&W contexts, print, or monochrome embeds.
            </p>
            <div class="grid">
              <figure>
                <div class="tile paper"><img src="icon_black.svg"></div>
                <figcaption><strong>icon_black.svg</strong></figcaption>
              </figure>
            </div>
            <div style="height: 12px;"></div>
            <div class="panel paper"><img src="lockup_black.svg"></div>
            <div style="height: 12px;"></div>
            <div class="panel paper"><img src="lockup_stacked_black.svg"></div>
          </section>

          <section>
            <h2>Wordmark only</h2>
            <div class="panel"><img src="wordmark.svg"></div>
          </section>
        </body>
        </html>
        """
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def write(dst: Path, content: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(content, encoding="utf-8")


def main() -> None:
    BRAND.mkdir(parents=True, exist_ok=True)
    DOCS_IMAGES.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)

    # ----- Three canonical sets ---------------------------------------------
    # 1. Dark (primary): mint + cream on ink — for dark surfaces (the site,
    #    OG card, GitHub dark mode).
    # 2. Light:          mint + ink on white — for white surfaces (PyPI,
    #    GitHub light mode, plain README hosts).
    # 3. Black:          pure ink everywhere — for B&W contexts, print,
    #    monochrome embeds.
    # ------------------------------------------------------------------------

    # 1. Dark (primary)
    icon_svg = mark()
    write(BRAND / "icon.svg", icon_svg)
    write(BRAND / "mark.svg", icon_svg)            # alias
    write(BRAND / "favicon.svg", favicon())
    write(BRAND / "lockup.svg", lockup())
    write(BRAND / "lockup_stacked.svg", lockup_stacked())

    # 2. Light (white-bg)
    icon_light_svg = mark_light()
    write(BRAND / "icon_light.svg", icon_light_svg)
    write(BRAND / "mark_light.svg", icon_light_svg)  # alias
    write(BRAND / "favicon_light.svg", favicon_light())
    write(BRAND / "lockup_light.svg", lockup(color=SIGNAL, accent=SIGNAL))
    write(BRAND / "lockup_stacked_light.svg", lockup_stacked(color=SIGNAL, accent=SIGNAL))

    # 3. Black (pure ink)
    icon_black_svg = mark_black()
    write(BRAND / "icon_black.svg", icon_black_svg)
    write(BRAND / "mark_black.svg", icon_black_svg)  # alias
    write(BRAND / "lockup_black.svg", lockup(color=INK, accent=INK))
    write(BRAND / "lockup_stacked_black.svg", lockup_stacked(color=INK, accent=INK))

    # Wordmark only
    write(BRAND / "wordmark.svg", wordmark())
    write(BRAND / "preview.html", preview_html())
    write(
        BRAND / "README.md",
        "# orxhestra brand\n\nGenerated — do not edit by hand.\n"
        "Source of truth: `scripts/generate_brand.py`.\n"
        "Open `preview.html` to review.\n",
    )

    # Distribute to docs (Mintlify) and assets (raw GitHub URL).
    distributables = (
        "icon.svg", "lockup.svg", "lockup_stacked.svg", "favicon.svg", "mark.svg",
        "icon_light.svg", "lockup_light.svg", "lockup_stacked_light.svg",
        "favicon_light.svg",
        "icon_black.svg", "lockup_black.svg", "lockup_stacked_black.svg",
    )
    for fname in distributables:
        shutil.copy(BRAND / fname, DOCS_IMAGES / fname)
        shutil.copy(BRAND / fname, ASSETS / fname)

    print(f"Wrote to {BRAND}:")
    for p in sorted(BRAND.iterdir()):
        print(f"  {p.name}")
    print(f"Distributed {len(distributables)} files to {DOCS_IMAGES} and {ASSETS}:")
    for fname in distributables:
        print(f"  {fname}")


if __name__ == "__main__":
    main()
