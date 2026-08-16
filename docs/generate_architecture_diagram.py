"""
docs/generate_architecture_diagram.py
────────────────────────────────────────
Legacy architecture-diagram generator retained for historical reference.

Deliberately dependency-free (plain string templating, no matplotlib/
graphviz) so the diagram can be regenerated in any environment without
installing anything extra — run:

    python docs/generate_architecture_diagram.py

The final release uses the checked-in docs/architecture_diagram.svg directly.
This older generator is not the release source of truth and should not be run
to replace the final diagram without first bringing its layout up to date.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parent / "architecture_diagram.svg"

# ── Palette (mirrors static/css/style.css's clinical-ledger tokens) ─────
INK = "#1c2333"
INK_SOFT = "#4b5566"
INK_FAINT = "#9aa1ad"
PAPER = "#faf8f4"
PAPER_RAISED = "#ffffff"
TEAL = "#1f6f63"
TEAL_DEEP = "#143f38"
TEAL_TINT = "#e3efec"
OCHRE = "#b9711f"
OCHRE_TINT = "#f5e9d6"

W, H = 1190, 700


@dataclass
class Box:
    id: str
    x: int
    y: int
    w: int
    h: int
    title: str
    subtitle: str = ""
    fill: str = PAPER_RAISED
    stroke: str = TEAL

    def anchor(self, side: str) -> tuple[float, float]:
        cx, cy = self.x + self.w / 2, self.y + self.h / 2
        return {
            "top": (cx, self.y),
            "bottom": (cx, self.y + self.h),
            "left": (self.x, cy),
            "right": (self.x + self.w, cy),
        }[side]


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def box_svg(b: Box) -> str:
    out = [
        f'<rect x="{b.x}" y="{b.y}" width="{b.w}" height="{b.h}" rx="10" '
        f'fill="{b.fill}" stroke="{b.stroke}" stroke-width="1.4" />'
    ]
    title_y = b.y + (23 if b.subtitle else b.h / 2 + 5)
    out.append(
        f'<text x="{b.x + b.w / 2}" y="{title_y}" text-anchor="middle" '
        f'font-family="IBM Plex Sans, sans-serif" font-size="13" font-weight="600" '
        f'fill="{INK}">{esc(b.title)}</text>'
    )
    for i, line in enumerate(b.subtitle.split("\n") if b.subtitle else []):
        out.append(
            f'<text x="{b.x + b.w / 2}" y="{title_y + 18 + i * 14}" text-anchor="middle" '
            f'font-family="IBM Plex Mono, monospace" font-size="10" fill="{INK_SOFT}">{esc(line)}</text>'
        )
    return "\n".join(out)


def straight(p1, p2, color=INK_FAINT, dashed=False, label="", label_pos=0.5, label_dy=-7):
    dash = ' stroke-dasharray="5,4"' if dashed else ""
    out = [
        f'<line x1="{p1[0]}" y1="{p1[1]}" x2="{p2[0]}" y2="{p2[1]}" stroke="{color}" '
        f'stroke-width="1.5"{dash} marker-end="url(#arrowhead)" />'
    ]
    if label:
        mx = p1[0] + (p2[0] - p1[0]) * label_pos
        my = p1[1] + (p2[1] - p1[1]) * label_pos + label_dy
        out.append(
            f'<text x="{mx}" y="{my}" text-anchor="middle" font-family="IBM Plex Mono, monospace" '
            f'font-size="10" fill="{INK_SOFT}">{esc(label)}</text>'
        )
    return "\n".join(out)


def elbow(points, color=INK_FAINT, dashed=False, label="", label_segment=0, label_dy=-7):
    """Draw a multi-segment arrow through `points`; arrowhead only on the last segment."""
    dash = ' stroke-dasharray="5,4"' if dashed else ""
    out = []
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i + 1]
        marker = ' marker-end="url(#arrowhead)"' if i == len(points) - 2 else ""
        out.append(
            f'<line x1="{p1[0]}" y1="{p1[1]}" x2="{p2[0]}" y2="{p2[1]}" '
            f'stroke="{color}" stroke-width="1.5"{dash}{marker} />'
        )
    if label:
        p1, p2 = points[label_segment], points[label_segment + 1]
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 + label_dy
        out.append(
            f'<text x="{mx}" y="{my}" text-anchor="middle" font-family="IBM Plex Mono, monospace" '
            f'font-size="10" fill="{INK_SOFT}">{esc(label)}</text>'
        )
    return "\n".join(out)


def build_svg() -> str:
    # ── Row 1: runtime request chain ────────────────────────────
    browser = Box("browser", 40, 70, 180, 70, "Browser", "static/index.html\ncss + vanilla js", fill=TEAL_TINT, stroke=TEAL_DEEP)
    api = Box("api", 290, 70, 180, 70, "Flask API", "api/routes.py")
    session = Box("session", 540, 70, 190, 70, "Session store", "chatbot/session_store.py")
    bot = Box("bot", 800, 70, 220, 70, "Chat state machine", "chatbot/bot.py\nChatSession")

    # ── Row 2: NLP utilities ─────────────────────────────────────
    lang = Box("lang", 40, 210, 170, 60, "Language detect", "utils/language_detector.py")
    norm = Box("norm", 235, 210, 190, 60, "Multilingual normalize", "utils/multilingual_normalizer.py")
    neg = Box("neg", 450, 210, 170, 60, "Negation (NegEx)", "utils/negation.py")
    flags = Box("flags", 645, 210, 190, 60, "Red-flag triage", "utils/red_flag_rules.py")
    decoder = Box("decoder", 860, 210, 200, 60, "Evidence bridge", "utils/ddxplus_decoder.py")

    # ── Row 3: ML + severity ─────────────────────────────────────
    predictor = Box(
        "predictor", 460, 340, 280, 90, "Structured ML predictor",
        "model/predictor.py\nTF-IDF -> linear classifier\n+ explain_case()",
        fill=TEAL_TINT, stroke=TEAL_DEEP,
    )
    severity = Box("severity", 810, 340, 220, 90, "Severity & response", "utils/severity_engine.py\nutils/response_summarizer.py\nutils/ollama_client.py*")

    # ── Row 4: offline training pipeline ──────────────────────────
    data = Box("data", 40, 530, 190, 70, "DDXPlus CSVs", "data/train.csv etc.\n(bring your own)", fill=OCHRE_TINT, stroke=OCHRE)
    train = Box("train", 290, 530, 230, 70, "Training pipeline", "model/train.py\nNB / SGD / LightGBM*")
    artifacts = Box("artifacts", 580, 530, 220, 70, "Model artifacts", "saved_models/*.pkl")

    boxes = [browser, api, session, bot, lang, norm, neg, flags, decoder, predictor, severity, data, train, artifacts]

    arrows = []
    # Row 1 chain
    arrows.append(straight(browser.anchor("right"), api.anchor("left"), color=TEAL_DEEP, label="JSON"))
    arrows.append(straight(api.anchor("right"), session.anchor("left"), label="session"))
    arrows.append(straight(session.anchor("right"), bot.anchor("left"), label="get/create"))

    # bot -> 4 of the 5 NLP utility boxes (fan-out, ordered left-to-right to avoid crossings)
    fan_targets = [lang, norm, neg, flags]
    fan_origin_xs = [820, 870, 920, 970]
    for target, ox in zip(fan_targets, fan_origin_xs):
        arrows.append(straight((ox, bot.y + bot.h), target.anchor("top")))
    arrows.append(straight(bot.anchor("right"), decoder.anchor("top")))

    # NLP utilities -> predictor
    arrows.append(straight(decoder.anchor("bottom"), predictor.anchor("top"), color=TEAL_DEEP, label="evidence codes"))
    arrows.append(straight(flags.anchor("bottom"), (predictor.x + 12, predictor.y)))

    # predictor -> severity
    arrows.append(straight(predictor.anchor("right"), severity.anchor("left"), label="scored Dx"))

    # severity -> bot: routed right of the decoder box to avoid overlapping it
    arrows.append(elbow(
        [severity.anchor("right"), (1120, severity.y + severity.h / 2), (1120, bot.y + bot.h / 2), bot.anchor("right")],
        color=TEAL_DEEP, dashed=True, label="reply + meta JSON", label_segment=1, label_dy=-8,
    ))

    # Training pipeline chain
    arrows.append(straight(data.anchor("right"), train.anchor("left")))
    arrows.append(straight(train.anchor("right"), artifacts.anchor("left")))
    arrows.append(elbow(
        [artifacts.anchor("top"), (artifacts.x + 110, predictor.y + predictor.h + 35), (predictor.x + 60, predictor.y + predictor.h)],
        color=OCHRE, dashed=True, label="loaded at inference time", label_segment=1, label_dy=18,
    ))

    legend = f"""
    <g transform="translate(40,600)">
      <rect x="0" y="0" width="14" height="14" rx="3" fill="{TEAL_TINT}" stroke="{TEAL_DEEP}" />
      <text x="20" y="11" font-family="IBM Plex Sans, sans-serif" font-size="11" fill="{INK_SOFT}">Request-path component</text>
      <rect x="225" y="0" width="14" height="14" rx="3" fill="{PAPER_RAISED}" stroke="{TEAL}" />
      <text x="245" y="11" font-family="IBM Plex Sans, sans-serif" font-size="11" fill="{INK_SOFT}">Supporting NLP utility</text>
      <rect x="420" y="0" width="14" height="14" rx="3" fill="{OCHRE_TINT}" stroke="{OCHRE}" />
      <text x="440" y="11" font-family="IBM Plex Sans, sans-serif" font-size="11" fill="{INK_SOFT}">Offline training pipeline</text>
      <line x1="660" y1="7" x2="700" y2="7" stroke="{INK_FAINT}" stroke-width="1.5" stroke-dasharray="5,4" />
      <text x="710" y="11" font-family="IBM Plex Sans, sans-serif" font-size="11" fill="{INK_SOFT}">Artifact / async hand-off</text>
    </g>
    """

    caption = (
        "One user message -&gt; Flask resolves the session -&gt; ChatSession runs language detection, multilingual "
        "normalization, negation-aware red-flag checks, and the evidence bridge -&gt; the structured TF-IDF + linear "
        "classifier scores 49 pathologies -&gt; severity triage and response formatting -&gt; JSON (reply + predictions "
        "+ explanation) streams back to the chat UI. *optional, off by default."
    )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="IBM Plex Sans, sans-serif">',
        f'<rect x="0" y="0" width="{W}" height="{H}" fill="{PAPER}" />',
        f'''<defs>
          <marker id="arrowhead" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill="{INK_FAINT}" />
          </marker>
        </defs>''',
        f'<text x="40" y="34" font-size="18" font-weight="600" fill="{INK}">HealthBot — request &amp; training architecture</text>',
        f'<text x="40" y="54" font-size="11.5" fill="{INK_FAINT}">Structured-evidence symptom checker &#183; Flask + vanilla JS &#183; TF-IDF / linear classifier</text>',
    ]
    parts.extend(arrows)
    parts.extend(box_svg(b) for b in boxes)
    parts.append(legend)
    parts.append(
        f'<foreignObject x="40" y="628" width="1020" height="60">'
        f'<text xmlns="http://www.w3.org/1999/xhtml" style="font-family:IBM Plex Sans, sans-serif; '
        f'font-size:11px; color:{INK_SOFT}; line-height:1.5;">{caption}</text>'
        f'</foreignObject>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    OUT_PATH.write_text(build_svg(), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
