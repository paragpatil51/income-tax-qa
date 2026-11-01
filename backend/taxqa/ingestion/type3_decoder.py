"""Recover the text layer of PDFs whose text is drawn with Type3 bitmap fonts.

The official CBDT PDF of the Act (typeset in PageMaker, distilled with Acrobat)
draws almost all body text with Type3 fonts T1, T2, ... whose glyphs are 1-bit
bitmaps. Character codes are assigned in order of first use per font, and the
fonts have no ToUnicode map, so ordinary extraction yields gibberish.

The glyph *names*, however, keep the original code: /c100 is code 100 of the
source font, i.e. "d". The source encoding is Windows-1252 (/c151 is an em
dash, /c147 and /c148 are curly quotes, /c173 a soft hyphen), so

    code -> /Differences glyph name /cNNN -> bytes([NNN]).decode("cp1252")

recovers the exact text. Style (bold / italic) is not recorded anywhere, so it
is measured from the glyph bitmaps: bold glyphs carry more ink than the same
letter in the regular face, italic glyphs lean to the right.

The print also contains invisible filler glyphs that the typesetter used for
spacing: a letter or hyphen placed after a word space or a leading marker
("Provided -that", "Sthe", "(a)Sany", "16.-The"). They exist in the content
stream but leave no ink on the rendered page. Two tests catch them, applied to
every glyph that starts a word:
  * ink   - the page is rendered once and the core of the glyph's box is blank;
  * squeeze - the next glyph starts after well under the filler's own advance
    width (ratio ~0.2-0.75, real letters advance ~0.99), i.e. it is tucked
    under the following letter, where an ink test would see that letter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from statistics import median

import numpy as np

GLYPH_NAME = re.compile(r"c(\d{1,3})")
STYLE_LETTERS = set("abcdeghmnopqsuxz")   # letters whose ink and slant are comparable across faces
BOLD_INK_RATIO = 1.18                     # font is bold if its letters carry >= 18% more ink than regular
ITALIC_SLANT = 0.12                       # dx/dy lean of painted pixels
FILLER_ADVANCE = 0.8                      # actual / nominal advance below this marks a filler glyph
INK_DPI = 100                             # resolution of the page render used for the ink test
INK_MIN = 0.01                            # share of dark pixels in a glyph's core below which it is invisible


@dataclass
class _Glyph:
    char: str
    ink: float | None = None      # painted share of the glyph bitmap
    slant: float | None = None    # least-squares lean of painted pixels (x per unit y)


@dataclass
class Type3Font:
    codes: dict[int, _Glyph] = field(default_factory=dict)
    widths: dict[int, float] = field(default_factory=dict)   # code -> advance in text space units per point
    bold: bool = False
    italic: bool = False

    def char(self, code: int) -> str:
        if code in self.codes:
            return self.codes[code].char
        # PyMuPDF inserts a synthetic space (code 32) at wide gaps even when the font has no such code.
        return " " if code == 32 else "�"

    def decode(self, text: str) -> str:
        return "".join(self.char(ord(c)) for c in text)


def _bitmap_metrics(stream: bytes) -> tuple[float, float] | None:
    """Ink share and slant of an inline 1-bit image mask inside a glyph procedure."""
    m = re.search(rb"BI\s*(.*?)\s*ID\s", stream, re.S)
    if not m:
        return None
    head = m.group(1)
    if re.search(rb"/(F|Filter)\b", head):
        return None   # compressed inline image: not used by this PDF, skip rather than guess
    w = re.search(rb"/W(?:idth)?\s+(\d+)", head)
    h = re.search(rb"/H(?:eight)?\s+(\d+)", head)
    if not (w and h):
        return None
    w, h = int(w.group(1)), int(h.group(1))
    row = (w + 7) // 8
    data = stream[m.end(): m.end() + row * h]
    if len(data) < row * h or w < 4 or h < 4:
        return None
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8).reshape(h, row), axis=1)[:, :w]
    painted = bits == 0                      # image mask, default Decode [0 1]: 0 paints
    ys, xs = np.nonzero(painted)
    if len(xs) < 10:
        return None
    y = (h - 1 - ys).astype(float)           # image rows run top-down; make y point up
    x = xs.astype(float)
    vy = float(np.var(y))
    slant = float(np.cov(x, y, bias=True)[0, 1] / vy) if vy > 0 else 0.0
    return float(painted.mean()), slant


class Type3Decoder:
    """Decodes Type3 spans of one document; caches per font xref and per glyph stream."""

    def __init__(self, doc):
        self.doc = doc
        self._fonts: dict[int, Type3Font | None] = {}
        self._metrics: dict[int, tuple[float, float] | None] = {}
        self._regular: dict[str, tuple[float, float]] | None = None

    # -- discovery -------------------------------------------------------------------

    def _obj(self, xref: int, key: str) -> str:
        kind, value = self.doc.xref_get_key(xref, key)
        if kind == "xref":
            return self.doc.xref_object(int(value.split()[0]))
        return value if kind != "null" else ""

    def _load(self, xref: int) -> Type3Font | None:
        enc = re.search(r"/Differences\s*\[(.*?)\]", self._obj(xref, "Encoding"), re.S)
        procs = dict(re.findall(r"/([^\s/]+)\s+(\d+)\s+0\s+R", self._obj(xref, "CharProcs")))
        if not enc:
            return None
        font = Type3Font()
        first = self.doc.xref_get_key(xref, "FirstChar")[1]
        matrix = re.findall(r"-?[\d.]+", self.doc.xref_get_key(xref, "FontMatrix")[1])
        scale = float(matrix[0]) if matrix else 0.001
        if first.isdigit():
            for i, w in enumerate(re.findall(r"-?[\d.]+", self._obj(xref, "Widths"))):
                font.widths[int(first) + i] = float(w) * scale
        code = 0
        for tok in re.findall(r"\d+|/[^\s/\[\]]+", enc.group(1)):
            if tok[0] != "/":
                code = int(tok)
                continue
            m = GLYPH_NAME.fullmatch(tok[1:])
            if not m or int(m.group(1)) > 255:
                return None   # not the /cNNN naming scheme: leave this font to normal extraction
            glyph = _Glyph(bytes([int(m.group(1))]).decode("cp1252", errors="replace"))
            proc = procs.get(tok[1:])
            if proc is not None and glyph.char in STYLE_LETTERS:
                pxref = int(proc)
                if pxref not in self._metrics:
                    self._metrics[pxref] = _bitmap_metrics(self.doc.xref_stream(pxref))
                if self._metrics[pxref]:
                    glyph.ink, glyph.slant = self._metrics[pxref]
            font.codes[code] = glyph
            code += 1
        return font

    def font(self, xref: int) -> Type3Font | None:
        if xref not in self._fonts:
            self._fonts[xref] = self._load(xref)
        return self._fonts[xref]

    def page_fonts(self, page) -> dict[str, Type3Font]:
        """Span font name -> decoder for the Type3 fonts of one page."""
        out = {}
        for xref, _ext, ftype, basefont, *_ in page.get_fonts():
            if ftype == "Type3":
                f = self.font(xref)
                if f is not None:
                    out[basefont] = f
        return out

    # -- style -----------------------------------------------------------------------

    def classify_styles(self) -> None:
        """Mark each loaded font bold/italic relative to the document's regular face.

        The regular face is the most common rendering of each letter, taken as the
        lowest-ink upright variant seen across all fonts (body text dominates).
        """
        per_char: dict[str, list[tuple[float, float]]] = {}
        for f in self._fonts.values():
            for g in (f.codes.values() if f else ()):
                if g.ink is not None:
                    per_char.setdefault(g.char, []).append((g.ink, g.slant))
        regular = {}
        for ch, vals in per_char.items():
            upright = [v for v in vals if abs(v[1]) < ITALIC_SLANT] or vals
            regular[ch] = (median(sorted(v[0] for v in upright)[: max(1, len(upright) // 2)]), 0.0)
        self._regular = regular
        for f in self._fonts.values():
            if not f:
                continue
            inks = [g.ink / regular[g.char][0] for g in f.codes.values() if g.ink is not None and regular.get(g.char)]
            slants = [g.slant for g in f.codes.values() if g.slant is not None]
            f.bold = bool(inks) and median(inks) >= BOLD_INK_RATIO
            f.italic = bool(slants) and median(slants) >= ITALIC_SLANT

    # -- span rewriting ----------------------------------------------------------------

    def rewrite_spans(self, page, blocks: list[dict]) -> int:
        """Give every span of a page's get_text('rawdict') blocks a decoded "text".

        Type3 spans are decoded and filler glyphs dropped; decoded spans get a
        synthetic font name ("Type3-Bold-Italic") so callers that look for
        "Bold"/"Italic" in the font name see the measured style. Other spans
        just get their characters joined. Returns the number of Type3 spans.
        """
        fonts = self.page_fonts(page)
        if not fonts:
            for block in blocks:
                for line in block.get("lines", []):
                    for span in line["spans"]:
                        span["text"] = "".join(c["c"] for c in span.get("chars", []))
            return 0
        ink = _InkMap(page)
        n = 0
        for block in blocks:
            for line in block.get("lines", []):
                # Flatten the line so filler detection can look across span boundaries.
                flat = []
                for si, span in enumerate(line["spans"]):
                    f = fonts.get(span["font"])
                    for c in span.get("chars", []):
                        code = ord(c["c"])
                        if f is None:
                            flat.append((si, c["c"], c["origin"][0], None, c["bbox"]))
                        else:
                            nominal = f.widths.get(code, 0.0) * span["size"]
                            flat.append((si, f.char(code), c["origin"][0], nominal, c["bbox"]))
                keep = [True] * len(flat)
                for i in range(len(flat)):
                    _, ch, x, nominal, bbox = flat[i]
                    prev = flat[i - 1][1] if i else " "   # fragment start (e.g. after an "(a)" marker) counts as a space
                    if not nominal or ch.isspace() or not (prev.isspace() or prev in ".)"):
                        continue
                    nxt = flat[i + 1] if i + 1 < len(flat) else None
                    squeezed = nxt is not None and not nxt[1].isspace() and (nxt[2] - x) < FILLER_ADVANCE * nominal
                    if squeezed or ink.blank(bbox):
                        keep[i] = False
                texts = [[] for _ in line["spans"]]
                for i, ((si, ch, _, _, _), k) in enumerate(zip(flat, keep)):
                    if k:
                        texts[si].append(ch)
                    elif 0 < i < len(flat) - 1 and not flat[i - 1][1].isspace() and not flat[i + 1][1].isspace():
                        texts[si].append(" ")   # the filler was the only gap: "(a)Sany" -> "(a) any"
                for si, span in enumerate(line["spans"]):
                    span["text"] = "".join(texts[si])
                    f = fonts.get(span["font"])
                    if f is not None:
                        span["font"] = "Type3" + ("-Bold" if f.bold else "") + ("-Italic" if f.italic else "")
                        n += 1
        return n


class _InkMap:
    """Greyscale render of one page, to ask whether a glyph box received any ink."""

    def __init__(self, page):
        import pymupdf

        pix = page.get_pixmap(dpi=INK_DPI, colorspace=pymupdf.csGRAY, alpha=False)
        self.scale = INK_DPI / 72.0
        self.x0, self.y0 = page.rect.x0, page.rect.y0
        self.img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)[:, : pix.width]

    def blank(self, bbox) -> bool:
        x0, y0, x1, y1 = bbox
        w, h = x1 - x0, y1 - y0
        # Core of the box: glyph boxes are generous (full ascent/descent), so trim the margins.
        c0 = int((x0 - self.x0 + 0.2 * w) * self.scale)
        c1 = int((x1 - self.x0 - 0.2 * w) * self.scale) + 1
        r0 = int((y0 - self.y0 + 0.2 * h) * self.scale)
        r1 = int((y1 - self.y0 - 0.2 * h) * self.scale) + 1
        core = self.img[max(0, r0): max(0, r1), max(0, c0): max(0, c1)]
        if core.size == 0:
            return False   # too small to judge: keep the glyph
        return float((core < 128).mean()) < INK_MIN


# A filler right after a leading marker ("16.-The") stood in for a space.
_GLUED_AFTER_MARKER = re.compile(r"^(\[?(?:\d{1,3}[A-Z]{0,5}\.|\([0-9A-Za-z]{1,6}\)))(?=[A-Za-z\[\"'])")


def clean_type3_text(text: str) -> str:
    return _GLUED_AFTER_MARKER.sub(r"\1 ", text)
