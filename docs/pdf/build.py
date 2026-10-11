"""Build the patent-candidate PDF source from docs/patent_inventory.md.

    python3 docs/pdf/build.py
    chromium --headless --no-pdf-header-footer \
      --print-to-pdf=docs/pdf/occam_patent_candidates.pdf \
      docs/pdf/patent_candidates.html

The markdown is the only copy of the text. Extracting the sections here
rather than retyping them into a document is the same rule the rest of this
repository runs on: two copies of a statement drift, and the one handed to
someone outside is the copy nobody rereads.

The survey-scope section is dropped, being about this repository rather than
about the candidates.
"""

import html as html_mod
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE = ROOT / "docs" / "patent_inventory.md"
OUT = ROOT / "docs" / "pdf" / "patent_candidates.html"

CSS = """
@page { size: A4; margin: 18mm 16mm 20mm 16mm;
        @bottom-center { content: counter(page); } }
body { font-family: "Noto Sans CJK JP", sans-serif; font-size: 9.6pt;
       line-height: 1.65; color: #111; }
h1 { font-size: 17pt; margin: 0 0 2mm 0; }
h2 { font-size: 12.5pt; margin: 9mm 0 2mm 0; padding-bottom: 1.5mm;
     border-bottom: 1.6pt solid #111; page-break-after: avoid; }
h3 { font-size: 10.8pt; margin: 6mm 0 1.5mm 0; page-break-after: avoid; }
h4 { font-size: 10pt; margin: 4.5mm 0 1mm 0; color: #333;
     page-break-after: avoid; }
p { margin: 0 0 2.2mm 0; }
p.bullet { padding-left: 5mm; text-indent: -3mm; }
p.bullet:before { content: "— "; color: #666; }
p.num { padding-left: 6mm; text-indent: -6mm; }
table { border-collapse: collapse; width: 100%; margin: 2.5mm 0 4mm 0;
        font-size: 8.9pt; page-break-inside: avoid; }
th { background: #f0f0f0; text-align: left; font-weight: 600; }
th, td { border: 0.5pt solid #bbb; padding: 1.4mm 2.2mm;
         vertical-align: top; }
pre { background: #f6f6f6; border-left: 2.5pt solid #999; padding: 2.5mm 3mm;
      font-family: "DejaVu Sans Mono", monospace; font-size: 8.2pt;
      line-height: 1.45; white-space: pre-wrap; margin: 2.5mm 0 4mm 0;
      page-break-inside: avoid; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 8.6pt;
       background: #f2f2f2; padding: 0 0.6mm; }
blockquote { margin: 2.5mm 0; padding: 2mm 3mm; background: #fbf7e8;
             border-left: 2.5pt solid #c9a227; }
.head { border-bottom: 2.5pt solid #111; padding-bottom: 3mm;
        margin-bottom: 5mm; }
.sub { color: #555; font-size: 9pt; margin: 0; }
.warn { background: #fdf0f0; border: 0.8pt solid #c66; padding: 3mm;
        margin: 0 0 5mm 0; }
"""


def inline(text):
    text = html_mod.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)


def convert(body):
    out, in_table, in_code = [], False, False
    for line in body.split("\n"):
        if line.startswith("```"):
            out.append("</pre>" if in_code else "<pre>")
            in_code = not in_code
            continue
        if in_code:
            out.append(html_mod.escape(line))
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                continue
            tag = "td" if in_table else "th"
            if not in_table:
                out.append("<table>")
                in_table = True
            out.append("<tr>" + "".join(f"<{tag}>{inline(c)}</{tag}>"
                                        for c in cells) + "</tr>")
            continue
        if in_table:
            out.append("</table>")
            in_table = False
        stripped = line.strip()
        if not stripped or stripped == "---":
            continue
        heading = re.match(r"^(#{2,4})\s+(.*)$", stripped)
        if heading:
            level = len(heading.group(1))
            out.append(f"<h{level}>{inline(heading.group(2))}</h{level}>")
            continue
        item = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if item:
            out.append(f"<p class='num'><b>{item.group(1)}.</b> "
                       f"{inline(item.group(2))}</p>")
            continue
        if stripped.startswith("- "):
            out.append(f"<p class='bullet'>{inline(stripped[2:])}</p>")
            continue
        if stripped.startswith(">"):
            out.append(f"<blockquote>{inline(stripped.lstrip('> '))}"
                       f"</blockquote>")
            continue
        out.append(f"<p>{inline(stripped)}</p>")
    if in_table:
        out.append("</table>")
    return out


def main():
    src = SOURCE.read_text()
    # The disclaimer onwards; the survey-scope section is about the
    # repository, not the candidates.
    body = src[src.index("**判断できないことを先に書く。**"):]
    body = re.sub(r"## 0\. 全探索の範囲.*?(?=## 1\.)", "", body, flags=re.S)

    OUT.write_text(f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<title>Occam 技術の棚卸し（特許相談用）</title>
<style>{CSS}</style></head><body>
<div class="head">
  <h1>Occam — 技術の棚卸し（特許相談用）</h1>
  <p class="sub">2026-10-11 ／ グラム陰性菌スクリーニングの成立判定システム</p>
</div>
<div class="warn">
<strong>これは特許性の判断ではない。</strong>新規性・進歩性・先行技術調査は含まない。
弁理士に渡すための、技術の所在と、公知との境界の整理。
</div>
{chr(10).join(convert(body))}
</body></html>""")
    print(f"-> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
