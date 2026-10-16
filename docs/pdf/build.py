"""Build the PDF sources from the markdown beside this file.

    python3 build.py                 every document
    python3 build.py patent report   just those

    chromium --headless --no-pdf-header-footer \
      --print-to-pdf=occam_patent_candidates.pdf patent_candidates.html

The documents were moved out of the repository to /home/shogo/Docs2/Occam,
so the markdown, this script and the rendered output all sit in one
directory, under docs/: the markdown beside the other documents and the
rendered output in docs/pdf/.

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

# The documents lived outside the repository for two days, while a patent was
# still being considered and only prompt_format.md had to stay behind -- a
# test reads it and checks every library spelling in it against the parser.
# The project is published under Apache-2.0 instead, so they are all back and
# the split, and the OCCAM_REPO it needed, are gone.
# Each entry: source markdown, output html, title, subtitle, and an optional
# (start marker, regex of sections to drop) pair.
DOCUMENTS = {
    "provenance": {
        "source": ROOT / "docs" / "provenance.md",
        "out": ROOT / "docs" / "pdf" / "provenance.html",
        "date": "2026-10-16",
        "title": "Occam — 出所の棚卸し",
        "subtitle": "どの機構が誰のもので、どこからがここで組んだものか",
        "start": "**先に書いておくこと。**",
        "drop": None,
        "warn": "<strong>新規性の主張ではない。</strong>"
                "先行技術調査は含まない。"
                "引用すべき相手を取り違えないための、出所の地図。",
    },
    "report": {
        "source": ROOT / "docs" / "final_report.md",
        "out": ROOT / "docs" / "pdf" / "final_report.html",
        "date": "2026-10-12",
        "title": "Occam — 最終報告書",
        "subtitle": "どの機構が誰のもので、どこからがここで組んだものか",
        "start": "## 0. 要約",
        "drop": None,
        "warn": "<strong>すべての数字は現行コードでの実測。</strong>"
                "撤回したものは撤回として残す。外部検証はゼロ、実験はゼロ。",
    },
    "prompt": {
        "source": ROOT / "docs" / "prompt_format.md",
        "out": ROOT / "docs" / "pdf" / "prompt_format.html",
        "date": "2026-10-13",
        "title": "Occam — プロンプトの書き方と条件",
        "subtitle": "閉じた語彙への決定的な照合。推測はしない",
        "start": "## 1. 必須の4つ",
        "drop": None,
        "warn": "<strong>本文の綴りは全行を parse に通して確かめてある。</strong>"
                "ライブラリ11・nitrogen 6・rotatable 10・weight 5・成果物3・"
                "strict 6・停止14・破棄8。実装とずれたら実装が正しい。",
    },
    "algorithms": {
        "source": ROOT / "docs" / "algorithms.md",
        "out": ROOT / "docs" / "pdf" / "algorithms.html",
        "date": "2026-10-12",
        "title": "Occam — アルゴリズム解剖書",
        "subtitle": "全モジュールの機構・定数・不変条件",
        "start": "## 0. 読み方",
        "drop": None,
        "warn": "<strong>すべて実装から転記。</strong>"
                "定数・閾値・正規表現は出典の行から取っている。"
                "実装とずれたら実装が正しい。",
    },
}

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


def build(spec):
    # Every document states its own date. It used to be one literal in
    # the shared template, so three documents written on different days
    # all carried the day the first one was rendered.
    if "date" not in spec:
        raise KeyError(f'{spec["title"]!r} has no "date"')
    src = spec["source"]
    if not src.exists():
        raise SystemExit(
            f"{src} does not exist. Nothing is rendered rather than "
            f"rendering a document from a source that is not there.")
    body = src.read_text()
    body = body[body.index(spec["start"]):]
    if spec["drop"]:
        body = re.sub(spec["drop"], "", body, flags=re.S)
    spec["out"].write_text(f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<title>{spec["title"]}</title>
<style>{CSS}</style></head><body>
<div class="head">
  <h1>{spec["title"]}</h1>
  <p class="sub">{spec["date"]} ／ {spec["subtitle"]}</p>
</div>
<div class="warn">{spec["warn"]}</div>
{chr(10).join(convert(body))}
</body></html>""")
    print(f"-> {spec['out'].name}")


def main():
    wanted = sys.argv[1:] or list(DOCUMENTS)
    for name in wanted:
        if name not in DOCUMENTS:
            print(f"unknown document: {name}; have {list(DOCUMENTS)}")
            return 1
        build(DOCUMENTS[name])
    return 0


if __name__ == "__main__":
    sys.exit(main())
