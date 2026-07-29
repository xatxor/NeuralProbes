#! /usr/bin/env python

"""Typeset the twenty-five steering demonstrations as one PDF.

The document is a contents page, then a single page of run metadata set vertically centred, then the
twenty-five examples grouped under six themes. Each example prints the shared prompt once and both
steered responses under it, exactly as generated -- the markdown the model emitted is left visible
rather than rendered, so a reader sees the literal output and not an interpretation of it.

**LuaLaTeX, not XeLaTeX or pdfTeX.** Three constraints pick the engine between them. The responses
contain emoji that are part of the behaviour being demonstrated (the whole antagonist arm of the
first example is `Hello! <smiling face>`), and only `Apple Color Emoji` supplies them; XeTeX cannot
open that file at all, because it is a `.ttc` collection carrying sbix colour bitmaps, while
luaotfload loads it without complaint. The Russian names need a Unicode engine, which rules pdfTeX
out. So: LuaLaTeX.

**Three faces, for reasons rather than taste.** Latin Modern sets the body. Its OpenType files carry
no Cyrillic whatsoever -- this is why the pdfTeX route needs T2A and cm-super -- so Russian set in it
comes out as blank space with no warning of any kind. Linux Libertine covers Cyrillic and is used for
those names alone. Apple Color Emoji covers the emoji. Nothing else in this TeX Live install offers
Cyrillic outside of Church Slavonic.
"""

import json
import logging
import re
import subprocess
from argparse import ArgumentParser, Namespace
from pathlib import Path

log = logging.getLogger("render")

# Every character TeX would otherwise read as an instruction. Order matters: the backslash has to be
# replaced first or it would be re-escaped inside the replacements that introduce backslashes.
SPECIAL = [
    ("\\", r"\textbackslash{}"),
    ("{", r"\{"),
    ("}", r"\}"),
    ("$", r"\$"),
    ("&", r"\&"),
    ("#", r"\#"),
    ("^", r"\^{}"),
    ("_", r"\_"),
    ("%", r"\%"),
    ("~", r"\textasciitilde{}"),
    ("<", r"\textless{}"),
    (">", r"\textgreater{}"),
    ("|", r"\textbar{}"),
    ('"', r"\textquotedbl{}"),
    ("'", r"\textquotesingle{}"),
]

# Latin Modern has no glyph for any of these, and a missing glyph is silent -- it simply occupies no
# space. Each is handed to the emoji face individually rather than by a fallback rule, so a character
# that is not on this list fails loudly during the compile instead of vanishing from the page.
EMOJI = (
    "\u2728\u2705\U0001f308\U0001f30c\U0001f310\U0001f31f\U0001f331\U0001f389"
    "\U0001f3a8\U0001f495\U0001f499\U0001f49b\U0001f4a5\U0001f4bc\U0001f4c1"
    "\U0001f4d6\U0001f501\U0001f50d\U0001f525\U0001f527\U0001f604\U0001f60a"
    "\U0001f680\U0001f9c0\U0001f9e0\U0001f9e4\U0001f9e9\U0001f9ed"
)

# Box-drawing, which one response used to draw a directory tree. Neither Latin Modern nor an emoji
# font carries these; Menlo does. A separate face rather than a font-fallback rule, for the same
# reason as the emoji: an uncovered character must fail the build, not silently vanish from the page.
BOXES = "\u2500\u250c\u2510\u2514\u2518\u251c\u2524\u252c\u2534\u253c\u2502"


def escape(text: str, *, verbatim: bool, strip: bool = True) -> str:
    """Make one piece of model output safe to typeset without altering what it says.

    :param text: raw generated text.
    :param verbatim: when true, spaces and line breaks are preserved individually, which is what a
        transcript needs; when false the text is allowed to flow as a paragraph.
    :param strip: whether to drop surrounding whitespace. A generated response carries incidental
        leading and trailing newlines worth dropping, but a chat-templated prompt ends in the blank
        lines the model was actually handed, and those are part of the input.

    :return: LaTeX source.
    """
    for character, replacement in SPECIAL:
        text = text.replace(character, replacement)
    for character in EMOJI:
        text = text.replace(character, f"{{\\emojifont {character}}}")
    for character in BOXES:
        text = text.replace(character, f"{{\\boxfont {character}}}")
    if not verbatim:
        return re.sub(r"\s+", " ", text).strip()
    # A run of spaces collapses to one unless each is escaped, and a transcript's indentation is part
    # of what the model produced.
    lines = [line.replace(" ", "\\ ") for line in (text.strip() if strip else text).split("\n")]
    return "\\newline{}\n".join(line if line.strip() else "\\ " for line in lines)


def cyrillic(text: str) -> str:
    """Wrap Russian in the one installed face that can draw it.

    :param text: a translated class or concept name.

    :return: LaTeX source that selects Linux Libertine for the span.
    """
    return f"{{\\cyr {escape(text, verbatim=False)}}}"


PREAMBLE = r"""\documentclass[9pt,a4paper]{article}
\usepackage{fontspec}
\usepackage[a4paper,margin=0.7in]{geometry}
\usepackage{parskip}
\usepackage{enumitem}
\usepackage{hyperref}
\hypersetup{colorlinks=false,pdfborder={0 0 0},
  pdftitle={Steering Qwen3-8B along behavioural concept vectors},
  pdfsubject={Twenty-five demonstrations, one per class}}

%% Body face. Latin only -- see the module docstring for why that is not an oversight.
\setmainfont{lmroman10-regular.otf}[
  BoldFont=lmroman10-bold.otf,
  ItalicFont=lmroman10-italic.otf,
  BoldItalicFont=lmroman10-bolditalic.otf]
\setmonofont{lmmono10-regular.otf}[BoldFont=lmmonolt10-bold.otf]

%% \cyr -- Cyrillic, which the body face cannot draw at all.
\newfontface\cyr{LinLibertine_R.otf}[BoldFont=LinLibertine_RB.otf, ItalicFont=LinLibertine_RI.otf]

%% \emojifont -- loaded by path because the name lookup does not resolve a .ttc collection.
\newfontface\emojifont[Path=/System/Library/Fonts/,Renderer=Harfbuzz]{Apple Color Emoji.ttc}

%% \boxfont -- box-drawing characters, which neither of the other two faces covers.
\newfontface\boxfont{Menlo}

%% A transcript is a block of generated text reproduced exactly: every space, every line break, and
%% the markdown the model emitted left as literal characters. \sloppy because the escaped spaces
%% carry no stretchable glue, so TeX cannot justify these lines and would run into the margin.
\newenvironment{transcript}
  {\par\medskip\noindent\ttfamily\scriptsize\sloppy\raggedright
   \setlength{\parindent}{0pt}\setlength{\parskip}{0pt}}
  {\par\medskip}

\setcounter{tocdepth}{2}
\pagestyle{plain}
\begin{document}
"""

# Set vertically centred on a page of its own, immediately after the contents.
FRONT = r"""\clearpage
\thispagestyle{empty}
\null\vfill
\begin{center}
{\Large Steering Qwen3-8B along behavioural concept vectors}\\[4pt]
{\large Twenty-five demonstrations, one per class}
\end{center}
\vspace{18pt}
\begin{center}
\begin{minipage}{0.82\textwidth}
\begin{description}[leftmargin=0pt]
\item[Model] \texttt{Qwen/Qwen3-8B}, bf16, reasoning disabled.
\item[Vectors] 1036 contrast directions, construction \texttt{diff}, blocks 18 and 25 of 36.
\item[Steering] $h \mathrel{{+}{=}} \alpha N \hat{v}$ at one block, applied at all token positions, with
  $\alpha = 0.5$ and $N$ that layer's own mean residual norm -- 92.25 at block 18, 263.04 at block 25.
\item[Prompts] 1,108 curated from lmsys-chat-1m, eight per class, reviewer-chosen from a
  model-ranked shortlist.
\item[Judge] \texttt{gpt-oss-120b}, blind, with the presentation order randomised per comparison.
\item[Scale] 257,296 generations, 168,576 paired verdicts.
\item[Selection] 202 cells in which all four seeds agreed, mean lean was at least 0.8, the judge
  confirmed the prompt could express the concept, and neither arm was damaged; cut to 25 by reading
  the text, one per class.
\item[Ontology] The 148 classes and 1036 pairs are not ours. They come from
  \texttt{AntonKorznikov/feature\_stories} (Apache 2.0), which also supplied the story corpus the
  vectors were extracted from.
\end{description}
\end{minipage}
\end{center}
\vspace{18pt}
\begin{center}2026-07-29\end{center}
\vfill\null
\clearpage
"""


def main(args: Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    selected = json.loads(args.source.read_text())

    unknown = {
        character
        for entry in selected
        for field in ("prompt_templated", "response_baseline", "response_toward_concept",
                     "response_toward_antagonist")
        for character in entry[field]
        if ord(character) > 0x24ff and character not in EMOJI and character not in BOXES
    }
    assert not unknown, f"glyphs with no face to draw them: {[hex(ord(c)) for c in unknown]}"

    body = [PREAMBLE, r"\tableofcontents", FRONT]
    theme = None
    for entry in selected:
        if entry["theme"] != theme:
            theme = entry["theme"]
            body.append(f"\\section{{{escape(theme, verbatim=False)}}}")
        body += [
            f"\\subsection{{{escape(entry['class'], verbatim=False)}}}",
            r"\noindent",
            f"\\textbf{{{escape(entry['concept'], verbatim=False)}}} versus "
            f"\\textbf{{{escape(entry['antagonist'], verbatim=False)}}}\\\\",
            f"{cyrillic(entry['class_ru'])} --- {cyrillic(entry['concept_ru'])} \\textit{{vs}} "
            f"{cyrillic(entry['antagonist_ru'])}\\\\",
            f"Vector {entry['pair']}, block {entry['layer']} of 36, construction \\texttt{{diff}}. "
            f"Against the unsteered model: {entry['lean_plus_vs_baseline']:+.2f} for the concept arm, "
            f"{entry['lean_minus_vs_baseline']:+.2f} for the antagonist arm.",
            "",
            r"\medskip\noindent\textbf{Prompt}, exactly as the model received it",
            r"\begin{transcript}",
            escape(entry["prompt_templated"], verbatim=True, strip=False),
            r"\end{transcript}",
            f"\\noindent\\textbf{{Unsteered}} --- no intervention, seed {entry['seed_shown']}",
            r"\begin{transcript}",
            escape(entry["response_baseline"], verbatim=True),
            r"\end{transcript}",
            f"\\noindent\\textbf{{Steered toward {escape(entry['concept'], verbatim=False)}}}",
            r"\begin{transcript}",
            escape(entry["response_toward_concept"], verbatim=True),
            r"\end{transcript}",
            f"\\noindent\\textbf{{Steered toward {escape(entry['antagonist'], verbatim=False)}}}",
            r"\begin{transcript}",
            escape(entry["response_toward_antagonist"], verbatim=True),
            r"\end{transcript}",
            "",
        ]
    body.append(r"\end{document}")

    source = "\n".join(body)
    args.tex.parent.mkdir(parents=True, exist_ok=True)
    args.tex.write_text(source)
    log.info(f"wrote {args.tex} ({len(source):,} bytes)")

    if args.compile:
        # Twice: the contents page is written on the first pass and read back on the second.
        for attempt in (1, 2):
            done = subprocess.run(
                ["lualatex", "-interaction=nonstopmode", "-halt-on-error", args.tex.name],
                cwd=args.tex.parent, capture_output=True, text=True,
            )
            if done.returncode:
                tail = [line for line in done.stdout.splitlines() if line.startswith("!")][:5]
                log.error(f"pass {attempt} failed: {tail or done.stdout.splitlines()[-5:]}")
                raise SystemExit(1)
        pdf = args.tex.with_suffix(".pdf")
        pages = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
        log.info(f"wrote {pdf} ({pdf.stat().st_size:,} bytes)")
        for line in pages.splitlines():
            if line.startswith(("Pages:", "Page size:")):
                log.info(line)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("top25.json"))
    parser.add_argument("--tex", type=Path, default=Path(".bak/FINDINGS/top25.tex"))
    parser.add_argument("--compile", action="store_true", default=True)
    main(parser.parse_args())
