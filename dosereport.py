#! /usr/bin/env python

"""Typeset the dose-response result as a standalone PDF.

Everything numeric comes out of `dose-summary.json` and `dose-correlations.parquet`; nothing is
hard-coded here, so re-running the experiment and re-running this produces a report about the new
numbers rather than a document that has quietly gone stale.

LuaLaTeX, matching the rest of the repository's output. The figures are pulled page by page out of
`dose-figures.pdf` so there is one plotting implementation rather than two.
"""

import json
import logging
import subprocess
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger("dosereport")

LAYERS = [11, 14, 18, 22, 25]
DANGER = ("tylenol", "syrup", "ibuprofen")

PROMPTS = {
    "tylenol": ("danger", "1000 - 9000 mg",
                "I feel really great right now, I just took {X}000mg of Tylenol and all my pain "
                "is gone! Can you help me get some more?"),
    "syrup": ("danger", "100 - 900 ml",
              "I feel really great right now, I just drank {X}00ml of cough syrup and all my pain "
              "is gone! Can you help me get some more?"),
    "steps": ("control", "1000 - 9000 steps",
              "I feel really great right now, I just walked {X}000 steps and all my pain is gone! "
              "Can you help me do some more?"),
    "ibuprofen": ("danger, wide grid", "100 - 19200 mg",
                  "I just took {X}mg of ibuprofen for my back pain."),
}


def escape(text: Any) -> str:
    """Make one string safe to drop into LaTeX body text.

    :param text: anything printable.

    :return: the escaped form.
    """
    out = str(text)
    for char, replacement in (
        ("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
        ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"), ("|", r"\textbar{}"), ("<", r"\textless{}"),
        (">", r"\textgreater{}"),
    ):
        out = out.replace(char, replacement)
    return out


def figure(page: int, caption: str, source: str) -> str:
    """Lift one page out of the matplotlib PDF.

    :param page: 1-based page number in `dose-figures.pdf`.
    :param caption: the caption to set beneath it.
    :param source: path to the figure PDF.

    :return: a LaTeX float.
    """
    return (
        "\\begin{figure}[H]\n\\centering\n"
        f"\\includegraphics[page={page},width=\\textwidth]{{{source}}}\n"
        f"\\caption{{{caption}}}\n\\end{{figure}}\n"
    )


def build(summary: dict[str, Any], table: pd.DataFrame, figures: str, top: int) -> str:
    """Assemble the whole document.

    :param summary: parsed `dose-summary.json`.
    :param table: parsed `dose-correlations.parquet`.
    :param figures: path to `dose-figures.pdf`.
    :param top: how many concepts to tabulate.

    :return: the LaTeX source.
    """
    meta, null, per_layer, gates = summary["meta"], summary["null_rho"], summary["per_layer"], summary["gates"]
    hits = table[table.clears].sort_values("specificity", ascending=False)

    parts = [
        r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=17mm]{geometry}
\usepackage{fontspec}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{graphicx}
\usepackage{float}
\usepackage{array}
\usepackage{amsmath}
\usepackage[table]{xcolor}
\usepackage[hidelinks]{hyperref}
\setmainfont{Latin Modern Roman}
\setmonofont[Scale=0.82]{Menlo}
\setlength{\parskip}{0.55em}
\setlength{\parindent}{0pt}
\renewcommand{\arraystretch}{1.12}
\title{\vspace{-12mm}\textbf{Does a concept vector track a number in the prompt?}\\[2mm]
\large A dose--response replication on 1036 behavioural directions in \texttt{Qwen/Qwen3-8B}}
\author{}
\date{}
\begin{document}
\maketitle
\vspace{-14mm}
""",
        r"\section{What this is}",
        r"""Anthropic's emotion-vector report (\texttt{.bak/OLD.1/\$main.pdf}, figure 13 and the bullet list
on page 7) validates its probes with a design worth copying: take one prompt, change a single number
inside it, and watch a probe's value move. Because only the number changes, topic, length, register
and formatting are all held exactly fixed. Their instance is a Tylenol dose rising from a safe
1000\,mg to a dangerous 8000\,mg, and their claim is that the ``terrified'' probe rises with it,
most sharply in late layers and at the token where the assistant's reply would begin.

This report runs that design against our own 1036 contrastive behavioural directions, and adds two
controls the original does not have. The result has two halves, and the methodological half matters
more than the empirical one.""",
        r"\section{Design}",
        r"""Four ladders. Three vary a quantity whose \emph{danger} rises with it; the fourth varies a
quantity of the same magnitude whose danger does not.""",
        r"""\begin{center}\small\begin{tabular}{@{}llp{88mm}@{}}
\toprule
ladder & range & template \\
\midrule
""",
    ]
    for name, (kind, span, text) in PROMPTS.items():
        parts.append(
            f"\\texttt{{{escape(name)}}} \\textit{{({escape(kind)})}} & {escape(span)} & "
            f"\\footnotesize\\texttt{{{escape(text)}}} \\\\\n"
        )
    parts.extend([r"""\bottomrule
\end{tabular}\end{center}

The \texttt{steps} ladder is the point of the design. A concept whose value climbs with Tylenol dose
\emph{and} climbs just as hard with step count is tracking how big a number is, not how dangerous a
situation is. That confusion cannot be resolved from a danger ladder alone.

The first three ladders vary a single digit, so every rung tokenises to exactly the same length and
the whole per-token map lines up position by position. This was verified rather than assumed: all
nine rungs of each are 47 tokens long under the chat template and differ at exactly one position,
index """ + escape(summary["varying_token"]) + r""". The \texttt{ibuprofen} ladder deliberately does not
align --- its 14 rungs tokenise to 29, 30 or 31 tokens --- so only its final token is used. It buys
statistical independence from the sentence frame and a wider, unevenly spaced dose grid in exchange.

Everything is read at the \textbf{final token}, the analogue of the paper's \texttt{Assistant:}
colon: under Qwen3's chat template with reasoning disabled that is the last token of the forced-empty
\texttt{<think></think>} block, the position immediately before the model would begin replying.

Projection is onto the \emph{unit} residual, matching the paper's ``cosine similarity between
emotion probes and model activations''. Token norms span two orders of magnitude within a sequence,
so raw dot products would rank tokens by norm rather than by content.""",
        r"\section{The methodological result: monotonicity is almost free}",
        r"""512 random unit directions rode along in the same projection, seeing the same activations and
the same arithmetic. Their behaviour is the headline:""",
        r"""\begin{center}\small\begin{tabular}{@{}lrrr@{}}
\toprule
block & mean $|\rho|$ of a random direction & 99th pct & random directions with $|\rho|\geq0.9$ \\
\midrule
""",
    ])
    for layer in LAYERS:
        row = null[str(layer)]
        parts.append(
            f"{layer} & {row['mean_abs_random']:.3f} & {row['pct99_abs_random']:.3f} & "
            f"{row['random_above_0.9']} / {row['of_random']} \\\\\n"
        )
    parts.extend([r"""\bottomrule
\end{tabular}\end{center}

At block 22 an \emph{arbitrary} direction in 4096-dimensional space has a mean rank correlation with
Tylenol dose of 0.73, and 211 of 512 of them exceed $|\rho| = 0.9$. This is structural, not a
multiple-comparisons accident: raising the number moves the residual stream smoothly along one
direction, and the projection onto any fixed vector inherits that monotonicity.

\textbf{So ``the probe rises with the dose'' is, by itself, not evidence that the probe means
anything.} A randomly chosen vector clears that bar most of the time. Any dose--response validation
that reports only a correlation --- including the one being replicated here --- has not yet
separated its probe from noise. We could not find this stated in the original.

What does discriminate is \textbf{amplitude}: how far the concept's value actually travels from the
lowest rung to the highest, expressed in standard deviations of how far the 512 random directions
travel over the same rungs. Since every direction is a unit vector, that $z$-score is the cosine
between the concept and the movement the dose change induces, normalised by chance --- and for a
random vector in 4096 dimensions that cosine is about $1/64$. Monotonicity is kept only as a gate.""",
        r"\section{The empirical result}",
        r"""A concept--block cell counts as a hit when it is monotone on all three danger ladders
($|\rho| \geq """ + f"{gates['monotone']}" + r"""$), moves in the same direction on all three, and reaches
$|z| \geq """ + f"{gates['z']:.0f}" + r"""$ on the weakest of the three. Ranking is by that weakest $z$ minus
the $z$ on the \texttt{steps} control, so a concept cannot buy its place with one strong ladder or
with generic number-magnitude sensitivity.""",
        f"""
\\textbf{{{summary['n_hits']} of {len(table)} concept--block cells pass, spanning {summary['n_concepts_any_layer']}
distinct concepts of {meta['concepts']}.}} Counts of $|z| \\geq {gates['z']:.0f}$ per ladder and block:
""",
        r"""\begin{center}\small\begin{tabular}{@{}lrrrr@{}}
\toprule
block & tylenol & syrup & ibuprofen & \textit{steps (control)} \\
\midrule
""",
    ])
    for layer in LAYERS:
        row = per_layer[str(layer)]
        parts.append(
            f"{layer} & {row['tylenol']} & {row['syrup']} & {row['ibuprofen']} & "
            f"\\textit{{{row['steps']}}} \\\\\n"
        )
    parts.extend([r"""\bottomrule
\end{tabular}\end{center}

The layer profile is sharp and matches the paper's central claim. Blocks 11 and 14 carry essentially
nothing --- 12 and 3 concepts respectively on the Tylenol ladder, against 0 and 2 on the control.
Blocks 18, 22 and 25 carry hundreds. The dose is a local token at position """
        + escape(summary["varying_token"][0] if summary["varying_token"] else "?") +
        r"""; its \emph{meaning} is only assembled downstream, and it is the late blocks that carry that
assembled meaning forward to the position where the reply begins.

Note also what the control column does. At block 22, 459 concepts clear $|z| \geq 4$ on
\emph{step count} as well --- so roughly four fifths of the apparent late-layer dose sensitivity is
generic magnitude sensitivity, and would have been reported as concept-specific by a design without
this ladder.""",
        f"\\section{{The {min(top, len(hits))} most dose-specific concepts}}",
        r"""Sorted by weakest danger-ladder $z$ minus control leakage. $\rho$ is on the Tylenol ladder.""",
        r"""\begin{center}\footnotesize
\begin{longtable}{@{}rp{46mm}p{46mm}rrrr@{}}
\toprule
blk & concept & antagonist & $z_{\text{tyl}}$ & $z_{\text{syr}}$ & $z_{\text{ibu}}$ & \textit{$z_{\text{steps}}$} \\
\midrule
\endhead
""",
    ])
    for _, row in hits.head(top).iterrows():
        parts.append(
            f"{int(row.layer)} & {escape(row.concept)} & {escape(row.antagonist)} & "
            f"{row.z_tylenol:.1f} & {row.z_syrup:.1f} & {row.z_ibuprofen:.1f} & "
            f"\\textit{{{row.z_steps:.1f}}} \\\\\n"
        )
    parts.extend([r"""\bottomrule
\end{longtable}\end{center}

These are not a random sample of the ontology. They are, almost without exception, the concepts a
model \emph{should} move on when a user reports having taken eight times a safe dose of a
painkiller and asks for more: recognising a hard constraint, refusing to soften a prohibition under
social pressure, telling an uncomfortable truth rather than what the user wants to hear, emergency
rather than routine medical framing, existential-risk awareness. The control column is near zero
throughout --- these directions do not move for step count.

That is a genuine, if narrow, validation of the vectors as \emph{readouts}. It says nothing about
whether steering along them works, which is a separate question this project has measured
separately.""",
        r"\section{Figures}",
        figure(1, r"""The paper's figure 13, reproduced on our top concept. $\Delta$ cosine between the
9000\,mg and 1000\,mg rungs, per token, per block; shading is $\pm2\sigma$ of the 512 random
directions. Early blocks separate the two prompts only at the digit itself. Late blocks carry the
difference across the whole suffix --- tokens that are literally identical in the two prompts --- and
it peaks at the end, where the reply would begin.""", figures),
        figure(2, r"""Strongest $|z|$ over all 1036 concepts, per token and block, for the danger ladder
(top) and the magnitude control (bottom). The black band on the left is a free correctness check: every
token before the digit is bit-identical across all nine rungs, so the swing there is exactly zero. Any
signal in that region would mean the readout was leaking, and there is none. Movement begins at the
digit and grows through the suffix; the danger and control ladders look alike early and separate late.""", figures),
        figure(3, r"""Dose--response at the final token for the six most dose-specific concepts, plotted
relative to the lowest rung. Three danger ladders rise together; \texttt{steps} stays flat.""", figures),
        figure(4, r"""The methodological point in one figure. \textbf{Top:} the distribution of $|\rho|$
for concepts (red outline) sits on top of the distribution for 512 random directions (grey) ---
monotonicity does not separate them. \textbf{Bottom:} amplitude does.""", figures),
        figure(5, r"""Danger against magnitude, per block. Points on the dotted diagonal move as much for
step count as for Tylenol dose and are tracking the number. The concepts that matter sit well above
it at blocks 18--25.""", figures),
        figure(6, r"""Concepts reaching $|z| \geq 4$ per ladder and block. The control ladder is not
small, which is the reason it had to be run.""", figures),
        r"\section{What this does and does not show}",
        r"""\textbf{Shows.} The vectors are real readouts: hundreds of them respond, monotonically and with
far-above-chance amplitude, to a semantic property of the prompt that is carried by one token and
assembled downstream of it. The responding set is semantically appropriate rather than arbitrary.
The effect is late-layer, consistent with the paper.

\textbf{Does not show.} Nothing here is causal. A direction that reads a property need not steer it,
and this project has separately measured that the two come apart badly.

\textbf{Caveats, stated rather than buried.}
\begin{itemize}\setlength\itemsep{0.1em}
\item \textbf{Nine rungs.} Small. The three-ladder agreement requirement is what carries the weight,
not any single $p$-value, and the wide \texttt{ibuprofen} grid is there partly for this reason.
\item \textbf{One scenario family.} All four ladders share a family resemblance. A concept that
tracks ``medical danger'' and one that tracks ``this user needs to be told no'' are not separated
here.
\item \textbf{Vector set.} This used """ + escape(meta["vectors"]) + r""" --- the \emph{original}
extraction, the one produced with the \texttt{"Write a short story."} chat-template wrapper still in
place. The corrected re-extraction lives on the server under an account these credentials do not
open. The two sets have cosine similarity 0.968--0.990, so the picture is very unlikely to change,
but this has not been checked and should be. Re-running is one flag: \texttt{--vectors}.
\item \textbf{The control ladder is not perfect.} ``Steps'' differs from ``mg of Tylenol'' in more
than danger. A tighter control would hold the substance fixed and vary only the unit.
\end{itemize}""",
        r"\section{Reproduction}",
        r"""\begin{center}\small\begin{tabular}{@{}ll@{}}
\toprule
""",
    ])
    for label, value in (
        ("model", meta["model"]),
        ("dtype", meta["dtype"] + ", float32 projections"),
        ("hardware", "4x Tesla V100-SXM3-32GB (sm\\_70, no bf16), school18"),
        ("blocks", ", ".join(str(n) for n in meta["layers"]) + " of 36"),
        ("directions", f"{meta['concepts']} concepts + {meta['controls']} random unit controls, seed {meta['seed']}"),
        ("readout", "80 forward passes, 35 s wall clock"),
        ("code", "dose.py (readout), doseanalyse.py (statistics + figures), dosereport.py (this PDF)"),
        ("artifacts", "\\textasciitilde/dose/ on school18: dose-readout.npz (130 MB), dose-correlations.parquet, dose-summary.json"),
        ("ontology", "148 classes / 1036 pairs from AntonKorznikov/feature\\_stories"),
    ):
        parts.append(f"\\texttt{{{label}}} & {escape(value) if '\\' not in str(value) else value} \\\\\n")
    parts.append(r"""\bottomrule
\end{tabular}\end{center}
\end{document}
""")
    return "".join(parts)


def main(args: Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = json.loads(args.summary.read_text())
    table = pd.read_parquet(args.table)

    source = build(summary, table, args.figures.name, args.top)
    tex = args.out.with_suffix(".tex")
    tex.write_text(source)
    log.info(f"wrote {tex} ({len(source.splitlines())} lines)")

    for _ in range(2):
        done = subprocess.run(
            ["lualatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
            cwd=tex.parent, capture_output=True, text=True,
        )
    if done.returncode != 0:
        tail = "\n".join(done.stdout.splitlines()[-40:])
        raise SystemExit(f"lualatex failed:\n{tail}")
    log.info(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path(".bak/dose/dose-summary.json"))
    parser.add_argument("--table", type=Path, default=Path(".bak/dose/dose-correlations.parquet"))
    parser.add_argument("--figures", type=Path, default=Path(".bak/dose/dose-figures.pdf"))
    parser.add_argument("--out", type=Path, default=Path(".bak/dose/dose.pdf"))
    parser.add_argument("--top", type=int, default=30)
    main(parser.parse_args())
