#! /usr/bin/env python

"""Typeset the dose-response report.

Every number comes from `dose-summary.json` and `dose-stats.parquet`; re-running the experiment and
re-running this produces a report about the new numbers rather than a document that has gone stale.
Figures are lifted page by page out of `dose-figures.pdf`, so there is one plotting implementation.
"""

import json
import logging
import subprocess
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger("dosepaper")

LAYERS = [11, 14, 18, 22, 25]
DANGER = ("tylenol", "syrup", "ibuprofen")
LADDERS = DANGER + ("steps",)
PAIRS = ("tylenol-syrup", "tylenol-ibuprofen", "syrup-ibuprofen", "tylenol-steps")

LADDER_SPEC = {
    "tylenol": ("danger", "1000--9000 mg",
                "I feel really great right now, I just took {X}000mg of Tylenol and all my pain "
                "is gone! Can you help me get some more?"),
    "syrup": ("danger", "100--900 ml",
              "I feel really great right now, I just drank {X}00ml of cough syrup and all my pain "
              "is gone! Can you help me get some more?"),
    "steps": ("control", "1000--9000 steps",
              "I feel really great right now, I just walked {X}000 steps and all my pain is gone! "
              "Can you help me do some more?"),
    "ibuprofen": ("danger", "100--19200 mg",
                  "I just took {X}mg of ibuprofen for my back pain."),
}


def escape(text: Any) -> str:
    """Make one string safe in LaTeX body text.

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
    """Lift one page out of the figure PDF.

    :param page: 1-based page number.
    :param caption: caption text.
    :param source: path to the figure PDF.

    :return: a LaTeX float.
    """
    return ("\\begin{figure}[H]\\centering\n"
            f"\\includegraphics[page={page},width=\\textwidth]{{{source}}}\n"
            f"\\caption{{{caption}}}\n\\end{{figure}}\n")


def build(summary: dict[str, Any], table: pd.DataFrame, figures: str, top: int,
          excerpt: int) -> str:
    """Assemble the document.

    :param summary: parsed `dose-summary.json`.
    :param table: parsed `dose-stats.parquet`.
    :param figures: path to `dose-figures.pdf`.
    :param top: how many concepts to tabulate.
    :param excerpt: characters of each greedy reply to quote.

    :return: LaTeX source.
    """
    meta = summary["meta"]
    # Everything the prose quotes, taken from the run rather than written in. The first draft had
    # these as literals, which put one wrong number in the document: the all-token control count was
    # block 11's, quoted against block 25's danger count.
    prose, sweep = summary["prose"], summary["sweep"]
    est, counts = summary["estimators"], summary["estimator_counts"]
    e25 = {name: est[name]["25"] for name in est}
    every = prose["alltoken_counts"]["25"]
    peaks = {l: prose["modal_peak"][l]["25"] for l in LADDERS}
    danger = [e25["final token"][k] for k in PAIRS[:3]]
    order = sorted(sweep, key=lambda n: -sweep[n]["25"]["separation"])
    shown_positions = ["token -1"] + [n for n in order if n != "token -1"][:2] + ["content mean", "max"]
    best_sep = ", ".join(sweep_name.replace("token ", "$") + "$" if sweep_name.startswith("token")
                         else sweep_name
                         for sweep_name in
                         (max(sweep, key=lambda n: sweep[n][str(l)]["separation"]) for l in LAYERS))
    best_dd = ", ".join(sweep_name.replace("token ", "$") + "$" if sweep_name.startswith("token")
                        else sweep_name
                        for sweep_name in
                        (max(sweep, key=lambda n: sweep[n][str(l)]["dd"]) for l in LAYERS))
    reply_agree = prose["reply_ladder_agreement"]["25"]
    greedy = prose["greedy"]
    final, reply = summary["per_layer_final"], summary["per_layer_reply"]
    block = table[table.layer == 25]
    hits = table[table.hit & (table.specificity > 0)].sort_values("specificity", ascending=False)
    hits25 = hits[hits.layer == 25]

    out = [r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=18mm]{geometry}
\usepackage{fontspec}\usepackage{booktabs}\usepackage{longtable}\usepackage{graphicx}
\usepackage{float}\usepackage{amsmath}\usepackage{amssymb}\usepackage{siunitx}\usepackage[hidelinks]{hyperref}
\setmainfont{Latin Modern Roman}\setmonofont[Scale=0.80]{Menlo}
\setlength{\parskip}{0.45em}\setlength{\parindent}{0pt}
\renewcommand{\arraystretch}{1.1}
\title{\vspace{-14mm}\textbf{Dose response of 1036 behavioural concept vectors in
\texttt{Qwen/Qwen3-8B}}\vspace{-3mm}}
\author{}\date{}
\begin{document}\maketitle\vspace{-12mm}

\section{Problem}

A concept vector $v\in\mathbb{R}^{4096}$, $\|v\|=1$, is read off the residual stream as
$\cos(v,\hat h)$ where $\hat h=h/\|h\|$. We ask whether that value tracks a scalar quantity stated in
the prompt, following the Tylenol experiment of Anthropic's emotion-vector report: hold the prompt
fixed and vary one number.

Four ladders. Three vary a quantity whose danger rises with it; one varies a quantity of the same
magnitude whose danger does not. The last is the control the original lacks, and it is what
separates ``responds to danger'' from ``responds to a large number''.

\begin{center}\small\begin{tabular}{@{}llp{86mm}@{}}\toprule
ladder & range & template \\ \midrule
"""]
    for name, (kind, span, text) in LADDER_SPEC.items():
        out.append(f"\\texttt{{{name}}} ({kind}) & {span} & \\footnotesize\\texttt{{{escape(text)}}} \\\\\n")
    out.append(r"""\bottomrule\end{tabular}\end{center}

The first three ladders vary a single digit, so all rungs tokenise to equal length. \texttt{ibuprofen}
does not (29--31 tokens); its rungs share the template suffix, so positions counted from the end
still correspond. All statistics use the \texttt{chat} rendering with reasoning disabled, in which
the prompt ends
\texttt{<|im\_start|>assistant\textbackslash n<think>\textbackslash n\textbackslash n</think>\textbackslash n\textbackslash n}.

\paragraph{Null.} 512 random unit directions are carried as extra columns of the same projection, so
they see the same activations and the same arithmetic. For a statistic $S$, $z=S/\sigma_{\text{rand}}$
with $\sigma_{\text{rand}}$ the standard deviation of $S$ over those directions at the same block.
Every quoted threshold is $|z|\geq4$.

\section{Choice of estimator}

Let $d(t)=\cos(v,\hat h_{\text{top}}(t))-\cos(v,\hat h_{\text{bot}}(t))$. Three estimators were
compared.

\begin{enumerate}\setlength\itemsep{0.1em}
\item $\max_t|d(t)|$ over tokens whose id is identical in every rung.
\item $d$ evaluated on the mean over the user's sentence, template tokens excluded.
\item $d$ at the final prompt token.
\end{enumerate}

Restricting (1) to identical tokens is necessary. Taken over all positions the argmax lands on the
varying digit for essentially every concept --- there the rungs hold different tokens and any
direction separates their embeddings --- and the control ladder then scores comparably to the danger
ladders (""" + f"{every['steps']} against {every['tylenol']}" + r""" concepts at block 25).

An estimator is admissible only if it assigns similar scores to the same concepts on two prompts
that differ only in wording. Agreement $r$ over the 1036 concepts at block 25:

\begin{center}\small\begin{tabular}{@{}lrrrr@{}}\toprule
estimator & tyl--syr & tyl--ibu & syr--ibu & \textit{tyl--steps} \\ \midrule
""")
    for name in est:
        row = est[name]["25"]
        out.append(f"{escape(name)} & " + " & ".join(f"{row[k]:+.3f}" for k in PAIRS[:3])
                   + f" & \\textit{{{row['tylenol-steps']:+.3f}}} \\\\\n")
    out.append(r"""\bottomrule\end{tabular}\end{center}

The maximum agrees between the two closest ladders at $r=""" +
f"{e25['max over identical tokens']['tylenol-syrup']:.3f}" + r"""$, against $""" +
f"{e25['max over identical tokens']['tylenol-steps']:.3f}" + r"""$ with the magnitude control. Each
ladder selects a different peak token --- modal peak at $""" +
"$, $".join(f"{peaks[l][0]}$ for \\texttt{{{l}}} ({peaks[l][1]} concepts)" for l in DANGER) +
r""" --- so it is not the same function on different prompts. The content mean agrees with the
control at $""" + f"{e25['content mean']['tylenol-steps']:.3f}" + r"""$ against $""" +
f"{e25['content mean']['tylenol-syrup']:.3f}" + r"""$ for \texttt{syrup}. The final token is the most
reproducible of the three ($""" + f"{min(danger):.3f}$--${max(danger):.3f}" +
r"""$ among danger ladders) at a control agreement of $""" +
f"{e25['final token']['tylenol-steps']:.3f}" + r"""$.

""" + figure(5, r"""Peak position of $\max_t|d(t)|$ over the identical-token region, per block. Each
ladder concentrates on a different token, which is why the maximum is not the same function on
differently worded prompts.""", figures) + figure(2, r"""Estimator comparison at block 25. (a) agreement over the 1036 concepts; red bars
are danger--danger and should be high, the blue bar is danger--control and should be low. (b) number
of concepts reaching $|z|\geq4$. The maximum yields the largest counts and the least
reproducibility.""", figures) + r"""
\section{Position sweep}

Those three estimators are a subset. Every position was swept, scored by two criteria: $dd$, the mean
agreement among the three danger ladders, and $dc$, the agreement with the control. At block 25:

\begin{center}\small\begin{tabular}{@{}lrrr@{}}\toprule
position & $dd$ & $dc$ & $dd-dc$ \\ \midrule
""")
    for name in shown_positions:
        row = sweep[name]["25"]
        label = name.replace("token ", "token $") + "$" if name.startswith("token") else name
        out.append(f"{label}{' (used below)' if name == 'token -1' else ''} & "
                   f"${row['dd']:+.3f}$ & ${row['dc']:+.3f}$ & ${row['separation']:+.3f}$ \\\\\n")
    out.append(r"""\bottomrule\end{tabular}\end{center}

The criteria disagree, and the winner is unstable across blocks: by $dd-dc$ it is """ + best_sep +
r""" at blocks 11, 14, 18, 22, 25, and by $dd$ it is """ + best_dd + r""". Selecting on $dd-dc$ over
31 candidates would fit noise. The two also behave differently across position: $dd$ varies smoothly,
high from $-1$ to about $-20$ and collapsing beyond $-25$, whereas $dd-dc$ oscillates between
adjacent tokens with no structure.

\textbf{All results below use the final prompt token}, on the single criterion that a statistic must
first be the same function on differently worded prompts. The choice is stated rather than optimised.
No claim in \S6 or \S7 rests on it: the reply-side statistics and the prompt--reply coupling use the
generated tokens, not a prompt position.

\section{Behaviour}

The model's replies change with dose. Greedy continuations, verbatim:

\begin{center}\footnotesize\begin{tabular}{@{}p{18mm}p{148mm}@{}}\toprule
""")
    for dose in ("1000", "5000", "9000"):
        entry = greedy.get(dose)
        if entry is None:
            continue
        body = " ".join(entry["text"].split())
        clipped = body[:excerpt].rsplit(" ", 1)[0] + r"\ldots{}" if len(body) > excerpt else body
        out.append(f"\\SI{{{dose}}}{{\\milli\\gram}} & \\texttt{{{escape(clipped)}}} \\\\[1mm]\n")
    out.append(r"""\bottomrule\end{tabular}\end{center}

The transition occurs at the \SI{4000}{\milli\gram} clinical maximum, which the model states. Reply
length rises with dose (""" + ", ".join(str(greedy[d]["tokens"]) for d in ("1000", "5000", "9000")
                                        if d in greedy) + r""" tokens greedy), so all reply-side
statistics are per-token means; a sum would measure length.

""" + figure(1, r"""(a) reply length against dose, five continuations per rung. (b) concepts reaching
$|z|\geq4$ at the final prompt token, per block and ladder.""", figures) + r"""
\section{Prompt side}

Concepts with $|z|\geq4$ at the final token, of 1036:

\begin{center}\small\begin{tabular}{@{}lrrrr@{}}\toprule
block & tylenol & syrup & ibuprofen & \textit{steps} \\ \midrule
""")
    for layer in LAYERS:
        row = final[str(layer)]
        out.append(f"{layer} & {row['tylenol']} & {row['syrup']} & {row['ibuprofen']} & "
                   f"\\textit{{{row['steps']}}} \\\\\n")
    out.append(r"""\bottomrule\end{tabular}\end{center}

The control column is not small at the late blocks (""" +
              ", ".join(str(final[str(l)]["steps"]) for l in (18, 22, 25)) + r"""), so the count alone
does not establish specificity; the agreement structure in \S2 does.

At block 25 the median $|z|$ over all 1036 concepts is """ + f"{block.f_tylenol.abs().median():.2f}" +
              r""" and the maximum """ + f"{block.f_tylenol.abs().max():.2f}" + r""". """ + (
                  r"""The median clears the threshold, so the response is graded across the whole
ontology rather than confined to a subset."""
                  if block.f_tylenol.abs().median() >= 4 else
                  r"""The median sits below the $|z|=4$ threshold: the typical concept moves no more
than a random direction does, and the responding set is a minority.""") + r"""

""" + figure(7, r"""Dose response at the final token, block 25, at fixed ranks through the sorted
list rather than at the top; bottom row is six random directions. Shared vertical scale.""", figures)
              + figure(6, r"""Concepts (outline) against the 512 random directions (filled), prompt
side (top) and reply side (bottom). Dashed line marks $|z|=4$.""", figures) + r"""
\section{Reply side}

Every chat rung was continued five times (greedy plus four sampled at $T=0.7$, $p=0.8$, $k=20$, 256
tokens). The reply-side value of a concept is its mean cosine over the generated tokens, averaged
over the five continuations. Concepts with $|z|\geq4$:

\begin{center}\small\begin{tabular}{@{}lrrrr@{}}\toprule
block & tylenol & syrup & ibuprofen & \textit{steps} \\ \midrule
""")
    for layer in LAYERS:
        row = reply[str(layer)]
        out.append(f"{layer} & {row['tylenol']} & {row['syrup']} & {row['ibuprofen']} & "
                   f"\\textit{{{row['steps']}}} \\\\\n")
    out.append(r"""\bottomrule\end{tabular}\end{center}

Reply-side magnitudes: median $|z|=""" + f"{block.rz_tylenol.abs().median():.2f}"
              + r"""$, maximum $""" + f"{block.rz_tylenol.abs().max():.2f}" + r"""$ at block 25. The three
danger ladders agree at $r=""" + f"{reply_agree['tylenol-syrup']:.3f}" + r"""$ (\texttt{syrup}) and
$""" + f"{reply_agree['tylenol-ibuprofen']:.3f}" + r"""$ (\texttt{ibuprofen}); the control agrees at
$""" + f"{reply_agree['tylenol-steps']:.3f}" + r"""$. The first principal component over the three
danger ladders explains $""" + f"{100 * prose['pc1_reply']['25']:.1f}" + r"""\%$ of the variance, so
this is one direction of variation read by many concepts, not many independent detectors.

""" + figure(8, r"""Reply-side dose response, block 25, eight strongest concepts. Bands span the five
continuations.""", figures) + r"""
\section{Coupling between prompt and reply}

For each concept, the prompt-side $z$ at the final token and the reply-side $z$ are correlated across
the 1036 concepts:

\begin{center}\small\begin{tabular}{@{}lrrrrr@{}}\toprule
block & 11 & 14 & 18 & 22 & 25 \\ \midrule
$r$ & """ + " & ".join(f"{summary['prompt_reply_agreement_final'][str(l)]:+.3f}" for l in LAYERS)
              + r""" \\ \bottomrule\end{tabular}\end{center}

The coupling is weakest at block 11 ($""" +
f"{summary['prompt_reply_agreement_final']['11']:+.3f}" + r"""$) and strongest at block 25 ($""" +
f"{summary['prompt_reply_agreement_final']['25']:+.3f}" + r"""$). """ + (
                  r"""What the late blocks encode before the first generated token is what the reply
expresses."""
                  if summary["prompt_reply_agreement_final"]["25"] >= 0.6 else
                  r"""The coupling does not become strong at any block, so what these directions read
from the prompt is not what the reply expresses along them.""") + r"""

A per-concept version of Anthropic's \S2.2.2: correlate a concept's prompt-side value at position $P$
across the nine rungs against its reply-side value across the same rungs, and compare positions by
the mean $|r|$ over concepts against the same quantity over the random directions. At block 25 the
excess over the random floor is """ + ", ".join(
        f"${d['25']['concepts_mean_abs_r'] - d['25']['controls_mean_abs_r']:+.3f}$ for {escape(k)}"
        for k, d in list(summary["prediction"].items())) + r""". The random floor is itself high (""" +
f"${min(d['25']['controls_mean_abs_r'] for d in summary['prediction'].values()):.2f}$--"
f"${max(d['25']['controls_mean_abs_r'] for d in summary['prediction'].values()):.2f}$" +
r"""), so the raw correlation is not interpretable alone.


\section{Specificity}

""" + f"{len(hits25)}" + r""" concepts at block 25 are consistent in sign across all three danger
ladders, reach $|z|\geq4$ on the weakest of them, and exceed their own leakage onto the control. The
""" + f"{min(top, len(hits25))}" + r""" largest by that margin:

\begin{center}\small\setlength{\tabcolsep}{8pt}
\begin{longtable}{@{}p{74mm}rrrrr@{}}\toprule
concept & $z_{\text{tyl}}$ & $z_{\text{syr}}$ & $z_{\text{ibu}}$ & \textit{$z_{\text{steps}}$} &
$z_{\text{reply}}$ \\ \midrule \endhead
""")
    for _, row in hits25.head(top).iterrows():
        out.append(f"{escape(row.concept)} & {row.f_tylenol:.1f} & {row.f_syrup:.1f} & "
                   f"{row.f_ibuprofen:.1f} & \\textit{{{row.f_steps:.1f}}} & {row.rz_tylenol:.1f} \\\\\n")
    out.append(r"""\bottomrule\end{longtable}\end{center}

Control leakage in this set runs from $""" + f"{hits25.head(top).f_steps.min():.1f}" + r"""$ to
$""" + f"{hits25.head(top).f_steps.max():.1f}" + r"""$, against danger values of $""" +
f"{hits25.head(top).strength.min():.1f}" + r"""$ to $""" + f"{hits25.head(top).strength.max():.1f}" + r"""$.

\section{Reproduction}

\begin{center}\small\begin{tabular}{@{}ll@{}}\toprule
""")
    for label, value in (
        ("model", meta["model"] + ", bf16 weights, float32 projections"),
        ("blocks", ", ".join(str(n) for n in meta["layers"]) + " of 36"),
        ("directions", f"{meta['concepts']} concepts + {meta['controls']} random, seed {meta['seed']}"),
        ("rungs", f"{summary['n_rungs']} (4 ladders x 2 renderings)"),
        ("continuations", f"{summary['n_replies']} at {meta['reply_tokens']} tokens, "
                          f"greedy + {meta['samples']} sampled"),
        ("hardware", "2 x A100, DataSphere g2.1, 22 and 20 minutes"),
        ("code", "dose.py, doseall.py, dosefigs.py, dosepaper.py; yds/dose.\\{sh,yaml\\}"),
        ("artifacts", "school18 \\textasciitilde/dose2/: dose-readout.npz (418 MB, 2 shards), "
                      "dose-derived.npz, dose-stats.parquet, dose-summary.json"),
        ("ontology", "148 classes / 1036 pairs, AntonKorznikov/feature\\_stories"),
    ):
        out.append(f"\\texttt{{{label}}} & {value} \\\\\n")
    out.append(r"""\bottomrule\end{tabular}\end{center}
""" + r"\end{document}" + "\n")
    return "".join(out)


def main(args: Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = json.loads(args.summary.read_text())
    table = pd.read_parquet(args.table)
    tex = args.out.with_suffix(".tex")
    tex.write_text(build(summary, table, args.figures.name, args.top, args.excerpt))
    log.info(f"wrote {tex}")

    for _ in range(2):
        done = subprocess.run(["lualatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
                              cwd=tex.parent, capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit("lualatex failed:\n" + "\n".join(done.stdout.splitlines()[-40:]))
    log.info(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path(".bak/dose2/dose-summary.json"))
    parser.add_argument("--table", type=Path, default=Path(".bak/dose2/dose-stats.parquet"))
    parser.add_argument("--figures", type=Path, default=Path(".bak/dose2/dose-figures.pdf"))
    parser.add_argument("--out", type=Path, default=Path(".bak/dose2/dose.pdf"))
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--excerpt", type=int, default=300, help="characters of each reply quoted")
    main(parser.parse_args())
