"""Build paper/arxiv/arxiv-source.tar.gz for arXiv upload.

Run from paper/ after a full pdflatex + bibtex build (main.bbl must exist):
    python3 arxiv/make_tarball.py

The tarball holds main.tex, sections/*.tex, main.bbl and the figures that
\\includegraphics actually uses. All % comments are stripped, since arXiv
shows the source and the comments hold internal source pointers. A comment
that follows code on the same line is reduced to a bare %, so line-end
spacing is unchanged. Verbatim-like environments are copied as is.
"""

import io
import re
import sys
import tarfile
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent
OUT = PAPER / "arxiv" / "arxiv-source.tar.gz"
VERBATIM = ("lstlisting", "verbatim", "Verbatim")


def strip_comments(text):
    out, in_verb = [], None
    for line in text.splitlines():
        if in_verb:
            out.append(line)
            if f"\\end{{{in_verb}}}" in line:
                in_verb = None
            continue
        m = re.search(r"\\begin\{(" + "|".join(VERBATIM) + r")\}", line)
        if m and f"\\end{{{m.group(1)}}}" not in line:
            in_verb = m.group(1)
        # first % not escaped by an odd number of backslashes
        cut = None
        for i in re.finditer(r"%", line):
            j, n = i.start() - 1, 0
            while j >= 0 and line[j] == "\\":
                n, j = n + 1, j - 1
            if n % 2 == 0:
                cut = i.start()
                break
        if cut is None:
            out.append(line)
        elif line[:cut].strip() == "":
            continue  # whole-line comment: drop the line
        else:
            out.append(line[:cut].rstrip() + "%" if line[cut - 1] != " " else line[:cut].rstrip() + " %")
    return "\n".join(out) + "\n"


def used_figures(tex):
    names = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    files = []
    for n in names:
        cands = [PAPER / "figures" / n, PAPER / "figures" / (n + ".pdf"), PAPER / n]
        hit = next((c for c in cands if c.is_file()), None)
        if hit is None:
            sys.exit(f"figure not found: {n}")
        files.append(hit)
    return files


def main():
    if not (PAPER / "main.bbl").is_file():
        sys.exit("main.bbl missing: run pdflatex, bibtex, pdflatex first")
    sources = [PAPER / "main.tex"] + sorted((PAPER / "sections").glob("*.tex"))
    texts = {p: strip_comments(p.read_text()) for p in sources}
    all_tex = "".join(texts.values())
    for bad in ("/Users", "/tmp", "TODO", "src:"):
        if bad in all_tex:
            sys.exit(f"stripped source still contains {bad!r}")
    files = [(p.relative_to(PAPER).as_posix(), texts[p].encode()) for p in sources]
    files.append(("main.bbl", (PAPER / "main.bbl").read_bytes()))
    for f in used_figures(all_tex):
        files.append((f.relative_to(PAPER).as_posix(), f.read_bytes()))
    with tarfile.open(OUT, "w:gz") as tar:
        for name, data in files:
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            tar.addfile(info, io.BytesIO(data))
    for name, data in files:
        print(f"{len(data):>8}  {name}")
    print(f"wrote {OUT.relative_to(PAPER)}")


if __name__ == "__main__":
    main()
