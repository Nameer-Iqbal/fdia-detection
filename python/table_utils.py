"""
python/table_utils.py

Parsing helpers shared by python/make_final_tables.py and python/check_tables.py.

run_task8.py prints every result table in the same plain-text layout:

    --- <Table Title> ---
    (optional note line in parentheses)
    Header 1 | Header 2 | ...
    ---------------------------
    row cell | row cell | ...
    ...
    <blank line or a line without '|'>

parse_text_tables() turns such a log into {title: {"header": [...], "rows": [[...], ...]}}.
parse_markdown_tables() reads the same tables back out of results/final_tables.md, where each
table sits under a heading of the form '### <prefix><Table Title>'.
"""
import re

TITLE_RE = re.compile(r"^--- (.+) ---\s*$")
NUM_RE = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")


def _split_row(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse_text_tables(text):
    """Returns an ordered dict {title: {"header": [...], "rows": [[...]], "note": str}} from a run log."""
    lines = text.replace("﻿", "").splitlines()
    tables = {}
    i = 0
    while i < len(lines):
        m = TITLE_RE.match(lines[i])
        if not m:
            i += 1
            continue
        title = m.group(1).strip()
        j = i + 1
        note = ""
        if j < len(lines) and lines[j].startswith("(") and "|" not in lines[j]:
            note = lines[j].strip()
            j += 1
        # A table needs a header row with '|' followed by a dashed rule
        if j + 1 < len(lines) and "|" in lines[j] and set(lines[j + 1].strip()) == {"-"}:
            header = _split_row(lines[j])
            rows = []
            k = j + 2
            while k < len(lines) and "|" in lines[k] and lines[k].strip():
                rows.append(_split_row(lines[k]))
                k += 1
            tables[title] = {"header": header, "rows": rows, "note": note}
            i = k
        else:
            i = j
    return tables


def parse_markdown_tables(md_text):
    """Returns {heading_text: {"header": [...], "rows": [[...]]}} for every '### heading' followed by a table."""
    lines = md_text.splitlines()
    tables = {}
    current = None
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("### "):
            current = line[4:].strip()
            i += 1
            continue
        if current and line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|\s*:?-{3,}", lines[i + 1]):
            header = _split_row(line)
            rows = []
            k = i + 2
            while k < len(lines) and lines[k].startswith("|"):
                rows.append(_split_row(lines[k]))
                k += 1
            tables[current] = {"header": header, "rows": rows}
            current = None
            i = k
            continue
        i += 1
    return tables


def numbers(cell):
    """All numbers appearing in a cell (markdown emphasis and % signs ignored)."""
    return [float(x) for x in NUM_RE.findall(cell.replace("**", "").replace("`", "").replace(",", ""))]


def cell_by_header(table, row_label, header_name):
    """Looks up one cell by its row label (first column) and its column header."""
    col = table["header"].index(header_name)
    for r in table["rows"]:
        if r[0] == row_label:
            return r[col]
    raise KeyError(f"row '{row_label}' not found")


def first_number(cell):
    return numbers(cell)[0]
