"""
python/check_tables.py

Cross-checks every table in results/final_tables.md against the raw run logs it was built from:
  - results/task8_final_output.txt            (main study, configs/models.toml)
  - results/lownoise/task8_output.txt         (low-noise sensitivity, configs/models_lownoise.toml)

Unlike the earlier version (which silently skipped any section its regexes did not find),
this check FAILS if a table present in a log is missing from final_tables.md, if row counts
differ, or if any number differs by more than 1e-3. It also checks the permutation null-check
numbers quoted in the markdown.

Usage:  python python/check_tables.py
Writes: results/check_tables_report.txt   (exit code 1 on any problem)
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from table_utils import parse_text_tables, parse_markdown_tables, numbers

SOURCES = [
    ("", "results/task8_final_output.txt"),
    ("Low-Noise Sensitivity: ", "results/lownoise/task8_output.txt"),
]
MD_PATH = "results/final_tables.md"
REPORT_PATH = "results/check_tables_report.txt"


def main():
    with open(MD_PATH, encoding="utf-8") as f:
        md_text = f.read()
    md_tables = parse_markdown_tables(md_text)

    report = ["=" * 80, "CROSS-CHECK VERIFICATION REPORT: final_tables.md vs raw run logs", "=" * 80]
    problems = []
    total_values = 0

    for prefix, path in SOURCES:
        if not os.path.exists(path):
            problems.append(f"missing raw log: {path}")
            continue
        with open(path, encoding="utf-8") as f:
            raw_text = f.read()
        raw_tables = parse_text_tables(raw_text)
        report.append(f"\nSource: {path}  ({len(raw_tables)} tables)")
        for title, raw in raw_tables.items():
            heading = prefix + title
            if heading not in md_tables:
                # Low-noise log: only the tables that final_tables.md chooses to show are required
                if prefix:
                    continue
                problems.append(f"table missing from markdown: '{heading}'")
                report.append(f"  [MISSING] {heading}")
                continue
            md = md_tables[heading]
            n_checked, n_bad = 0, 0
            if len(md["rows"]) != len(raw["rows"]):
                problems.append(f"'{heading}': {len(raw['rows'])} raw rows vs {len(md['rows'])} markdown rows")
            for r_row, m_row in zip(raw["rows"], md["rows"]):
                if r_row[0] != m_row[0].replace("**", ""):
                    problems.append(f"'{heading}': row label '{r_row[0]}' vs '{m_row[0]}'")
                for c_idx, (r_cell, m_cell) in enumerate(zip(r_row, m_row)):
                    rn, mn = numbers(r_cell), numbers(m_cell)
                    if not rn and not mn:
                        continue
                    n_checked += len(rn)
                    if len(rn) != len(mn) or any(abs(a - b) > 1e-3 for a, b in zip(rn, mn)):
                        n_bad += 1
                        problems.append(f"'{heading}' row '{r_row[0]}' col {c_idx}: raw '{r_cell}' vs md '{m_cell}'")
            total_values += n_checked
            report.append(f"  [{'OK ' if n_bad == 0 else 'BAD'}] {heading}: {len(raw['rows'])} rows, {n_checked} values, {n_bad} mismatched cells")

        # Permutation null-check lines quoted in the markdown (main study only)
        if not prefix:
            for label in ["RF Primary Permutation Null Check", "AE All Permutation Null Check"]:
                m_raw = re.search(re.escape(label) + r":\s*(.*)", raw_text)
                if not m_raw:
                    problems.append(f"null-check line '{label}' missing from raw log")
                    continue
                if m_raw.group(1).strip() not in md_text:
                    problems.append(f"null-check line '{label}' not reproduced verbatim in markdown")
                else:
                    total_values += len(numbers(m_raw.group(1)))
                    report.append(f"  [OK ] {label}: quoted verbatim")

    report.append(f"\nTotal numeric values compared: {total_values}")
    report.append(f"Problems found: {len(problems)}")
    for p in problems:
        report.append(f"  - {p}")
    report.append("\nVERIFICATION RESULT: " + ("ALL TABLES MATCH (0 PROBLEMS)." if not problems else "FAILED."))
    text = "\n".join(report)
    print(text)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    sys.exit(0 if not problems else 1)


if __name__ == "__main__":
    main()
