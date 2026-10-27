"""
python/check_tables.py

Cross-checks results/final_tables.md against results/task8_final_output.txt.
Compares:
  1. Operating thresholds (Val Thresh 5% and 1%)
  2. Realized test false alarm rates (Realized FA @ 5% and 1%)
  3. Table 1 overall AUC, exact AC AUC, and detection at 5% for every method
  4. Table 1 per-attack-type detection rates for every method
  5. Table 2 overall AUC, exact AC AUC, and detection at 5% for every method
  6. Table 2 per-attack-type detection rates (including held-out exact AC)
  7. Detection vs shift_sigmas bins
  8. Feature-group ablation rows
  9. Limited-meter detection vs shift ratio rows

Saves report to results/check_tables_report.txt.
"""

import sys
import re

def normalize_val(val_str):
    """Normalize string representation of numbers for robust comparison."""
    s = val_str.strip().replace("`", "").replace("**", "").replace("%", "").strip()
    # Check if float
    try:
        f = float(s)
        return f"{f:.4f}"
    except ValueError:
        pass
    # Check if confidence interval [low, high]
    ci_match = re.search(r"(\d+\.?\d*)\s*\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]", s)
    if ci_match:
        val = float(ci_match.group(1))
        low = float(ci_match.group(2))
        high = float(ci_match.group(3))
        return f"{val:.4f} [{low:.4f}, {high:.4f}]"
    return s.lower()

def extract_numbers(s):
    """Extract all float numbers from a string."""
    return [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", s)]

def main():
    with open("results/task8_final_output.txt", "r", encoding="utf-8") as f:
        raw_output = f.read()

    with open("results/final_tables.md", "r", encoding="utf-8") as f:
        md_tables = f.read()

    comparisons = []
    mismatches = []

    # Helper to record a comparison
    def record_check(category, item_name, field, val_raw, val_md):
        # Compare extracted float numbers if present
        nums_raw = extract_numbers(str(val_raw))
        nums_md = extract_numbers(str(val_md))

        match = False
        if nums_raw and nums_md:
            if len(nums_raw) == len(nums_md):
                # Check float closeness
                diffs = [abs(a - b) for a, b in zip(nums_raw, nums_md)]
                if all(d < 1e-3 for d in diffs):
                    match = True
        elif str(val_raw).strip() == str(val_md).strip():
            match = True

        comparisons.append({
            "category": category,
            "item": item_name,
            "field": field,
            "raw": val_raw,
            "md": val_md,
            "match": match
        })
        if not match:
            mismatches.append({
                "category": category,
                "item": item_name,
                "field": field,
                "raw": val_raw,
                "md": val_md
            })

    # -------------------------------------------------------------------------
    # 1. Compare Operating Thresholds & Realized FA Rates
    # -------------------------------------------------------------------------
    # From raw output:
    th_raw_match = re.search(
        r"--- Validation Normal Thresholds and Realized Normal Test False Alarm Rates ---\n.*?\n-+\n(.*?)\n\n--- Chi-Square",
        raw_output, re.DOTALL
    )
    # From md:
    th_md_match = re.search(
        r"## 1. Operating Thresholds Derived from Normal Validation Rows.*?\n\|---\|.*?\|\n(.*?)\n\n---",
        md_tables, re.DOTALL
    )

    if th_raw_match and th_md_match:
        raw_rows = [r.strip() for r in th_raw_match.group(1).strip().split("\n") if r.strip()]
        md_rows = [r.strip() for r in th_md_match.group(1).strip().split("\n") if r.strip()]

        for r_line, m_line in zip(raw_rows, md_rows):
            r_parts = [p.strip() for p in r_line.split("|")]
            m_parts = [p.strip().replace("`", "").replace("**", "") for p in m_line.strip("|").split("|")]

            m_name = r_parts[0]
            record_check("Thresholds", m_name, "Val Thresh 5%", r_parts[1], m_parts[1])
            record_check("Thresholds", m_name, "Val Thresh 1%", r_parts[2], m_parts[2])
            record_check("Thresholds", m_name, "Realized FA 5%", r_parts[3], m_parts[3])
            record_check("Thresholds", m_name, "Realized FA 1%", r_parts[4], m_parts[4])

    # -------------------------------------------------------------------------
    # 2. Compare Table 1: Standard Split Summary
    # -------------------------------------------------------------------------
    t1_raw_match = re.search(
        r"--- TABLE 1: STANDARD TEST SPLIT COMPARISON \(N=2,000, VALIDATION THRESHOLDS\) ---\n.*?\n-+\n(.*?)\n\n--- Table 1 Per Attack Type",
        raw_output, re.DOTALL
    )
    t1_md_match = re.search(
        r"## 2. Table 1: Standard Test Split Benchmark.*?\n\|---\|.*?\|\n(.*?)\n\n### Table 1 Per-Attack-Type",
        md_tables, re.DOTALL
    )

    if t1_raw_match and t1_md_match:
        raw_rows = [r.strip() for r in t1_raw_match.group(1).strip().split("\n") if r.strip()]
        md_rows = [r.strip() for r in t1_md_match.group(1).strip().split("\n") if r.strip()]

        for r_line, m_line in zip(raw_rows, md_rows):
            r_parts = [p.strip() for p in r_line.split("|")]
            m_parts = [p.strip().replace("`", "").replace("**", "") for p in m_line.strip("|").split("|")]

            m_name = r_parts[0]
            record_check("Table 1 Summary", m_name, "Realized Normal FA", r_parts[1], m_parts[1])
            record_check("Table 1 Summary", m_name, "Overall AUC", r_parts[2], m_parts[2])
            record_check("Table 1 Summary", m_name, "Exact AC AUC", r_parts[3], m_parts[3])
            record_check("Table 1 Summary", m_name, "Det @ 5% FA", r_parts[4], m_parts[4])
            record_check("Table 1 Summary", m_name, "Det @ 1% FA", r_parts[5], m_parts[5])
            record_check("Table 1 Summary", m_name, "Exact AC Det @ 5% FA", r_parts[6], m_parts[6])
            record_check("Table 1 Summary", m_name, "Effective Det @ 5% FA", r_parts[7], m_parts[7])

    # -------------------------------------------------------------------------
    # 3. Compare Table 1 Per-Attack-Type
    # -------------------------------------------------------------------------
    t1_pt_raw_match = re.search(
        r"--- Table 1 Per Attack Type Detection @ 5% FA ---\n.*?\n-+\n(.*?)\n\n====",
        raw_output, re.DOTALL
    )
    t1_pt_md_match = re.search(
        r"### Table 1 Per-Attack-Type Detection Rate @ 5% Validation Normal FA:\n\n.*?\n\|---\|.*?\|\n(.*?)\n\n---",
        md_tables, re.DOTALL
    )

    if t1_pt_raw_match and t1_pt_md_match:
        raw_rows = [r.strip() for r in t1_pt_raw_match.group(1).strip().split("\n") if r.strip()]
        md_rows = [r.strip() for r in t1_pt_md_match.group(1).strip().split("\n") if r.strip()]

        for r_line, m_line in zip(raw_rows, md_rows):
            r_parts = [p.strip() for p in r_line.split("|")]
            m_parts = [p.strip().replace("`", "").replace("**", "") for p in m_line.strip("|").split("|")]

            m_name = r_parts[0]
            record_check("Table 1 Per-Type", m_name, "exact_single", r_parts[1], m_parts[1])
            record_check("Table 1 Per-Type", m_name, "exact_multi", r_parts[2], m_parts[2])
            record_check("Table 1 Per-Type", m_name, "limited", r_parts[3], m_parts[3])
            record_check("Table 1 Per-Type", m_name, "lin_single", r_parts[4], m_parts[4])
            record_check("Table 1 Per-Type", m_name, "lin_multi", r_parts[5], m_parts[5])
            record_check("Table 1 Per-Type", m_name, "random", r_parts[6], m_parts[6])

    # -------------------------------------------------------------------------
    # 4. Compare Table 2: Unseen Split Summary
    # -------------------------------------------------------------------------
    t2_raw_match = re.search(
        r"--- TABLE 2: UNSEEN-ATTACK SPLIT COMPARISON \(N=4,026, VALIDATION THRESHOLDS\) ---\n.*?\n-+\n(.*?)\n\n--- Table 2 Held-Out",
        raw_output, re.DOTALL
    )
    t2_md_match = re.search(
        r"## 3. Table 2: Unseen-Attack Scheme Benchmark.*?\|---\|.*?\|\n(.*?)\n\n### Table 2 Detection",
        md_tables, re.DOTALL
    )

    if t2_raw_match and t2_md_match:
        raw_rows = [r.strip() for r in t2_raw_match.group(1).strip().split("\n") if r.strip()]
        md_rows = [r.strip() for r in t2_md_match.group(1).strip().split("\n") if r.strip()]

        for r_line, m_line in zip(raw_rows, md_rows):
            r_parts = [p.strip() for p in r_line.split("|")]
            m_parts = [p.strip().replace("`", "").replace("**", "") for p in m_line.strip("|").split("|")]

            m_name = r_parts[0]
            record_check("Table 2 Summary", m_name, "Realized Normal FA", r_parts[1], m_parts[1])
            record_check("Table 2 Summary", m_name, "Overall AUC", r_parts[2], m_parts[2])
            record_check("Table 2 Summary", m_name, "Exact AC AUC", r_parts[3], m_parts[3])
            record_check("Table 2 Summary", m_name, "Det @ 5% FA", r_parts[4], m_parts[4])
            record_check("Table 2 Summary", m_name, "Det @ 1% FA", r_parts[5], m_parts[5])
            record_check("Table 2 Summary", m_name, "Exact AC Det @ 5% FA", r_parts[6], m_parts[6])
            record_check("Table 2 Summary", m_name, "Effective Det @ 5% FA", r_parts[7], m_parts[7])

    # -------------------------------------------------------------------------
    # 5. Compare Table 2 Per-Attack-Type (Held-Out & Seen)
    # -------------------------------------------------------------------------
    t2_pt_raw_match = re.search(
        r"--- Table 2 Held-Out and Seen Attack Detection @ 5% FA ---\n.*?\n-+\n(.*?)\n\n====",
        raw_output, re.DOTALL
    )
    t2_pt_md_match = re.search(
        r"### Table 2 Detection on Completely Held-Out and Seen Attack Types @ 5% FA:\n\n.*?\n\|---\|.*?\|\n(.*?)\n\n---",
        md_tables, re.DOTALL
    )

    if t2_pt_raw_match and t2_pt_md_match:
        raw_rows = [r.strip() for r in t2_pt_raw_match.group(1).strip().split("\n") if r.strip()]
        md_rows = [r.strip() for r in t2_pt_md_match.group(1).strip().split("\n") if r.strip()]

        for r_line, m_line in zip(raw_rows, md_rows):
            r_parts = [p.strip() for p in r_line.split("|")]
            m_parts = [p.strip().replace("`", "").replace("**", "") for p in m_line.strip("|").split("|")]

            m_name = r_parts[0]
            record_check("Table 2 Per-Type", m_name, "HELD exact_sin", r_parts[1], m_parts[1])
            record_check("Table 2 Per-Type", m_name, "HELD exact_mul", r_parts[2], m_parts[2])
            record_check("Table 2 Per-Type", m_name, "HELD limited", r_parts[3], m_parts[3])
            record_check("Table 2 Per-Type", m_name, "lin_single", r_parts[4], m_parts[4])
            record_check("Table 2 Per-Type", m_name, "lin_multi", r_parts[5], m_parts[5])
            record_check("Table 2 Per-Type", m_name, "random", r_parts[6], m_parts[6])

    # -------------------------------------------------------------------------
    # 6. Compare Detection vs Shift Sigmas Bins
    # -------------------------------------------------------------------------
    sig_raw_match = re.search(
        r"--- \(1\) Detection vs Shift Sigmas Bins \(@ 5% Validation-Derived FA\) ---\n.*?\n-+\n(.*?)\n\n--- \(2\)",
        raw_output, re.DOTALL
    )
    sig_md_match = re.search(
        r"## 4. Detection vs Attack Size Bins \(`shift_sigmas`\) at 5% Validation Normal FA\n\n.*?\n\|---\|.*?\|\n(.*?)\n\n---",
        md_tables, re.DOTALL
    )

    if sig_raw_match and sig_md_match:
        raw_rows = [r.strip() for r in sig_raw_match.group(1).strip().split("\n") if r.strip()]
        md_rows = [r.strip() for r in sig_md_match.group(1).strip().split("\n") if r.strip()]

        for r_line, m_line in zip(raw_rows, md_rows):
            r_parts = [p.strip() for p in r_line.split("|")]
            m_parts = [p.strip().replace("`", "").replace("**", "") for p in m_line.strip("|").split("|")]

            bin_name = r_parts[0]
            record_check("Sigma Bins", bin_name, "Count", r_parts[1], m_parts[1])
            record_check("Sigma Bins", bin_name, "RF (Primary)", r_parts[2], m_parts[2])
            record_check("Sigma Bins", bin_name, "Autoencoder", r_parts[3], m_parts[3])
            record_check("Sigma Bins", bin_name, "Mahalanobis", r_parts[4], m_parts[4])
            record_check("Sigma Bins", bin_name, "Per-Hour Max-Z", r_parts[5], m_parts[5])

    # -------------------------------------------------------------------------
    # 7. Compare Feature-Group Ablation Rows
    # -------------------------------------------------------------------------
    abl_raw_match = re.search(
        r"--- \(2\) Feature Group Ablations \(Identical Tuned Hyperparams: 300 trees, depth 12, min leaf 1\) ---\n.*?\n-+\n(.*?)\n\n--- \(3\)",
        raw_output, re.DOTALL
    )
    abl_md_match = re.search(
        r"## 5. Feature Group Ablation \(300 Trees, Depth 12, Min Samples Leaf 1\)\n\n.*?\n\|---\|.*?\|\n(.*?)\n\n---",
        md_tables, re.DOTALL
    )

    if abl_raw_match and abl_md_match:
        raw_rows = [r.strip() for r in abl_raw_match.group(1).strip().split("\n") if r.strip()]
        md_rows = [r.strip() for r in abl_md_match.group(1).strip().split("\n") if r.strip()]

        for r_line, m_line in zip(raw_rows, md_rows):
            r_parts = [p.strip() for p in r_line.split("|")]
            m_parts = [p.strip().replace("`", "").replace("**", "") for p in m_line.strip("|").split("|")]

            abl_setting = r_parts[0]
            record_check("Ablation", abl_setting, "Feats", r_parts[1], m_parts[1])
            record_check("Ablation", abl_setting, "Val AUC", r_parts[2], m_parts[2])
            record_check("Ablation", abl_setting, "Test AUC", r_parts[3], m_parts[3])
            record_check("Ablation", abl_setting, "Norm FA", r_parts[4], m_parts[4])
            record_check("Ablation", abl_setting, "Overall Det@5%", r_parts[5], m_parts[5])
            record_check("Ablation", abl_setting, "Exact AC Det@5%", r_parts[6], m_parts[6])

    # -------------------------------------------------------------------------
    # 8. Compare Limited-Meter Detection vs Shift Ratio
    # -------------------------------------------------------------------------
    lm_raw_match = re.search(
        r"--- \(3\) Limited-Meter: Detection vs Shift Ratio \(@ 5% FA\) ---\n.*?\n-+\n(.*?)\n\n====",
        raw_output, re.DOTALL
    )
    lm_md_match = re.search(
        r"## 6. Limited-Meter Attacks: Detection vs Shift Ratio \(@ 5% Validation Normal FA\)\n\n.*?\n\|---\|.*?\|\n(.*?)\n\n---",
        md_tables, re.DOTALL
    )

    if lm_raw_match and lm_md_match:
        raw_rows = [r.strip() for r in lm_raw_match.group(1).strip().split("\n") if r.strip()]
        md_rows = [r.strip() for r in lm_md_match.group(1).strip().split("\n") if r.strip()]

        for r_line, m_line in zip(raw_rows, md_rows):
            r_parts = [p.strip() for p in r_line.split("|")]
            m_parts = [p.strip().replace("`", "").replace("**", "") for p in m_line.strip("|").split("|")]

            ratio_bin = r_parts[0]
            record_check("Limited-Meter Ratio", ratio_bin, "Count", r_parts[1], m_parts[1])
            record_check("Limited-Meter Ratio", ratio_bin, "RF Primary", r_parts[2], m_parts[2])
            record_check("Limited-Meter Ratio", ratio_bin, "Autoencoder", r_parts[3], m_parts[3])
            record_check("Limited-Meter Ratio", ratio_bin, "Mahalanobis", r_parts[4], m_parts[4])
            record_check("Limited-Meter Ratio", ratio_bin, "Calibrated LNR", r_parts[5], m_parts[5])
            record_check("Limited-Meter Ratio", ratio_bin, "Chi-Square", r_parts[6], m_parts[6])

    # -------------------------------------------------------------------------
    # Generate Report
    # -------------------------------------------------------------------------
    report_lines = []
    report_lines.append("================================================================================")
    report_lines.append("CROSS-CHECK VERIFICATION REPORT: final_tables.md vs task8_final_output.txt")
    report_lines.append("================================================================================")
    report_lines.append(f"Total values compared: {len(comparisons)}")
    report_lines.append(f"Total mismatches:      {len(mismatches)}\n")

    # Group by category
    cats = {}
    for c in comparisons:
        cats.setdefault(c["category"], []).append(c)

    for cat_name, cat_items in cats.items():
        cat_matches = sum(1 for x in cat_items if x["match"])
        cat_mismatches = sum(1 for x in cat_items if not x["match"])
        report_lines.append(f"[{cat_name}] {len(cat_items)} checked | {cat_matches} matched | {cat_mismatches} mismatched")

    if mismatches:
        report_lines.append("\n" + "!" * 80)
        report_lines.append("MISMATCH DETAILS:")
        report_lines.append("!" * 80)
        for mm in mismatches:
            report_lines.append(f"- Category: {mm['category']}, Item: {mm['item']}, Field: {mm['field']}")
            report_lines.append(f"    Raw Output: {mm['raw']}")
            report_lines.append(f"    Markdown:   {mm['md']}")
    else:
        report_lines.append("\nVERIFICATION RESULT: ALL VALUES MATCH PERFECTLY (0 MISMATCHES).")

    report_text = "\n".join(report_lines) + "\n"
    with open("results/check_tables_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    if mismatches:
        print(f"FAILED: Found {len(mismatches)} mismatches.")
        sys.exit(1)
    else:
        print("SUCCESS: 0 mismatches found.")
        sys.exit(0)

if __name__ == "__main__":
    main()
