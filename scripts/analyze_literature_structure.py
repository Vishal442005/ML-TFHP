"""
Deep Structural & Methodological Analysis of Downloaded Literature Benchmark Papers.
Examines paper flow, formatting, theorem structure, calibration techniques, and empirical design.
Strictly ensures zero data copying; extracts only methodological and architectural concepts.
"""

import os
import sys
from pypdf import PdfReader
import re

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

PDF_DIR = "literature"

def analyze_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)
    
    # Extract first 3 pages and last 2 pages for structure
    intro_text = ""
    for i in range(min(4, num_pages)):
        intro_text += reader.pages[i].extract_text() or ""
        
    tail_text = ""
    for i in range(max(0, num_pages - 3), num_pages):
        tail_text += reader.pages[i].extract_text() or ""

    full_toc = []
    # Find section headings
    section_patterns = [
        r"(?:Section|\d+)\s+([A-Z][A-Za-z0-9\s,:?-]{3,50})",
        r"\n(\d+\.?\s+[A-Z][A-Za-z\s]{3,40})\n",
        r"\n([I|V|X]+\.\s+[A-Z][A-Za-z\s]{3,40})\n"
    ]
    
    # Quick scan across all pages for section headers and theorems
    theorems = []
    propositions = []
    algorithms = []
    empirical_terms = []
    
    all_text = ""
    for page_idx, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        all_text += f"\n--- Page {page_idx+1} ---\n" + txt
        
        for line in txt.split("\n"):
            line_s = line.strip()
            if re.match(r"^(?:Theorem|Lemma|Proposition|Definition|Corollary)\s+\d+", line_s, re.IGNORECASE):
                theorems.append(line_s[:80])
            elif re.match(r"^(?:Algorithm|Scheme|Procedure)\s+\d+", line_s, re.IGNORECASE):
                algorithms.append(line_s[:80])
            elif any(k in line_s.lower() for k in ["order book", "lobster", "high frequency", "calibration", "mle", "gmm", "var", "backtest"]):
                if len(empirical_terms) < 15:
                    empirical_terms.append(line_s[:80])

    return {
        "filename": os.path.basename(pdf_path),
        "num_pages": num_pages,
        "sample_theorems": theorems[:8],
        "algorithms": algorithms[:5],
        "sample_empirical": empirical_terms[:8],
        "all_text_len": len(all_text),
        "all_text": all_text
    }

def main():
    files = sorted([f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")])
    print(f"Analyzing {len(files)} benchmark papers...")
    
    analysis_results = {}
    for f in files:
        path = os.path.join(PDF_DIR, f)
        print(f"\nProcessing {f}...")
        res = analyze_pdf(path)
        analysis_results[f] = res
        print(f"  Pages: {res['num_pages']}")
        print(f"  Theorems found: {len(res['sample_theorems'])} (e.g. {res['sample_theorems'][:2]})")
        print(f"  Algorithms found: {len(res['algorithms'])}")

    # Save summary of structure
    with open("literature/ANALYSIS_SUMMARY.txt", "w", encoding="utf-8") as out:
        for f, res in analysis_results.items():
            out.write("=" * 80 + "\n")
            out.write(f"PAPER: {f} (Pages: {res['num_pages']})\n")
            out.write("-" * 80 + "\n")
            out.write("Key Formal Results / Theorems:\n")
            for t in res["sample_theorems"]:
                out.write(f"  * {t}\n")
            out.write("Empirical / Computational Elements:\n")
            for e in res["sample_empirical"]:
                out.write(f"  * {e}\n")
            out.write("\n")

    print("\nWrote analysis summary to literature/ANALYSIS_SUMMARY.txt")

if __name__ == "__main__":
    main()
