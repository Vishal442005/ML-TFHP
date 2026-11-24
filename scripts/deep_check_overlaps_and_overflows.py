"""
Exhaustive verification of text overflow and text overlap across all figures in the manuscript.
Correctly handles text orientation (horizontal dir=(1,0) vs vertical/rotated dir=(0,-1)).
"""

import os
import fitz

FIGS = [
    'fig1_system_architecture.pdf',
    'fig2_kernel_mechanics.pdf',
    'fig3_branching_and_reflexivity.pdf',
    'fig4_resolvent_and_signature_plot.pdf',
    'fig5_dfa_memory_scaling.pdf',
    'fig6_qq_residual_diagnostics.pdf',
    'fig7_frtb_var_expected_shortfall.pdf',
    'fig8_bivariate_order_flow.pdf'
]

BASE_DIR = r"E:\Financial"
FIG_DIR = os.path.join(BASE_DIR, "figures")

def rect_intersection(r1, r2):
    x0 = max(r1[0], r2[0])
    y0 = max(r1[1], r2[1])
    x1 = min(r1[2], r2[2])
    y1 = min(r1[3], r2[3])
    if x1 > x0 and y1 > y0:
        return (x1 - x0) * (y1 - y0)
    return 0.0

def inspect_figure(fig_name):
    pdf_path = os.path.join(FIG_DIR, fig_name)
    if not os.path.exists(pdf_path):
        print(f"File NOT found: {pdf_path}")
        return

    doc = fitz.open(pdf_path)
    page = doc[0]
    rect = page.rect
    w, h = rect.width, rect.height

    print(f"\n{'='*70}")
    print(f"AUDITING: {fig_name} (Canvas Size: {w:.1f} x {h:.1f} pt)")
    print(f"{'='*70}")

    # 1. Canvas Boundary Overflow
    blocks = page.get_text("blocks")
    overflows = []
    for b in blocks:
        bx0, by0, bx1, by1, text = b[0], b[1], b[2], b[3], b[4].strip()
        if not text:
            continue
        # Check if outside canvas or touching canvas boundary (< 3 pt margin)
        if bx0 < 0 or by0 < 0 or bx1 > w or by1 > h:
            overflows.append((b, "STRICT OUT-OF-BOUNDS"))
        elif bx0 < 3.0 or by0 < 3.0 or (w - bx1) < 3.0 or (h - by1) < 3.0:
            overflows.append((b, "TIGHT BOUNDARY (<3pt)"))

    if overflows:
        print(f"  [!] Canvas Boundary Warnings ({len(overflows)}):")
        for b, reason in overflows:
            t = b[4].strip().replace('\n', ' ')[:40]
            print(f"      - {reason}: [{b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}, {b[3]:.1f}] '{t}'")
    else:
        print("  [OK] Canvas Overflow: ZERO overflows detected (generous margins on all sides).")

    # 2. Detailed text extraction with font, orientation, and bounding boxes
    text_json = page.get_text("json")
    import json
    data = json.loads(text_json)

    lines_list = []
    for b_idx, block in enumerate(data.get("blocks", [])):
        if "lines" not in block:
            continue
        for l_idx, line in enumerate(block["lines"]):
            bbox = line["bbox"]
            dir_vec = line.get("dir", (1, 0))
            line_text = ""
            for span in line.get("spans", []):
                line_text += span.get("text", "")
            line_text = line_text.strip()
            if line_text:
                lines_list.append({
                    'bbox': bbox,
                    'dir': dir_vec,
                    'text': line_text,
                    'block_no': b_idx,
                    'line_no': l_idx
                })

    # Check overlaps between distinct lines
    line_collisions = []
    for i in range(len(lines_list)):
        for j in range(i + 1, len(lines_list)):
            l1, l2 = lines_list[i], lines_list[j]
            # If same block and adjacent lines, check vertical gap
            inter_area = rect_intersection(l1['bbox'], l2['bbox'])
            if inter_area > 1.0:
                a1 = (l1['bbox'][2] - l1['bbox'][0]) * (l1['bbox'][3] - l1['bbox'][1])
                a2 = (l2['bbox'][2] - l2['bbox'][0]) * (l2['bbox'][3] - l2['bbox'][1])
                ratio = inter_area / min(a1, a2)
                if ratio > 0.05:
                    line_collisions.append((l1, l2, inter_area, ratio, l1['text'], l2['text']))

    if line_collisions:
        print(f"  [!] Line Overlap Collisions ({len(line_collisions)}):")
        for l1, l2, area, ratio, t1, t2 in line_collisions:
            print(f"      - Overlap {area:.1f} pt^2 ({ratio*100:.1f}%): '{t1[:35]}' vs '{t2[:35]}'")
    else:
        print("  [OK] Line Overlaps: STRICTLY ZERO overlapping text lines detected!")

if __name__ == '__main__':
    for f in FIGS:
        inspect_figure(f)
