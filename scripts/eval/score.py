#!/usr/bin/env python3
"""Score a bench run against curated ground truth.

  python3 score.py --results ~/vln_eval/run_x --gt gt/gt.json [--csv out.csv]

Scoring (mirrors challenge rules; instruction scorer is OUR PROXY — validate
against the provided .ply examples and refine):
  numerical            0 or 1   exact Int32 match
  object_reference     0..2     2 * 3D IoU(answer box, gt box)  [proxy for "degree of overlap"]
  instruction_following 0..6    constraint satisfaction in order, penalties for
                                avoid-zone entry and wrong order
"""
import argparse, csv, json, math, pathlib


def iou3d(c1, s1, c2, s2):
    lo = [max(c1[i] - s1[i] / 2, c2[i] - s2[i] / 2) for i in range(3)]
    hi = [min(c1[i] + s1[i] / 2, c2[i] + s2[i] / 2) for i in range(3)]
    inter = 1.0
    for i in range(3):
        d = hi[i] - lo[i]
        if d <= 0:
            return 0.0
        inter *= d
    v1 = s1[0] * s1[1] * s1[2]
    v2 = s2[0] * s2[1] * s2[2]
    return inter / (v1 + v2 - inter)


def point_in_poly(pt, poly):
    x, y, inside = pt[0], pt[1], False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            inside = not inside
    return inside


def score_instruction(res, gt):
    """gt: {pass_near:[{center,radius}...] (ordered), avoid:[{polygon}...], goal:{center,radius}}"""
    traj = [(p[1], p[2]) for p in res.get("trajectory", [])]
    if not traj:
        return 0.0, "no trajectory"
    notes = []
    max_pts = 6.0
    n_cons = len(gt.get("pass_near", [])) + 1  # + goal
    per = max_pts / max(n_cons, 1)
    score, cursor = 0.0, 0

    for c in gt.get("pass_near", []):
        hit = None
        for i in range(cursor, len(traj)):
            if math.dist(traj[i], c["center"][:2]) <= c.get("radius", 1.5):
                hit = i; break
        if hit is not None:
            score += per; cursor = hit
        else:
            # visited out of order?
            if any(math.dist(p, c["center"][:2]) <= c.get("radius", 1.5) for p in traj):
                score += per * 0.5; notes.append("out-of-order constraint")
            else:
                notes.append("missed constraint")

    g = gt.get("goal")
    if g and math.dist(traj[-1], g["center"][:2]) <= g.get("radius", 1.5):
        score += per
    else:
        notes.append("goal missed")

    for az in gt.get("avoid", []):
        if any(point_in_poly(p, az["polygon"]) for p in traj):
            score -= 1.5; notes.append("avoid-zone violation")

    return max(0.0, min(max_pts, score)), "; ".join(notes) or "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=pathlib.Path, required=True)
    ap.add_argument("--gt", type=pathlib.Path, required=True)
    ap.add_argument("--csv", type=pathlib.Path)
    args = ap.parse_args()
    gt_all = json.load(open(args.gt))

    rows, total, possible = [], 0.0, 0.0
    for f in sorted(args.results.glob("*.json")):
        res = json.load(open(f))
        scene, qid, qtype = res["scene"], res["qid"], res["qtype"]
        entry = (gt_all.get(scene, {}).get(qtype) or [])
        idx = int(qid[1:]) if qid[1:].isdigit() else 0
        g = entry[idx] if idx < len(entry) else None
        note = ""
        if g is None:
            s, mx, note = 0.0, 0.0, "NO GT — curate gt.json"
        elif qtype == "numerical":
            mx = 1.0
            s = 1.0 if res.get("numerical") == g["answer"] else 0.0
            note = f"got {res.get('numerical')} want {g['answer']}"
        elif qtype == "object_reference":
            mx = 2.0
            m = res.get("marker")
            if m:
                i = iou3d(m["center"], m["size"], g["target"]["center"], g["target"]["size"])
                s = 2.0 * i; note = f"IoU {i:.2f}"
            else:
                s, note = 0.0, "no marker"
        else:
            mx = 6.0
            s, note = score_instruction(res, g)
        if res.get("timed_out"):
            note += " [TIMED OUT]"
        rows.append([scene, qid, qtype, round(s, 2), mx, res.get("t_answer_s"), note])
        total += s; possible += mx

    w = [14, 4, 22, 6, 4, 8]
    hdr = ["scene", "qid", "qtype", "score", "max", "t_ans", "notes"]
    print(" | ".join(h.ljust(w[i]) if i < len(w) else h for i, h in enumerate(hdr)))
    for r in rows:
        print(" | ".join(str(c).ljust(w[i]) if i < len(w) else str(c) for i, c in enumerate(r)))
    print(f"\nTOTAL {total:.2f} / {possible:.0f}  ({100 * total / possible if possible else 0:.1f}%)")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            cw = csv.writer(fh); cw.writerow(hdr); cw.writerows(rows)
        print(f"CSV → {args.csv}")


if __name__ == "__main__":
    main()
