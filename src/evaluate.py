"""Evaluasi: baseline vs metode usulan vs ablasi pada data uji.

Mendukung dua mode:

1. **Satu scene** (layout lama):
       python -m src.evaluate --images data/raw --background data/background \\
           --roi data/roi/roi.json --gt data/ground_truth/ground_truth.csv \\
           --out data/outputs [--save-overlays] [--split 0.3]

2. **Multi-scene** (satu area, beberapa titik kamera; paket I-07). Setiap scene
   punya latar & ROI sendiri, lalu metrik digabung (pooled) antar scene:
       python -m src.evaluate --scenes data/scenes --out data/outputs [--save-overlays]

   Tata letak tiap scene:
       data/scenes/<scene>/raw/*.jpg
       data/scenes/<scene>/background/*.jpg   (atau satu berkas background.jpg)
       data/scenes/<scene>/roi.json
       data/scenes/<scene>/ground_truth.csv

Metrik dihitung pada level SLOT (akurasi, precision, recall, F1 makro) +
confusion matrix, disimpan sebagai CSV dan Markdown.
"""

import argparse
import csv
import json
import random
from pathlib import Path

from .config import VERSIONS, Params
from .pipeline import detect
from .preprocess import load_image
from .roi import draw_overlay, load_slots
from .segment import compute_background, load_reference_frame

POSITIVE = "terisi"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def _norm_label(value):
    s = str(value).strip().lower()
    if s in {"1", "terisi", "occupied", "ada", "true", "ya"}:
        return POSITIVE
    if s in {"0", "kosong", "empty", "free", "tidak", "false"}:
        return "kosong"
    raise ValueError(f"Label ground truth tidak dikenal: {value!r}")


def read_ground_truth(path):
    """CSV: kolom 'image' lalu satu kolom per slot (nama kolom = nama slot)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    gt = {}
    for row in rows:
        image = (row.get("image") or "").strip()
        if not image:
            continue
        gt[image] = {
            k: _norm_label(v)
            for k, v in row.items()
            if k and k != "image" and str(v).strip() != ""
        }
    return gt


def list_images(images):
    path = Path(images)
    if path.is_file():
        return [path]
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXT)


def split_images(images, test_frac=0.0, seed=42):
    """Bagi citra menjadi (latih, uji) secara deterministik.

    Bila `test_frac` <= 0 atau citra terlalu sedikit, tidak ada pembagian dan
    seluruh citra dipakai untuk evaluasi (mengembalikan ([], semua)).
    """
    images = list(images)
    if not (0.0 < test_frac < 1.0) or len(images) < 2:
        return [], images
    idx = list(range(len(images)))
    random.Random(seed).shuffle(idx)
    n_test = max(1, int(round(len(images) * test_frac)))
    test_idx = set(idx[:n_test])
    test = [images[i] for i in range(len(images)) if i in test_idx]
    train = [images[i] for i in range(len(images)) if i not in test_idx]
    return train, test


def confusion(pairs):
    """pairs: list (truth, pred). Positif = 'terisi'."""
    tp = fp = fn = tn = 0
    for truth, pred in pairs:
        if truth == POSITIVE and pred == POSITIVE:
            tp += 1
        elif truth != POSITIVE and pred == POSITIVE:
            fp += 1
        elif truth == POSITIVE and pred != POSITIVE:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def metrics_from_counts(tp, fp, fn, tn):
    total = tp + fp + fn + tn

    def prf(tp_, fp_, fn_):
        p = tp_ / (tp_ + fp_) if (tp_ + fp_) else 0.0
        r = tp_ / (tp_ + fn_) if (tp_ + fn_) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f

    p_pos, r_pos, f_pos = prf(tp, fp, fn)
    p_neg, r_neg, f_neg = prf(tn, fn, fp)  # kelas "kosong" sebagai positif
    acc = (tp + tn) / total if total else 0.0
    return {
        "accuracy": acc,
        "precision": p_pos,
        "recall": r_pos,
        "f1": f_pos,
        "precision_macro": (p_pos + p_neg) / 2,
        "recall_macro": (r_pos + r_neg) / 2,
        "f1_macro": (f_pos + f_neg) / 2,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_slot": total,
    }


def metrics(pairs):
    return metrics_from_counts(*confusion(pairs))


def evaluate_version(name, p: Params, images, gt, slots, background_paths, reference_path):
    """Evaluasi satu versi pada satu scene. Mengembalikan (metrik, per_citra)."""
    if p.method == "bgsub":
        reference = compute_background(background_paths, p)
    else:
        ref = reference_path or background_paths[0]
        reference = load_reference_frame(ref, p)

    pairs = []
    per_image = {}
    for img_path in images:
        bgr = load_image(img_path, p.resize)
        res = detect(bgr, reference, slots, p)
        preds = {s["name"]: s["status"] for s in res["states"]}
        per_image[img_path.name] = preds
        truth = gt.get(img_path.name)
        if truth is None:
            continue
        for s in slots:
            if s["name"] in truth:
                pairs.append((truth[s["name"]], preds[s["name"]]))

    result = metrics(pairs)
    result["version"] = name
    return result, per_image


def _save_overlays(images, background_paths, slots, p, ov_dir):
    import cv2

    reference = compute_background(background_paths, p) if p.method == "bgsub" \
        else load_reference_frame(background_paths[0], p)
    ov_dir.mkdir(parents=True, exist_ok=True)
    for img_path in images:
        bgr = load_image(img_path, p.resize)
        res = detect(bgr, reference, slots, p)
        cv2.imwrite(str(ov_dir / f"{img_path.stem}_overlay.jpg"),
                    draw_overlay(bgr, res["states"]))


def discover_scenes(root):
    """Telusuri folder multi-scene: tiap subfolder yang punya roi.json."""
    root = Path(root)
    scenes = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        roi = d / "roi.json"
        if not roi.exists():
            continue
        raw_dir = d / "raw"
        if raw_dir.is_dir():
            images = list_images(raw_dir)
        else:
            images = [p for p in list_images(d) if not p.name.startswith("background")]
        bg_dir = d / "background"
        if bg_dir.is_dir():
            background = list_images(bg_dir)
        else:
            background = sorted(p for p in list_images(d)
                                if p.name.startswith("background"))
        gt_path = d / "ground_truth.csv"
        gt = read_ground_truth(gt_path) if gt_path.exists() else {}
        if images and background:
            scenes.append({
                "name": d.name, "images": images, "background_paths": background,
                "slots": load_slots(roi), "gt": gt,
            })
    return scenes


# --------------------------------------------------------------------------- #
# Penulisan keluaran
# --------------------------------------------------------------------------- #

FIELDS = ["version", "accuracy", "precision", "recall", "f1",
          "precision_macro", "recall_macro", "f1_macro",
          "tp", "fp", "fn", "tn", "n_slot"]


def _pct(x):
    return f"{x * 100:.2f}%"


def _write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items() if k in fields})


def _md_table(results):
    out = [
        "| Versi | Akurasi | Precision (makro) | Recall (makro) | F1 (makro) | TP | FP | FN | TN |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        out.append(
            f"| {r['version']} | {_pct(r['accuracy'])} | {_pct(r['precision_macro'])} | "
            f"{_pct(r['recall_macro'])} | {_pct(r['f1_macro'])} | "
            f"{r['tp']} | {r['fp']} | {r['fn']} | {r['tn']} |"
        )
    return out


def aggregate_scenes(rows_per_scene):
    """Gabungkan confusion matrix antar scene (pooled) per versi, lalu hitung metrik."""
    acc = {name: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for name in VERSIONS}
    for r in rows_per_scene:
        acc[r["version"]]["tp"] += r["tp"]
        acc[r["version"]]["fp"] += r["fp"]
        acc[r["version"]]["fn"] += r["fn"]
        acc[r["version"]]["tn"] += r["tn"]
    results = []
    for name in VERSIONS:
        m = metrics_from_counts(**acc[name])
        m["version"] = name
        results.append(m)
    return results


def run_single(args):
    background_paths = list_images(args.background)
    if not background_paths:
        raise SystemExit("Folder latar kosong: " + args.background)

    images = list_images(args.images)
    gt = read_ground_truth(args.gt)
    slots = load_slots(args.roi)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, test = split_images(images, args.split, args.seed)
    eval_images = test or images

    results = []
    usulan_per_image = {}
    for name, p in VERSIONS.items():
        res, per_image = evaluate_version(
            name, p, eval_images, gt, slots, background_paths, args.reference
        )
        results.append(res)
        if name == "usulan":
            usulan_per_image = per_image
        print(f"[{name}] akurasi={res['accuracy']:.4f} f1_macro={res['f1_macro']:.4f} "
              f"n_slot={res['n_slot']}")

    _write_csv(out_dir / "results.csv", results, FIELDS)

    lines = ["# Hasil Evaluasi (level slot)", ""]
    if args.split > 0 and len(eval_images) != len(images):
        lines.append(f"Pembagian data: {len(images) - len(eval_images)} latih / "
                     f"{len(eval_images)} uji (test_frac={args.split}, seed={args.seed}).")
        lines.append("")
    lines.append(f"Data uji: {len(eval_images)} citra, {results[0]['n_slot']} slot total.")
    lines += [""] + _md_table(results)
    lines += ["", "Confusion matrix (positif = terisi) per versi:", ""]
    for r in results:
        lines.append(f"- **{r['version']}**: TP={r['tp']}, FP={r['fp']}, "
                     f"FN={r['fn']}, TN={r['tn']}")
    (out_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if args.save_overlays:
        _save_overlays(eval_images, background_paths, slots,
                       VERSIONS["usulan"], out_dir / "overlays")

    with open(out_dir / "per_image.json", "w", encoding="utf-8") as f:
        json.dump(usulan_per_image, f, indent=2, ensure_ascii=False)

    print(f"\nTersimpan: {out_dir/'results.csv'}, {out_dir/'results.md'}, "
          f"{out_dir/'per_image.json'}")


def run_scenes(args):
    scenes = discover_scenes(args.scenes)
    if not scenes:
        raise SystemExit(f"Tidak ada scene (subfolder ber-roi.json) di {args.scenes}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    per_image = {}
    for sc in scenes:
        _, test = split_images(sc["images"], args.split, args.seed)
        eval_images = test or sc["images"]
        print(f"== scene '{sc['name']}': {len(eval_images)} citra, "
              f"{len(sc['slots'])} slot")
        for name, p in VERSIONS.items():
            res, pi = evaluate_version(name, p, eval_images, sc["gt"], sc["slots"],
                                       sc["background_paths"], None)
            row = dict(res)
            row["scene"] = sc["name"]
            rows.append(row)
            if name == "usulan":
                per_image[sc["name"]] = pi
                if args.save_overlays:
                    _save_overlays(eval_images, sc["background_paths"], sc["slots"],
                                   p, out_dir / "overlays" / sc["name"])
            print(f"   [{name}] akurasi={res['accuracy']:.4f} n_slot={res['n_slot']}")

    aggregate = aggregate_scenes(rows)
    _write_csv(out_dir / "results.csv", aggregate, FIELDS)
    _write_csv(out_dir / "results_per_scene.csv", rows, ["scene"] + FIELDS)

    lines = ["# Hasil Evaluasi Multi-Scene (level slot)", "",
             f"Scene: {len(scenes)} — {', '.join(s['name'] for s in scenes)}", "",
             "## Gabungan (pooled) antar scene", ""]
    lines += _md_table(aggregate)
    lines += ["", "## Rincian per scene", "",
              "| Scene | Versi | Akurasi | F1 (makro) | TP | FP | FN | TN |",
              "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['scene']} | {r['version']} | {_pct(r['accuracy'])} | "
                     f"{_pct(r['f1_macro'])} | {r['tp']} | {r['fp']} | {r['fn']} | {r['tn']} |")
    (out_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with open(out_dir / "per_image.json", "w", encoding="utf-8") as f:
        json.dump(per_image, f, indent=2, ensure_ascii=False)

    print(f"\nTersimpan: {out_dir/'results.csv'}, {out_dir/'results_per_scene.csv'}, "
          f"{out_dir/'results.md'}, {out_dir/'per_image.json'}")


def main():
    ap = argparse.ArgumentParser(description="Evaluasi baseline vs usulan vs ablasi")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", help="folder/berkas citra uji (mode satu scene)")
    src.add_argument("--scenes", help="folder berisi subfolder scene (mode multi-scene)")
    ap.add_argument("--background", help="folder citra latar kosong (mode satu scene)")
    ap.add_argument("--roi", help="berkas roi.json (mode satu scene)")
    ap.add_argument("--gt", help="ground_truth.csv (mode satu scene)")
    ap.add_argument("--reference", default=None,
                    help="frame acuan tunggal untuk baseline (default: citra latar pertama)")
    ap.add_argument("--split", type=float, default=0.0,
                    help="fraksi data uji (mis. 0.3 untuk 70/30; 0 = pakai semua)")
    ap.add_argument("--seed", type=int, default=42, help="benih pembagian data")
    ap.add_argument("--out", default="data/outputs", help="folder keluaran")
    ap.add_argument("--save-overlays", action="store_true",
                    help="simpan citra overlay untuk metode usulan")
    args = ap.parse_args()

    if args.scenes:
        run_scenes(args)
    else:
        if not (args.background and args.roi and args.gt):
            ap.error("mode satu scene butuh --background, --roi, dan --gt")
        run_single(args)


if __name__ == "__main__":
    main()
