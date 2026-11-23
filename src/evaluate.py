"""Evaluasi: baseline vs metode usulan vs ablasi pada data uji.

Menghasilkan tabel metrik level SLOT (akurasi, precision, recall, F1 makro) +
confusion matrix, lalu menyimpannya sebagai CSV dan Markdown.

Contoh pemakaian:
    python -m src.evaluate --images data/raw --background data/background \
        --roi data/roi/roi.json --gt data/ground_truth/ground_truth.csv \
        --out data/outputs
"""

import argparse
import csv
import json
from pathlib import Path

from .config import TARGET_SIZE, VERSIONS, Params
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


def metrics(pairs):
    tp, fp, fn, tn = confusion(pairs)
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


def evaluate_version(name, p: Params, images, gt, slots, background_paths, reference_path):
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


def main():
    ap = argparse.ArgumentParser(description="Evaluasi baseline vs usulan vs ablasi")
    ap.add_argument("--images", required=True, help="folder/berkas citra uji")
    ap.add_argument("--background", required=True, help="folder citra latar kosong")
    ap.add_argument("--roi", required=True, help="berkas roi.json")
    ap.add_argument("--gt", required=True, help="ground_truth.csv")
    ap.add_argument("--reference", default=None,
                    help="frame acuan tunggal untuk baseline (default: citra latar pertama)")
    ap.add_argument("--out", default="data/outputs", help="folder keluaran")
    ap.add_argument("--save-overlays", action="store_true",
                    help="simpan citra overlay untuk metode usulan")
    args = ap.parse_args()

    background_paths = sorted(
        p for p in Path(args.background).iterdir() if p.suffix.lower() in IMAGE_EXT
    )
    if not background_paths:
        raise SystemExit("Folder latar kosong: " + args.background)

    images = list_images(args.images)
    gt = read_ground_truth(args.gt)
    slots = load_slots(args.roi)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    usulan_per_image = {}
    for name, p in VERSIONS.items():
        res, per_image = evaluate_version(
            name, p, images, gt, slots, background_paths, args.reference
        )
        results.append(res)
        if name == "usulan":
            usulan_per_image = per_image
        print(f"[{name}] akurasi={res['accuracy']:.4f} f1_macro={res['f1_macro']:.4f} "
              f"n_slot={res['n_slot']}")

    # --- simpan CSV ---
    fields = ["version", "accuracy", "precision", "recall", "f1",
              "precision_macro", "recall_macro", "f1_macro",
              "tp", "fp", "fn", "tn", "n_slot"]
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    # --- simpan Markdown (siap tempel ke Bab 4) ---
    def pct(x):
        return f"{x * 100:.2f}%"

    lines = [
        "# Hasil Evaluasi (level slot)",
        "",
        f"Data uji: {len(images)} citra, {results[0]['n_slot']} slot total.",
        "",
        "| Versi | Akurasi | Precision (makro) | Recall (makro) | F1 (makro) | TP | FP | FN | TN |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['version']} | {pct(r['accuracy'])} | {pct(r['precision_macro'])} | "
            f"{pct(r['recall_macro'])} | {pct(r['f1_macro'])} | "
            f"{r['tp']} | {r['fp']} | {r['fn']} | {r['tn']} |"
        )
    lines += ["", "Confusion matrix (positif = terisi) per versi:", ""]
    for r in results:
        lines.append(
            f"- **{r['version']}**: TP={r['tp']}, FP={r['fp']}, FN={r['fn']}, TN={r['tn']}"
        )
    (out_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- overlay metode usulan (opsional) ---
    if args.save_overlays:
        usulan = VERSIONS["usulan"]
        reference = compute_background(background_paths, usulan)
        ov_dir = out_dir / "overlays"
        ov_dir.mkdir(exist_ok=True)
        for img_path in images:
            bgr = load_image(img_path, usulan.resize)
            res = detect(bgr, reference, slots, usulan)
            overlay = draw_overlay(bgr, res["states"])
            import cv2
            cv2.imwrite(str(ov_dir / f"{img_path.stem}_overlay.jpg"), overlay)

    with open(out_dir / "per_image.json", "w", encoding="utf-8") as f:
        json.dump(usulan_per_image, f, indent=2, ensure_ascii=False)

    print(f"\nTersimpan: {out_dir/'results.csv'}, {out_dir/'results.md'}, {out_dir/'per_image.json'}")


if __name__ == "__main__":
    main()
