"""Evaluasi database adegan: baseline vs usulan vs ablasi.

Perintah utama::

    python -m src.evaluate \
        --database data/database \
        --images data/raw \
        --gt data/ground_truth/ground_truth.csv \
        --out data/outputs --save-overlays

Keluaran:
- ``results.csv``       : satu baris per versi (metrik level slot)
- ``results.md``        : tabel + confusion matrix + sensitivitas + kasus gagal
- ``sensitivity.csv``   : sensitivitas ambang rasio (metode usulan)
- ``failures.csv``      : slot salah klasifikasi & foto ditolak
- ``per_image.json``    : prediksi per citra untuk tiap versi
- ``overlays/``         : overlay metode usulan (opsional)

Foto yang tidak cocok ke adegan mana pun DITOLAK dan dicatat, bukan ditebak.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass, replace
from pathlib import Path

import cv2

from .config import (SENSITIVITY_THRESHOLDS, VERSION_ORDER, VERSIONS, Params)
from .database import IMAGE_EXT, SceneDatabase
from .pipeline import detect_scene
from .preprocess import load_image
from .roi import draw_overlay, load_slots

POSITIVE = "terisi"


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #
def _norm_label(value):
    s = str(value).strip().lower()
    if s in {"1", "terisi", "occupied", "ada", "true", "ya"}:
        return POSITIVE
    if s in {"0", "kosong", "empty", "free", "tidak", "false"}:
        return "kosong"
    raise ValueError(f"Label ground truth tidak dikenal: {value!r}")


def _gt_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_ground_truth(path):
    """Baca label per citra (kompatibel format lama, kolom ``scene`` diabaikan).

    Mengembalikan ``{nama_citra: {nama_slot: label}}``.
    """
    gt = {}
    for row in _gt_rows(path):
        image = (row.get("image") or "").strip()
        if not image:
            continue
        gt[image] = {
            k: _norm_label(v)
            for k, v in row.items()
            if k and k not in {"image", "scene"} and str(v).strip() != ""
        }
    return gt


def read_ground_truth_with_scene(path):
    """Baca GT baru: ``{nama_citra: {"scene": id|None, "labels": {...}}}``."""
    gt = {}
    for row in _gt_rows(path):
        image = (row.get("image") or "").strip()
        if not image:
            continue
        scene = (row.get("scene") or "").strip() or None
        labels = {
            k: _norm_label(v)
            for k, v in row.items()
            if k and k not in {"image", "scene"} and str(v).strip() != ""
        }
        gt[image] = {"scene": scene, "labels": labels}
    return gt


# --------------------------------------------------------------------------- #
# Utilitas data & metrik
# --------------------------------------------------------------------------- #
def list_images(images):
    path = Path(images)
    if path.is_file():
        return [path]
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXT)


def split_images(images, test_frac=0.0, seed=42):
    """Bagi citra menjadi (latih, uji) secara deterministik."""
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
    p_neg, r_neg, f_neg = prf(tn, fn, fp)
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


def aggregate_scenes(rows_per_scene):
    """Gabungkan confusion matrix antar scene (kompatibilitas format lama)."""
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


def discover_scenes(root):
    """Telusuri layout lama (data/scenes/<nama>/...) untuk kompatibilitas."""
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
# Evaluasi database
# --------------------------------------------------------------------------- #
@dataclass
class ImageRecord:
    name: str
    path: Path
    image: np.ndarray
    gt_scene: str | None
    gt_labels: dict
    match: object
    scene: object | None
    scene_for_eval: object | None
    reason: str = ""


def _resolve_records(db: SceneDatabase, images, gt, params: Params):
    records = []
    for path in images:
        entry = gt.get(path.name)
        if entry is None:
            print(f"  lewati (tidak ada baris GT): {path.name}")
            continue
        img = load_image(path)
        match = db.match(img, params)
        gt_scene = entry.get("scene")
        if gt_scene:
            scene = db.get(gt_scene)
            reason = "" if scene else f"adegan GT '{gt_scene}' tidak ada di database"
        else:
            scene = match.scene
            reason = match.reason
        records.append(ImageRecord(
            name=path.name,
            path=path,
            image=img,
            gt_scene=gt_scene,
            gt_labels=entry.get("labels", {}),
            match=match,
            scene=scene,
            scene_for_eval=scene,
            reason=reason,
        ))
    return records


def _match_for_scene(record, scene):
    """Pakai homografi hasil matching hanya bila adegannya sama."""
    m = record.match
    if m is not None and m.scene is not None and scene is not None \
            and m.scene.id == scene.id:
        return m.result
    return None


def evaluate_version(name, p: Params, records):
    """Evaluasi satu versi. Mengembalikan (metrik, per_citra, slot_records, ditolak)."""
    pairs = []
    per_image = {}
    slot_records = []
    rejected = []

    for rec in records:
        if rec.scene_for_eval is None:
            rejected.append({
                "image": rec.name, "scene": rec.gt_scene,
                "reason": rec.reason or "adegan tidak dikenali",
            })
            continue
        match = _match_for_scene(rec, rec.scene_for_eval)
        res = detect_scene(rec.image, rec.scene_for_eval, p, match=match)
        if not res.get("recognized", False):
            rejected.append({
                "image": rec.name, "scene": rec.scene_for_eval.id,
                "reason": res.get("reason", "registrasi gagal"),
            })
            continue

        preds = {s["name"]: s["status"] for s in res["states"]}
        ratios = {s["name"]: s["ratio"] for s in res["states"]}
        per_image[rec.name] = {
            "scene": rec.scene_for_eval.id,
            "states": res["states"],
            "free": res["free"],
            "total": res["total"],
            "threshold": res["threshold"],
        }
        for slot in rec.scene_for_eval.slots:
            name_slot = slot["name"]
            if name_slot not in rec.gt_labels:
                continue
            truth = rec.gt_labels[name_slot]
            pred = preds[name_slot]
            pairs.append((truth, pred))
            slot_records.append({
                "image": rec.name,
                "scene": rec.scene_for_eval.id,
                "slot": name_slot,
                "truth": truth,
                "pred": pred,
                "ratio": ratios[name_slot],
                "threshold": res["threshold"],
            })

    result = metrics(pairs)
    result["version"] = name
    result["n_images"] = len(records)
    result["n_accepted"] = len(per_image)
    result["n_rejected"] = len(rejected)
    return result, per_image, slot_records, rejected


def threshold_sensitivity(slot_records, thresholds=SENSITIVITY_THRESHOLDS):
    rows = []
    for t in thresholds:
        pairs = [(r["truth"], POSITIVE if r["ratio"] >= t else "kosong")
                 for r in slot_records]
        m = metrics(pairs)
        m["threshold"] = float(t)
        rows.append(m)
    return rows


def _scene_recognition(records):
    with_gt = [r for r in records if r.gt_scene]
    correct = 0
    rejected = 0
    wrong = []
    for r in with_gt:
        if not r.match.recognized:
            rejected += 1
            continue
        if r.match.scene.id == r.gt_scene:
            correct += 1
        else:
            wrong.append({
                "image": r.name, "truth": r.gt_scene,
                "pred": r.match.scene.id,
                "inliers": r.match.result.inliers if r.match.result else 0,
            })
    total = len(with_gt)
    return {
        "n_with_gt": total,
        "correct": correct,
        "rejected": rejected,
        "wrong": wrong,
        "accuracy": correct / total if total else 0.0,
    }


# --------------------------------------------------------------------------- #
# Penulisan keluaran
# --------------------------------------------------------------------------- #
FIELDS = ["version", "accuracy", "precision", "recall", "f1",
          "precision_macro", "recall_macro", "f1_macro",
          "tp", "fp", "fn", "tn", "n_slot", "n_images", "n_accepted", "n_rejected"]


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
        "| Versi | Akurasi | Precision (makro) | Recall (makro) | F1 (makro) | "
        "TP | FP | FN | TN | Ditolak |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        out.append(
            f"| {r['version']} | {_pct(r['accuracy'])} | {_pct(r['precision_macro'])} | "
            f"{_pct(r['recall_macro'])} | {_pct(r['f1_macro'])} | "
            f"{r['tp']} | {r['fp']} | {r['fn']} | {r['tn']} | {r['n_rejected']} |"
        )
    return out


def _md_sensitivity(rows):
    out = [
        "| Ambang rasio | Akurasi | Precision (makro) | Recall (makro) | F1 (makro) |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        out.append(f"| {r['threshold']:.2f} | {_pct(r['accuracy'])} | "
                   f"{_pct(r['precision_macro'])} | {_pct(r['recall_macro'])} | "
                   f"{_pct(r['f1_macro'])} |")
    return out


def _save_overlays(records, p: Params, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for rec in records:
        if rec.scene_for_eval is None:
            canvas = rec.image.copy()
            cv2.putText(canvas, "TIDAK DIKENALI - perlu foto latar baru",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),
                        2, cv2.LINE_AA)
            cv2.imwrite(str(out_dir / f"{Path(rec.name).stem}_rejected.jpg"), canvas)
            continue
        match = _match_for_scene(rec, rec.scene_for_eval)
        res = detect_scene(rec.image, rec.scene_for_eval, p, match=match)
        if not res.get("recognized", False):
            canvas = rec.image.copy()
            cv2.putText(canvas, "TIDAK DIKENALI - perlu foto latar baru",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),
                        2, cv2.LINE_AA)
            cv2.imwrite(str(out_dir / f"{Path(rec.name).stem}_rejected.jpg"), canvas)
            continue
        cv2.imwrite(str(out_dir / f"{Path(rec.name).stem}_overlay.jpg"),
                    draw_overlay(rec.image, res["states"]))


def _parse_size(text):
    w, h = (int(v) for v in str(text).lower().split("x"))
    return (w, h)


def run_evaluate(args):
    size = _parse_size(getattr(args, "size", "1280x960"))
    params = {name: replace(p, resize=size) for name, p in VERSIONS.items()}
    db = SceneDatabase(args.database, size=size)
    if not db.scenes:
        raise SystemExit(
            f"Database adegan kosong di {args.database}. Isi "
            f"<id_adegan>/refs/ lalu jalankan `python -m src.database build`."
        )

    images = list_images(args.images)
    if not images:
        raise SystemExit(f"Tidak ada citra uji di {args.images}")
    gt = read_ground_truth_with_scene(args.gt)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = _resolve_records(db, images, gt, params["usulan"])
    if not records:
        raise SystemExit("Tidak ada citra yang punya baris ground truth.")

    results, per_version, slot_records_by_version, rejected_by_version = [], {}, {}, {}
    for name in VERSION_ORDER:
        res, per_image, slot_records, rejected = evaluate_version(
            name, params[name], records
        )
        results.append(res)
        per_version[name] = per_image
        slot_records_by_version[name] = slot_records
        rejected_by_version[name] = rejected
        print(f"[{name}] akurasi={res['accuracy']:.4f} "
              f"f1_macro={res['f1_macro']:.4f} n_slot={res['n_slot']} "
              f"ditolak={res['n_rejected']}")

    _write_csv(out_dir / "results.csv", results, FIELDS)

    sensitivity = threshold_sensitivity(slot_records_by_version["usulan"])
    _write_csv(out_dir / "sensitivity.csv", sensitivity,
               ["threshold", "accuracy", "precision", "recall", "f1",
                "precision_macro", "recall_macro", "f1_macro",
                "tp", "fp", "fn", "tn", "n_slot"])

    failures = []
    for r in slot_records_by_version["usulan"]:
        if r["truth"] != r["pred"]:
            failures.append({
                "type": "salah_klasifikasi", "version": "usulan",
                "image": r["image"], "scene": r["scene"], "slot": r["slot"],
                "truth": r["truth"], "pred": r["pred"], "ratio": r["ratio"],
                "reason": "",
            })
    for r in rejected_by_version["usulan"]:
        failures.append({
            "type": "ditolak", "version": "usulan", "image": r["image"],
            "scene": r["scene"], "slot": "", "truth": "", "pred": "",
            "ratio": "", "reason": r["reason"],
        })
    _write_csv(out_dir / "failures.csv", failures,
               ["type", "version", "image", "scene", "slot", "truth",
                "pred", "ratio", "reason"])

    recognition = _scene_recognition(records)
    lines = ["# Hasil Evaluasi (level slot)", ""]
    if "sandbox" in str(args.database).lower() or "sandbox" in str(args.images).lower():
        lines += ["> **PERINGATAN:** hasil ini berasal dari data SINTETIS untuk "
                  "smoke test. Jangan dilaporkan sebagai hasil penelitian.", ""]
    lines += [
        "## Data",
        "",
        f"- Database adegan: `{args.database}` ({len(db.scenes)} adegan)",
        f"- Citra uji: `{args.images}` ({len(records)} citra ber-GT)",
        f"- Ground truth: `{args.gt}`",
        "",
        "## Tabel baseline vs usulan vs ablasi",
        "",
    ]
    lines += _md_table(results)
    lines += ["", "Confusion matrix (positif = terisi):", ""]
    for r in results:
        lines.append(f"- **{r['version']}**: TP={r['tp']}, FP={r['fp']}, "
                     f"FN={r['fn']}, TN={r['tn']}, ditolak={r['n_rejected']}")
    lines += ["", "## Sensitivitas ambang (metode usulan)", ""]
    lines += _md_sensitivity(sensitivity)
    lines += ["", "## Rekognisi adegan", "",
              f"- Citra ber-GT adegan: {recognition['n_with_gt']}",
              f"- Cocok benar: {recognition['correct']} "
              f"({_pct(recognition['accuracy'])})",
              f"- Ditolak: {recognition['rejected']}",
              f"- Salah adegan: {len(recognition['wrong'])}", ""]
    for w in recognition["wrong"]:
        lines.append(f"  - `{w['image']}`: GT={w['truth']}, pred={w['pred']} "
                     f"(inlier={w['inliers']})")
    lines += ["", "## Kasus gagal (metode usulan)", ""]
    if not failures:
        lines.append("Tidak ada kasus gagal pada data ini.")
    else:
        lines += ["| Tipe | Citra | Adegan | Slot | GT | Pred | Rasio | Alasan |",
                  "|---|---|---|---|---|---|---|---|"]
        for f in failures:
            lines.append(f"| {f['type']} | {f['image']} | {f['scene']} | "
                         f"{f['slot']} | {f['truth']} | {f['pred']} | "
                         f"{f['ratio']} | {f['reason']} |")
    (out_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with open(out_dir / "per_image.json", "w", encoding="utf-8") as f:
        json.dump(per_version, f, indent=2, ensure_ascii=False)

    if args.save_overlays:
        _save_overlays(records, params["usulan"], out_dir / "overlays")

    print(f"\nTersimpan: {out_dir/'results.csv'}, {out_dir/'results.md'}, "
          f"{out_dir/'sensitivity.csv'}, {out_dir/'failures.csv'}, "
          f"{out_dir/'per_image.json'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluasi baseline vs usulan vs ablasi")
    ap.add_argument("--database", default="data/database",
                    help="folder database adegan (hasil src.database build)")
    ap.add_argument("--images", required=True, help="folder/berkas citra uji")
    ap.add_argument("--gt", required=True, help="ground_truth.csv (image,scene,slot...)")
    ap.add_argument("--out", default="data/outputs", help="folder keluaran")
    ap.add_argument("--size", default="1280x960",
                    help="ukuran kerja WxH (default 1280x960)")
    ap.add_argument("--save-overlays", action="store_true",
                    help="simpan citra overlay metode usulan")
    args = ap.parse_args(argv)
    run_evaluate(args)


if __name__ == "__main__":
    main()
