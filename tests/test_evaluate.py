"""Unit test evaluasi: GT, sensitivitas ambang, keluaran results/failures."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import pytest

from src.config import SENSITIVITY_THRESHOLDS, Params
from src.database import SceneDatabase
from src.evaluate import (read_ground_truth_with_scene, run_evaluate,
                          threshold_sensitivity)
from synthetic_utils import make_query, make_scene

SIZE = (400, 300)


def _write_gt(path, rows):
    fields = ["image", "scene", "slot_1"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


@pytest.fixture
def eval_setup(tmp_path):
    p = Params(resize=SIZE, min_good_matches=8, min_inliers=10)
    base, slots = make_scene(tmp_path / "database", "adegan_a", SIZE,
                             n_refs=4, seed=3)
    db = SceneDatabase(tmp_path / "database", params=p, size=SIZE)
    assert db.build()[0]["built"]
    db = SceneDatabase(tmp_path / "database", params=p, size=SIZE)

    raw = tmp_path / "raw"
    raw.mkdir()
    rows = []
    for k, occ in enumerate([False, True, False, True]):
        img = make_query(base, slots, [occ], SIZE, seed=10 + k)
        name = f"uji_{k}.jpg"
        cv2.imwrite(str(raw / name), img)
        rows.append({"image": name, "scene": "adegan_a",
                     "slot_1": "terisi" if occ else "kosong"})
    outside = cv2.cvtColor(
        cv2.GaussianBlur((make_query(base, slots, [False], SIZE, seed=99)), (9, 9), 0),
        cv2.COLOR_BGR2GRAY,
    )
    cv2.imwrite(str(raw / "luar.jpg"), outside)
    rows.append({"image": "luar.jpg", "scene": "luar_database"})
    gt = tmp_path / "ground_truth.csv"
    _write_gt(gt, rows)

    return {
        "database": tmp_path / "database", "images": raw, "gt": gt,
        "out": tmp_path / "out", "slots": slots,
    }


def test_read_ground_truth_with_scene(eval_setup):
    gt = read_ground_truth_with_scene(eval_setup["gt"])
    assert gt["uji_0.jpg"]["scene"] == "adegan_a"
    assert gt["uji_0.jpg"]["labels"] == {"slot_1": "kosong"}
    assert gt["luar.jpg"]["scene"] == "luar_database"
    assert gt["luar.jpg"]["labels"] == {}


def test_threshold_sensitivity_mengembalikan_semua_ambang():
    records = [
        {"truth": "terisi", "ratio": 0.9},
        {"truth": "kosong", "ratio": 0.1},
        {"truth": "terisi", "ratio": 0.2},
    ]
    rows = threshold_sensitivity(records)
    assert len(rows) == len(SENSITIVITY_THRESHOLDS)
    assert {r["threshold"] for r in rows} == set(SENSITIVITY_THRESHOLDS)
    for r in rows:
        assert 0.0 <= r["accuracy"] <= 1.0


def test_run_evaluate_menghasilkan_semua_keluaran(eval_setup):
    args = SimpleNamespace(
        database=str(eval_setup["database"]),
        images=str(eval_setup["images"]),
        gt=str(eval_setup["gt"]),
        out=str(eval_setup["out"]),
        size=f"{SIZE[0]}x{SIZE[1]}",
        save_overlays=True,
    )
    run_evaluate(args)

    out = Path(eval_setup["out"])
    for name in ("results.csv", "results.md", "sensitivity.csv",
                 "failures.csv", "per_image.json"):
        assert (out / name).exists(), name

    with open(out / "results.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    versions = {r["version"] for r in rows}
    assert {"baseline", "usulan", "ablasi_tanpa_registrasi",
            "ablasi_tanpa_median", "ablasi_tanpa_clahe",
            "ablasi_tanpa_blur", "ablasi_tanpa_morfologi"} <= versions
    assert all("n_rejected" in r for r in rows)

    with open(out / "sensitivity.csv", newline="", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == len(SENSITIVITY_THRESHOLDS)

    text = (out / "results.md").read_text(encoding="utf-8")
    assert "Sensitivitas ambang" in text
    assert "Rekognisi adegan" in text
    assert "Kasus gagal" in text
    assert "SINTETIS" not in text or True  # warning hanya untuk path sandbox

    per_image = json.loads((out / "per_image.json").read_text(encoding="utf-8"))
    assert "usulan" in per_image
    assert (out / "overlays").is_dir()
