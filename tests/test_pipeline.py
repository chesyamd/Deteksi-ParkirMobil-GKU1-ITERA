"""Unit test pipeline (tanpa data lapangan).

Jalankan:
    python -m unittest discover -s tests -v
"""

import csv
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classify import occupancy_ratio
from src.config import Params, PROPOSED, VERSIONS
from src.evaluate import (aggregate_scenes, discover_scenes, metrics,
                          metrics_from_counts, read_ground_truth, split_images)
from src.pipeline import detect
from src.preprocess import preprocess, to_gray
from src.roi import is_polygon, load_slots, slot_area, slot_bbox, slot_points, slot_rect
from src.roi import save_slots as save_slots_fn

SIZE = (320, 180)  # (lebar, tinggi)


def scene(value=70, car=None):
    """Citra aspal rata, opsional dengan satu kotak terang (mobil)."""
    img = np.full((SIZE[1], SIZE[0], 3), value, np.uint8)
    if car is not None:
        x, y, w, h = car
        cv2.rectangle(img, (x, y), (x + w, y + h), (235, 235, 235), -1)
    return img


class TestRoi(unittest.TestCase):
    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "roi.json"
            slots = [{"name": "slot_1", "x": 10, "y": 10, "w": 50, "h": 80}]
            save_slots_fn(path, slots, size=SIZE, note="uji")
            self.assertEqual(load_slots(path), slots)

    def test_slot_rect(self):
        self.assertEqual(slot_rect({"x": 1, "y": 2, "w": 3, "h": 4}), (1, 2, 3, 4))


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.kosong = {"name": "slot_kosong", "x": 200, "y": 20, "w": 80, "h": 120}
        self.terisi = {"name": "slot_terisi", "x": 20, "y": 20, "w": 80, "h": 120}
        self.slots = [self.terisi, self.kosong]
        self.params = Params(resize=SIZE)

    def test_slot_terisi_dan_kosong_terdeteksi(self):
        background = to_gray(scene(70))
        current = scene(70, car=(30, 30, 60, 100))
        result = detect(current, background, self.slots, self.params)
        status = {s["name"]: s["status"] for s in result["states"]}
        self.assertEqual(status["slot_terisi"], "terisi")
        self.assertEqual(status["slot_kosong"], "kosong")
        self.assertEqual(result["free"], 1)
        self.assertEqual(result["total"], 2)

    def test_rasio_nol_pada_latar_rata(self):
        mask = np.zeros((SIZE[1], SIZE[0]), np.uint8)
        self.assertEqual(occupancy_ratio(mask, self.terisi), 0.0)

    def test_semua_versi_berjalan(self):
        background = to_gray(scene(70))
        current = scene(70, car=(30, 30, 60, 100))
        for name, preset in VERSIONS.items():
            with self.subTest(versi=name):
                params = replace(preset, resize=SIZE)
                result = detect(current, background, self.slots, params)
                status = {s["name"]: s["status"] for s in result["states"]}
                self.assertEqual(status["slot_terisi"], "terisi")
                self.assertEqual(status["slot_kosong"], "kosong")

    def test_ablasi_tanpa_clahe_tetap_mendeteksi(self):
        background = to_gray(scene(70))
        current = scene(70, car=(30, 30, 60, 100))
        params = replace(VERSIONS["ablasi_tanpa_clahe"], resize=SIZE)
        result = detect(current, background, self.slots, params)
        status = {s["name"]: s["status"] for s in result["states"]}
        self.assertEqual(status["slot_terisi"], "terisi")


class TestMetrics(unittest.TestCase):
    def test_confusion_dan_metrik(self):
        pairs = [("terisi", "terisi"), ("terisi", "kosong"),
                 ("kosong", "kosong"), ("kosong", "terisi")]
        m = metrics(pairs)
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["tn"]), (1, 1, 1, 1))
        self.assertAlmostEqual(m["accuracy"], 0.5)
        self.assertAlmostEqual(m["precision"], 0.5)
        self.assertAlmostEqual(m["recall"], 0.5)
        self.assertAlmostEqual(m["f1"], 0.5)
        self.assertEqual(m["n_slot"], 4)

    def test_sempurna(self):
        m = metrics([("terisi", "terisi"), ("kosong", "kosong")])
        self.assertAlmostEqual(m["accuracy"], 1.0)
        self.assertAlmostEqual(m["f1_macro"], 1.0)

    def test_semua_salah(self):
        m = metrics([("terisi", "kosong"), ("kosong", "terisi")])
        self.assertAlmostEqual(m["accuracy"], 0.0)
        self.assertAlmostEqual(m["recall"], 0.0)

    def test_baca_ground_truth(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "gt.csv"
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["image", "slot_1", "slot_2"])
                w.writerow(["a.jpg", "terisi", "kosong"])
                w.writerow(["b.jpg", 1, 0])
            gt = read_ground_truth(path)
            self.assertEqual(gt["a.jpg"], {"slot_1": "terisi", "slot_2": "kosong"})
            self.assertEqual(gt["b.jpg"], {"slot_1": "terisi", "slot_2": "kosong"})

    def test_label_tidak_dikenal(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "gt.csv"
            path.write_text("image,slot_1\nx.jpg,mungkin\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_ground_truth(path)


class TestPreprocess(unittest.TestCase):
    def test_preprocess_menghasilkan_grayscale_seukuran_target(self):
        out = preprocess(scene(70), Params(resize=SIZE))
        self.assertEqual(out.shape, (SIZE[1], SIZE[0]))
        self.assertEqual(out.dtype, np.uint8)

    def test_gaussian_blur_meratakan_bintik(self):
        noisy = scene(70).copy()
        noisy[40:60, 40:60] = 255
        halus = preprocess(noisy, Params(resize=None, use_clahe=False, blur_kernel=5))
        kasar = preprocess(noisy, Params(resize=None, use_blur=False, use_clahe=False))
        self.assertLess(float(halus.std()), float(kasar.std()))


class TestRoiPolygon(unittest.TestCase):
    POLY = {"name": "slot_p", "points": [[150, 10], [250, 10], [250, 90], [150, 90]]}

    def test_helpers_poligon(self):
        self.assertTrue(is_polygon(self.POLY))
        self.assertEqual(slot_bbox(self.POLY), (150, 10, 100, 80))
        self.assertEqual(slot_rect(self.POLY), (150, 10, 100, 80))
        self.assertAlmostEqual(slot_area(self.POLY), 8000.0)
        self.assertEqual(slot_points(self.POLY).shape, (4, 2))

    def test_helpers_kotak(self):
        rect = {"name": "s", "x": 5, "y": 6, "w": 40, "h": 20}
        self.assertFalse(is_polygon(rect))
        self.assertEqual(slot_bbox(rect), (5, 6, 40, 20))
        self.assertAlmostEqual(slot_area(rect), 800.0)

    def test_occupancy_poligon_hanya_dalam(self):
        mask = np.zeros((SIZE[1], SIZE[0]), np.uint8)
        mask[30:130, 30:90] = 255  # mobil di kiri
        self.assertAlmostEqual(occupancy_ratio(mask, self.POLY), 0.0)
        dalam = {"name": "d", "points": [[30, 30], [90, 30], [90, 130], [30, 130]]}
        r = occupancy_ratio(mask, dalam)          # hampir seluruh poligon berisi mobil
        self.assertGreater(r, 0.95)
        self.assertLessEqual(r, 1.0)
        luas = {"name": "l", "points": [[10, 10], [110, 10], [110, 150], [10, 150]]}
        r2 = occupancy_ratio(mask, luas)          # mobil 6000 px dari ~14000 px
        self.assertGreater(r2, 0.40)
        self.assertLess(r2, 0.46)

    def test_detect_dengan_slot_poligon(self):
        background = to_gray(scene(70))
        current = scene(70, car=(30, 30, 60, 100))
        slots = [
            {"name": "terisi", "points": [[20, 20], [100, 20], [100, 140], [20, 140]]},
            {"name": "kosong", "points": [[200, 20], [300, 20], [300, 140], [200, 140]]},
        ]
        res = detect(current, background, slots, Params(resize=SIZE))
        status = {s["name"]: s["status"] for s in res["states"]}
        self.assertEqual(status["terisi"], "terisi")
        self.assertEqual(status["kosong"], "kosong")
        self.assertIn("points", res["states"][0])


class TestEvaluateExtras(unittest.TestCase):
    def test_metrics_from_counts_sama_dengan_metrics(self):
        pairs = [("terisi", "terisi"), ("terisi", "kosong"),
                 ("kosong", "kosong"), ("kosong", "terisi")]
        self.assertEqual(metrics_from_counts(1, 1, 1, 1), metrics(pairs))

    def test_split_images_deterministik_dan_tidak_tumpang(self):
        imgs = [f"img_{i}.jpg" for i in range(10)]
        train, test = split_images(imgs, 0.3, seed=1)
        self.assertEqual(len(test), 3)
        self.assertEqual(len(train), 7)
        self.assertEqual(set(train) & set(test), set())
        self.assertEqual(set(train) | set(test), set(imgs))
        self.assertEqual(split_images(imgs, 0.3, seed=1)[1], test)

    def test_split_mati_bila_nol_atau_citra_sedikit(self):
        imgs = [f"img_{i}.jpg" for i in range(10)]
        self.assertEqual(split_images(imgs, 0.0), ([], imgs))
        self.assertEqual(split_images(["a.jpg"], 0.5), ([], ["a.jpg"]))

    def test_aggregate_scenes(self):
        rows = [
            {"version": "usulan", "tp": 2, "fp": 0, "fn": 0, "tn": 1},
            {"version": "usulan", "tp": 1, "fp": 1, "fn": 0, "tn": 1},
        ]
        agg = aggregate_scenes(rows)
        u = [r for r in agg if r["version"] == "usulan"][0]
        self.assertEqual((u["tp"], u["fp"], u["fn"], u["tn"]), (3, 1, 0, 2))
        self.assertEqual(u["n_slot"], 6)

    def test_discover_scenes(self):
        with tempfile.TemporaryDirectory() as d:
            sc = Path(d) / "sceneA"
            (sc / "raw").mkdir(parents=True)
            (sc / "background").mkdir()
            cv2.imwrite(str(sc / "raw" / "a.jpg"), scene(70, (30, 30, 60, 100)))
            cv2.imwrite(str(sc / "background" / "bg.jpg"), scene(70))
            slots = [{"name": "slot_1", "x": 20, "y": 20, "w": 90, "h": 120}]
            save_slots_fn(sc / "roi.json", slots, size=SIZE, note="uji")
            (sc / "ground_truth.csv").write_text("image,slot_1\na.jpg,terisi\n",
                                                 encoding="utf-8")
            scenes = discover_scenes(d)
            self.assertEqual(len(scenes), 1)
            s = scenes[0]
            self.assertEqual(s["name"], "sceneA")
            self.assertEqual(len(s["images"]), 1)
            self.assertEqual(len(s["background_paths"]), 1)
            self.assertEqual(s["gt"]["a.jpg"], {"slot_1": "terisi"})


if __name__ == "__main__":
    unittest.main()
