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
from src.evaluate import metrics, read_ground_truth
from src.pipeline import detect
from src.preprocess import preprocess, to_gray
from src.roi import load_slots, slot_rect
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


if __name__ == "__main__":
    unittest.main()
