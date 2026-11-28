"""Unit test klasifikasi slot dari rasio piksel kendaraan."""

import numpy as np

from src.classify import classify_slot, occupancy_ratio
from src.config import Params

SIZE = (200, 160)


def _slot(x=20, y=20, w=100, h=100):
    return {"name": "slot_uji", "x": x, "y": y, "w": w, "h": h}


def test_rasio_dan_status_terisi():
    mask = np.zeros((SIZE[1], SIZE[0]), np.uint8)
    mask[20:120, 20:120] = 255  # seluruh slot
    slot = _slot()
    p = Params(resize=None, ratio_threshold=0.30)
    assert occupancy_ratio(mask, slot) == 1.0
    assert classify_slot(mask, slot, p)["status"] == "terisi"


def test_rasio_dan_status_kosong():
    mask = np.zeros((SIZE[1], SIZE[0]), np.uint8)
    mask[20:40, 20:40] = 255  # 400 / 10000 = 0.04
    slot = _slot()
    p = Params(resize=None, ratio_threshold=0.30)
    state = classify_slot(mask, slot, p)
    assert state["status"] == "kosong"
    assert state["ratio"] == 0.04


def test_ambang_klasifikasi_dapat_diubah():
    mask = np.zeros((SIZE[1], SIZE[0]), np.uint8)
    mask[20:70, 20:120] = 255  # 5000 / 10000 = 0.5
    slot = _slot()
    assert classify_slot(mask, slot, Params(resize=None,
                                            ratio_threshold=0.4))["status"] == "terisi"
    assert classify_slot(mask, slot, Params(resize=None,
                                            ratio_threshold=0.6))["status"] == "kosong"


def test_poligon_hanya_menghitung_piksel_dalam():
    mask = np.zeros((SIZE[1], SIZE[0]), np.uint8)
    mask[20:120, 20:120] = 255
    poly = {"name": "p", "points": [[20, 20], [120, 20], [120, 120], [20, 120]]}
    assert occupancy_ratio(mask, poly) > 0.95
    luar = {"name": "luar", "points": [[130, 20], [190, 20], [190, 120], [130, 120]]}
    assert occupancy_ratio(mask, luar) == 0.0
