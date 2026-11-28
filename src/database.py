"""Database adegan parkir + pembangunan latar + pencocokan adegan.

Struktur satu adegan (satu sudut pandang kamera)::

    data/database/<id_adegan>/
        refs/                 # foto parkiran KOSONG (JPG hasil konversi HEIC)
        scene.json            # metadata adegan (manusia)
        roi.json              # polygon/kotak slot pada bingkai kanonik
        canonical.jpg         # (dibuat build) ref pertama = bingkai kanonik
        background.jpg        # (dibuat build) median refs yang sudah diselaraskan
        valid_mask.png        # (dibuat build) area valid hasil warp
        build.json            # (dibuat build) ringkasan proses build

Perintah::

    python -m src.database build
    python -m src.database list
    python -m src.database match --image data/raw/foto.jpg
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import TARGET_SIZE, Params
from .preprocess import load_image
from .registration import (RegistrationResult, detect_features,
                           match_homography, match_homography_from_features,
                           to_gray, warp_image, warp_valid_mask)
from .roi import load_slots

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
REFS_DIR = "refs"


# --------------------------------------------------------------------------- #
# Struktur data
# --------------------------------------------------------------------------- #
@dataclass
class Scene:
    """Satu adegan (satu sudut pandang) yang sudah di-build."""

    id: str
    directory: Path
    slots: list
    meta: dict
    canonical_bgr: np.ndarray
    background_bgr: np.ndarray
    valid_mask: np.ndarray
    _keypoints: list = field(default=None, repr=False)
    _descriptors: np.ndarray | None = field(default=None, repr=False)
    _feature_n: int = field(default=0, repr=False)

    @property
    def canonical_gray(self):
        return to_gray(self.canonical_bgr)

    @property
    def background_gray(self):
        return to_gray(self.background_bgr)

    def features(self, p: Params):
        """Keypoint/deskriptor SIFT bingkai kanonik (di-cache)."""
        if self._descriptors is None or self._feature_n != p.sift_features:
            self._keypoints, self._descriptors = detect_features(
                self.canonical_gray, p.sift_features
            )
            self._feature_n = p.sift_features
        return self._keypoints, self._descriptors


@dataclass
class SceneMatch:
    """Hasil pemilihan adegan untuk satu foto uji."""

    scene: Scene | None
    result: RegistrationResult | None
    reason: str = ""

    @property
    def recognized(self):
        return self.scene is not None

    def as_dict(self):
        return {
            "recognized": self.recognized,
            "scene": self.scene.id if self.scene else None,
            "reason": self.reason,
            "registration": self.result.as_dict() if self.result else None,
        }


# --------------------------------------------------------------------------- #
# Utilitas kecil
# --------------------------------------------------------------------------- #
def _read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _resize(img, size):
    if size is not None and (img.shape[1], img.shape[0]) != tuple(size):
        return cv2.resize(img, tuple(size), interpolation=cv2.INTER_AREA)
    return img


def _list_images(directory: Path):
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXT)


def _parse_size(text):
    w, h = (int(v) for v in str(text).lower().split("x"))
    return (w, h)


# --------------------------------------------------------------------------- #
# Build satu adegan
# --------------------------------------------------------------------------- #
def build_scene(scene_dir: Path, p: Params, size=TARGET_SIZE):
    """Bangun canonical/background/valid_mask untuk satu adegan.

    Semua refs diselaraskan ke ref pertama (bingkai kanonik), lalu diambil
    median per piksel. Median tahan terhadap satu-dua foto yang masih
    mengandung kendaraan/gerakan.
    """
    scene_dir = Path(scene_dir)
    refs = _list_images(scene_dir / REFS_DIR)
    if not refs:
        return {"scene": scene_dir.name, "built": False,
                "reason": f"tidak ada foto di {scene_dir / REFS_DIR}"}

    canonical = _resize(load_image(refs[0]), size)
    canonical_gray = to_gray(canonical)

    warped, masks, used, skipped, align_stats = [], [], [], [], []
    for i, ref in enumerate(refs):
        img = _resize(load_image(ref), size)
        if i == 0:
            H = np.eye(3, dtype=np.float64)
            stats = RegistrationResult(H, 0, 0, 1.0, 0.0, True, "kanonik")
        else:
            stats = match_homography(to_gray(img), canonical_gray, p)
            if not stats.success:
                skipped.append({"ref": ref.name, "reason": stats.reason,
                                **stats.as_dict()})
                continue
            H = stats.homography

        warped.append(warp_image(img, H, size))
        masks.append(warp_valid_mask(H, size))
        used.append(ref.name)
        align_stats.append({"ref": ref.name, **stats.as_dict()})

    if not warped:
        return {"scene": scene_dir.name, "built": False,
                "reason": "semua ref gagal diselaraskan",
                "skipped": skipped}

    stack = np.stack(warped).astype(np.float32)
    background = np.median(stack, axis=0).astype(np.uint8)

    valid = masks[0]
    for m in masks[1:]:
        valid = cv2.bitwise_and(valid, m)
    background[valid == 0] = 0

    scene_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(scene_dir / "canonical.jpg"), canonical,
                [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    cv2.imwrite(str(scene_dir / "background.jpg"), background,
                [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    cv2.imwrite(str(scene_dir / "valid_mask.png"), valid)

    meta_path = scene_dir / "scene.json"
    if not meta_path.exists():
        _write_json(meta_path, {
            "id": scene_dir.name,
            "name": scene_dir.name,
            "note": "TODO: isi deskripsi adegan, waktu, dan kondisi pengambilan.",
            "captured_at": "",
            "camera": "",
        })

    build = {
        "scene": scene_dir.name,
        "built": True,
        "size": list(size),
        "n_refs": len(refs),
        "n_used": len(used),
        "used": used,
        "skipped": skipped,
        "alignment": align_stats,
        "background": "median refs terselaraskan",
    }
    _write_json(scene_dir / "build.json", build)
    return build


# --------------------------------------------------------------------------- #
# Database
# --------------------------------------------------------------------------- #
class SceneDatabase:
    """Kumpulan adegan yang sudah di-build + pencocokan foto ke adegan."""

    def __init__(self, root="data/database", params: Params | None = None,
                 size=TARGET_SIZE, autoload=True):
        self.root = Path(root)
        self.params = params or Params()
        self.size = tuple(size)
        self.scenes: list[Scene] = []
        if autoload:
            self.load()

    # -- pembacaan ----------------------------------------------------------
    def scene_dirs(self):
        if not self.root.is_dir():
            return []
        return sorted(
            d for d in self.root.iterdir()
            if d.is_dir() and ((d / REFS_DIR).is_dir()
                               or (d / "scene.json").exists()
                               or (d / "roi.json").exists())
        )

    def load(self):
        self.scenes = []
        for d in self.scene_dirs():
            roi = d / "roi.json"
            background = d / "background.jpg"
            canonical = d / "canonical.jpg"
            if not (roi.exists() and background.exists() and canonical.exists()):
                continue
            bg = cv2.imread(str(background))
            can = cv2.imread(str(canonical))
            if bg is None or can is None:
                continue
            valid_path = d / "valid_mask.png"
            valid = (cv2.imread(str(valid_path), cv2.IMREAD_GRAYSCALE)
                     if valid_path.exists()
                     else np.full(bg.shape[:2], 255, np.uint8))
            meta = _read_json(d / "scene.json", {}) or {}
            meta.setdefault("id", d.name)
            self.scenes.append(Scene(
                id=str(meta.get("id") or d.name),
                directory=d,
                slots=load_slots(roi),
                meta=meta,
                canonical_bgr=can,
                background_bgr=bg,
                valid_mask=valid,
            ))
        return self.scenes

    def get(self, scene_id):
        for s in self.scenes:
            if s.id == scene_id:
                return s
        return None

    # -- build --------------------------------------------------------------
    def build(self):
        self.root.mkdir(parents=True, exist_ok=True)
        results = []
        for d in self.scene_dirs():
            results.append(build_scene(d, self.params, self.size))
        self.load()
        return results

    # -- pencocokan ---------------------------------------------------------
    def match(self, image, params: Params | None = None):
        """Cocokkan foto uji ke adegan dengan inlier RANSAC terbanyak.

        Mengembalikan ``SceneMatch``. Bila tidak ada adegan yang melewati
        ambang, ``recognized=False`` dengan alasan yang bisa ditampilkan ke
        pengguna ("perlu foto latar baru").
        """
        p = params or self.params
        if not self.scenes:
            return SceneMatch(None, None,
                              f"database kosong ({self.root}); jalankan "
                              "`python -m src.database build`")
        img = _resize(image, self.size)
        gray = to_gray(img)
        kp_test, des_test = detect_features(gray, p.sift_features)

        best_scene, best_result = None, None
        for scene in self.scenes:
            kp_ref, des_ref = scene.features(p)
            result = match_homography_from_features(
                kp_ref, des_ref, kp_test, des_test, p
            )
            if best_result is None or (
                result.inliers, result.good_matches
            ) > (best_result.inliers, best_result.good_matches):
                best_scene, best_result = scene, result

        if best_result is None or not best_result.success:
            detail = best_result.reason if best_result else "tidak ada adegan"
            return SceneMatch(
                None, best_result,
                f"foto tidak cocok ke adegan mana pun ({detail}); "
                "perlu foto latar baru untuk sudut ini",
            )
        return SceneMatch(best_scene, best_result, "")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _cmd_build(args):
    db = SceneDatabase(args.database, size=_parse_size(args.size), autoload=False)
    results = db.build()
    if not results:
        print(f"Tidak ada adegan di {db.root}. Buat folder "
              f"<id_adegan>/refs/ lalu isi foto kosong.")
        return 1
    ok = 0
    for r in results:
        if r.get("built"):
            ok += 1
            print(f"[OK] {r['scene']}: {r['n_used']}/{r['n_refs']} refs, "
                  f"latar median -> {db.root / r['scene'] / 'background.jpg'}")
            for s in r.get("skipped", []):
                print(f"     lewati {s['ref']}: {s['reason']}")
        else:
            print(f"[--] {r['scene']}: {r['reason']}")
    print(f"\n{ok}/{len(results)} adegan berhasil di-build.")
    return 0 if ok else 1


def _cmd_list(args):
    db = SceneDatabase(args.database)
    if not db.scenes:
        print(f"Tidak ada adegan siap pakai di {db.root}.")
        return 1
    for s in db.scenes:
        note = str(s.meta.get("note", "")).strip().replace("\n", " ")
        if len(note) > 70:
            note = note[:67] + "..."
        print(f"- {s.id}: {len(s.slots)} slot | {note} | {s.directory}")
    return 0


def _cmd_match(args):
    db = SceneDatabase(args.database)
    img = load_image(args.image)
    match = db.match(img)
    print(json.dumps(match.as_dict(), indent=2, ensure_ascii=False))
    return 0 if match.recognized else 2


def main(argv=None):
    ap = argparse.ArgumentParser(description="Database adegan parkir GKU 1")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="bangun latar median tiap adegan")
    b.add_argument("--database", default="data/database")
    b.add_argument("--size", default=f"{TARGET_SIZE[0]}x{TARGET_SIZE[1]}")
    b.set_defaults(func=_cmd_build)

    l = sub.add_parser("list", help="daftar adegan siap pakai")
    l.add_argument("--database", default="data/database")
    l.set_defaults(func=_cmd_list)

    m = sub.add_parser("match", help="cocokkan satu foto ke adegan")
    m.add_argument("--database", default="data/database")
    m.add_argument("--image", required=True)
    m.set_defaults(func=_cmd_match)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
