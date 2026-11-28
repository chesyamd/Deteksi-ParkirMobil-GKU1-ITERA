"""Konversi foto ponsel HEIC -> JPG ukuran kerja 1280x960 (4:3).

Dua jalur:
1. ``pillow-heif`` (disarankan, lintas platform).
2. Fallback ImageMagick (``magick``/``convert``) bila pillow-heif tidak ada.

Foto 4:3 (mis. 4032x3024) langsung di-resize. Foto non-4:3 di-crop tengah ke
rasio target lebih dulu agar tidak gepeng. Orientasi EXIF diterapkan.

Contoh::

    python tools/convert_heic.py --input data/source --out data/source/jpg
    python tools/convert_heic.py --input IMG_2846.HEIC --out data/source/jpg
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageOps

HEIC_EXT = {".heic", ".heif", ".hif"}
IMAGE_EXT = HEIC_EXT | {".jpg", ".jpeg", ".png", ".webp"}
TARGET = (1280, 960)


def center_crop_to_aspect(img, target=TARGET):
    """Crop tengah ke rasio target (tanpa mengubah skala isi)."""
    tw, th = target
    target_ratio = tw / th
    w, h = img.size
    ratio = w / h
    if abs(ratio - target_ratio) < 1e-6:
        return img
    if ratio > target_ratio:
        new_w = int(round(h * target_ratio))
        x0 = (w - new_w) // 2
        return img.crop((x0, 0, x0 + new_w, h))
    new_h = int(round(w / target_ratio))
    y0 = (h - new_h) // 2
    return img.crop((0, y0, w, y0 + new_h))


def resize_to_target(img, target=TARGET):
    return img.resize(target, Image.Resampling.LANCZOS)


def convert_with_pillow(src: Path, dst: Path, size=TARGET, quality=92):
    """Konversi satu berkas dengan Pillow (+pillow-heif bila HEIC)."""
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except Exception:
        if src.suffix.lower() in HEIC_EXT:
            raise RuntimeError(
                "pillow-heif tidak terpasang dan berkas HEIC tidak bisa dibaca "
                "Pillow biasa. Pasang `pip install pillow-heif` atau ImageMagick."
            )
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im = center_crop_to_aspect(im, size)
        im = resize_to_target(im, size)
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, "JPEG", quality=int(quality), optimize=True)


def _imagemagick_exe():
    for name in ("magick", "magick.exe", "convert", "convert.exe"):
        exe = shutil.which(name)
        if exe:
            return exe
    return None


def convert_with_imagemagick(src: Path, dst: Path, size=TARGET, quality=92):
    exe = _imagemagick_exe()
    if not exe:
        raise RuntimeError("ImageMagick tidak ditemukan di PATH")
    dst.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    cmd = [
        exe, str(src), "-auto-orient",
        "-resize", f"{w}x{h}^",
        "-gravity", "center", "-extent", f"{w}x{h}",
        "-quality", str(int(quality)), str(dst),
    ]
    subprocess.run(cmd, check=True)


def convert_one(src: Path, dst: Path, size=TARGET, quality=92, prefer="auto"):
    """Konversi satu berkas; ``prefer`` = auto|pillow|magick."""
    src, dst = Path(src), Path(dst)
    errors = []
    if prefer in {"auto", "pillow"}:
        try:
            convert_with_pillow(src, dst, size, quality)
            return "pillow"
        except Exception as exc:  # noqa: BLE001 - fallback sengaja
            errors.append(f"pillow: {exc}")
            if prefer == "pillow":
                raise
    if prefer in {"auto", "magick"}:
        try:
            convert_with_imagemagick(src, dst, size, quality)
            return "imagemagick"
        except Exception as exc:  # noqa: BLE001
            errors.append(f"imagemagick: {exc}")
            if prefer == "magick":
                raise
    raise RuntimeError("; ".join(errors))


def collect_inputs(path: Path):
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_EXT)


def main(argv=None):
    ap = argparse.ArgumentParser(description="HEIC -> JPG 1280x960")
    ap.add_argument("--input", required=True, help="berkas atau folder")
    ap.add_argument("--out", required=True, help="folder keluaran JPG")
    ap.add_argument("--size", default=f"{TARGET[0]}x{TARGET[1]}")
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--prefer", choices=["auto", "pillow", "magick"], default="auto")
    args = ap.parse_args(argv)

    w, h = (int(v) for v in args.size.lower().split("x"))
    size = (w, h)
    srcs = collect_inputs(Path(args.input))
    if not srcs:
        raise SystemExit(f"Tidak ada citra di {args.input}")

    out_dir = Path(args.out)
    ok = fail = 0
    for src in srcs:
        dst = out_dir / f"{src.stem}.jpg"
        try:
            via = convert_one(src, dst, size, args.quality, args.prefer)
            ok += 1
            print(f"[OK/{via}] {src.name} -> {dst}")
        except Exception as exc:  # noqa: BLE001
            fail += 1
            print(f"[GAGAL] {src.name}: {exc}", file=sys.stderr)
    print(f"\nSelesai: {ok} berhasil, {fail} gagal. Ukuran {w}x{h}.")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
