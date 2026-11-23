"""Aplikasi demo deteksi ketersediaan slot parkir (Streamlit).

Jalankan:
    pip install -r requirements.txt -r requirements-app.txt
    streamlit run app/streamlit_app.py
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import TARGET_SIZE, VERSIONS  # noqa: E402
from src.pipeline import detect  # noqa: E402
from src.preprocess import to_gray  # noqa: E402
from src.roi import draw_overlay  # noqa: E402
from src.segment import compute_background  # noqa: E402

SANDBOX = ROOT / "sandbox"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def decode(uploaded):
    data = np.frombuffer(uploaded.getvalue(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Tidak bisa membaca berkas: {uploaded.name}")
    return cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)


def bg_from_uploads(files):
    grays = [to_gray(decode(f)) for f in files]
    return grays[0] if len(grays) == 1 else np.median(np.stack(grays), axis=0).astype(np.uint8)


def bg_from_sandbox():
    paths = sorted(p for p in (SANDBOX / "background").iterdir() if p.suffix.lower() in IMAGE_EXT)
    return compute_background(paths, VERSIONS["usulan"]) if paths else None


def roi_from_sandbox():
    path = SANDBOX / "roi" / "roi.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["slots"]


def main():
    st.set_page_config(page_title="Deteksi Slot Parkir GKU 1", layout="wide")
    st.title("Deteksi Ketersediaan Slot Parkir Mobil — GKU 1 ITERA")
    st.caption("Tugas Besar Pengolahan Citra Digital · Kelompok KASKOS · Kelas RB")

    with st.sidebar:
        st.header("Parameter")
        versi = st.selectbox("Versi metode", list(VERSIONS))
        ambang = st.slider("Ambang rasio kendaraan", min_value=0.05, max_value=0.90,
                           value=0.30, step=0.05)
        clip = st.slider("CLAHE clipLimit", min_value=0.0, max_value=6.0,
                         value=2.0, step=0.5)
        pakai_contoh = st.checkbox("Pakai contoh (sandbox) bila tidak ada unggahan", value=True)
        st.markdown("---")
        st.markdown(
            "**Cara pakai**\n"
            "1. Unggah citra latar (area kosong) — boleh beberapa berkas.\n"
            "2. Unggah `roi.json` (opsional bila pakai contoh).\n"
            "3. Unggah citra uji (area parkir)."
        )

    params = replace(VERSIONS[versi], ratio_threshold=ambang, clahe_clip=clip)

    kol1, kol2 = st.columns(2)
    with kol1:
        bg_files = st.file_uploader("Citra latar (area kosong)", type=["jpg", "jpeg", "png"],
                                    accept_multiple_files=True)
    with kol2:
        roi_file = st.file_uploader("roi.json", type=["json"])

    uji_files = st.file_uploader("Citra uji (area parkir)", type=["jpg", "jpeg", "png"],
                                 accept_multiple_files=True)

    if bg_files:
        background = bg_from_uploads(bg_files)
    elif pakai_contoh:
        background = bg_from_sandbox()
    else:
        background = None

    if roi_file is not None:
        slots = json.loads(roi_file.getvalue().decode("utf-8"))
        slots = slots["slots"] if isinstance(slots, dict) else slots
    elif pakai_contoh:
        slots = roi_from_sandbox()
    else:
        slots = None

    if background is None:
        st.warning("Citra latar belum ada. Unggah 10–20 foto area kosong, atau aktifkan mode contoh.")
        return
    if not slots:
        st.warning("ROI slot belum ada. Unggah roi.json, atau aktifkan mode contoh.")
        return

    if not uji_files:
        if not pakai_contoh:
            st.info("Unggah citra uji untuk mulai memproses.")
            return
        candidates = sorted(p for p in (SANDBOX / "raw").iterdir() if p.suffix.lower() in IMAGE_EXT)
        if not candidates:
            st.info("Unggah citra uji untuk mulai memproses.")
            return
        uji_images = [("contoh: " + candidates[0].name,
                       cv2.resize(cv2.imread(str(candidates[0])), TARGET_SIZE))]
    else:
        uji_images = [(f.name, decode(f)) for f in uji_files]

    for nama, img in uji_images:
        hasil = detect(img, background, slots, params)
        st.subheader(nama)
        c1, c2, c3 = st.columns(3)
        c1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Citra uji", use_container_width=True)
        c2.image(cv2.cvtColor(hasil["mask"], cv2.COLOR_GRAY2RGB),
                 caption=f"Masker biner (ambang Otsu = {hasil['threshold']:.0f})", use_container_width=True)
        overlay = draw_overlay(img, hasil["states"])
        c3.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                 caption="Overlay hasil", use_container_width=True)

        st.metric("Slot tersedia", f"{hasil['free']} / {hasil['total']}")
        st.dataframe(
            [{"Slot": s["name"], "Rasio piksel": s["ratio"], "Status": s["status"]}
             for s in hasil["states"]],
            use_container_width=True, hide_index=True,
        )
        st.divider()


if __name__ == "__main__":
    main()
