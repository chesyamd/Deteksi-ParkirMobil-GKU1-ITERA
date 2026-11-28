"""Aplikasi demo deteksi ketersediaan slot parkir (Streamlit).

Alur: unggah foto -> cocokkan ke database adegan (SIFT+RANSAC) -> tampilkan
adegan terpilih, overlay slot hijau/merah, hitungan slot kosong, dan masker.
Foto yang tidak cocok ke adegan mana pun ditolak dengan pesan jelas.

Jalankan::

    pip install -r requirements.txt -r requirements-app.txt
    python -m src.database build
    streamlit run app/streamlit_app.py
"""

import io
import os
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import TARGET_SIZE, VERSIONS  # noqa: E402
from src.database import SceneDatabase  # noqa: E402
from src.pipeline import detect_photo  # noqa: E402
from src.roi import draw_overlay  # noqa: E402

DEFAULT_DB = os.environ.get("PARKIR_DATABASE", str(ROOT / "data" / "database"))
SANDBOX_DB = ROOT / "sandbox" / "database"
IMAGE_EXT = {"jpg", "jpeg", "png", "bmp", "heic", "heif"}


def decode_uploaded(uploaded):
    """Baca unggahan (JPG/PNG/HEIC) -> BGR, dengan fallback Pillow."""
    data = uploaded.getvalue()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is not None:
        return img
    try:
        from PIL import Image, ImageOps
        try:
            from pillow_heif import register_heif_opener
            register_heif_opener()
        except Exception:
            pass
        with Image.open(io.BytesIO(data)) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Tidak bisa membaca {uploaded.name}: {exc}")


@st.cache_resource(show_spinner=False)
def load_database(root: str):
    return SceneDatabase(root)


def _reset_database_cache():
    load_database.clear()


def main():
    st.set_page_config(page_title="Deteksi Slot Parkir GKU 1", layout="wide")
    st.title("Deteksi Ketersediaan Slot Parkir Mobil - GKU 1 ITERA")
    st.caption("Tugas Besar Pengolahan Citra Digital - Kelompok KASKOS - Kelas RB")

    with st.sidebar:
        st.header("Database adegan")
        default_root = DEFAULT_DB
        if not Path(default_root).is_dir() and SANDBOX_DB.is_dir():
            default_root = str(SANDBOX_DB)
        db_root = st.text_input("Folder database", value=default_root)
        if st.button("Muat ulang database"):
            _reset_database_cache()
            st.rerun()

        if Path(db_root).resolve() == SANDBOX_DB.resolve():
            st.warning("Database SINTETIS (sandbox). Bukan data laporan.")

        st.markdown("---")
        st.header("Parameter")
        versi = st.selectbox("Versi metode", list(VERSIONS))
        ambang = st.slider("Ambang rasio kendaraan", min_value=0.05,
                           max_value=0.90, value=0.30, step=0.05)
        clip = st.slider("CLAHE clipLimit", min_value=0.0, max_value=6.0,
                         value=2.0, step=0.5)
        st.markdown("---")
        st.markdown(
            "**Cara pakai**\n"
            "1. Pastikan database adegan sudah di-build.\n"
            "2. Unggah foto parkiran (JPG/PNG/HEIC).\n"
            "3. Sistem memilih adegan, menyelaraskan latar, lalu menilai slot.\n\n"
            "Foto dari sudut yang belum punya adegan akan **ditolak** dan "
            "diminta menambah foto latar baru."
        )

    params = replace(VERSIONS[versi], ratio_threshold=ambang, clahe_clip=clip)
    db = load_database(db_root)

    if not db.scenes:
        st.error(
            f"Database adegan kosong di `{db_root}`. Isi "
            "`data/database/<id_adegan>/refs/` dengan foto parkiran kosong, "
            "lalu jalankan:\n\n```bash\npython -m src.database build\n```"
        )
        return

    with st.expander(f"Adegan tersedia ({len(db.scenes)})", expanded=False):
        for s in db.scenes:
            note = s.meta.get("note", "")
            st.markdown(f"- **{s.id}** - {len(s.slots)} slot. {note}")
        if db.scenes:
            scene0 = db.scenes[0]
            st.image(cv2.cvtColor(scene0.background_bgr, cv2.COLOR_BGR2RGB),
                     caption=f"Contoh latar adegan: {scene0.id}",
                     use_container_width=True)

    uji_files = st.file_uploader(
        "Foto parkiran yang ingin diperiksa",
        type=sorted(IMAGE_EXT), accept_multiple_files=True,
    )
    if not uji_files:
        st.info("Unggah minimal satu foto untuk mulai memproses.")
        return

    for uploaded in uji_files:
        st.subheader(uploaded.name)
        try:
            img = decode_uploaded(uploaded)
        except ValueError as exc:
            st.error(str(exc))
            continue

        result = detect_photo(img, db, params)
        if not result["recognized"]:
            st.error("Foto tidak dikenali: " + result.get("reason", ""))
            match = result.get("scene_match") or {}
            reg = match.get("registration") or {}
            if reg:
                st.caption(
                    f"Bukti terbaik: inlier={reg.get('inliers')}, "
                    f"good={reg.get('good_matches')}, "
                    f"rasio={reg.get('inlier_ratio')}. "
                    "Tambahkan foto kosong dari sudut ini ke database."
                )
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Foto uji",
                     use_container_width=True)
            st.divider()
            continue

        scene = db.get(result["scene_id"])
        reg = result.get("registration") or {}
        st.success(
            f"Adegan terpilih: **{result['scene_id']}** "
            f"(inlier={reg.get('inliers', '-')}, "
            f"good={reg.get('good_matches', '-')})"
        )
        c1, c2, c3 = st.columns(3)
        c1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Foto uji",
                 use_container_width=True)
        c2.image(cv2.cvtColor(result["mask"], cv2.COLOR_GRAY2RGB),
                 caption=f"Masker biner (ambang Otsu = {result['threshold']:.0f})",
                 use_container_width=True)
        overlay = draw_overlay(img, result["states"])
        c3.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                 caption="Overlay hasil (hijau=kosong, merah=terisi)",
                 use_container_width=True)

        st.metric("Slot tersedia", f"{result['free']} / {result['total']}")
        st.dataframe(
            [{"Slot": s["name"], "Rasio piksel": s["ratio"],
              "Status": s["status"]} for s in result["states"]],
            use_container_width=True, hide_index=True,
        )
        if scene is not None:
            with st.expander("Lihat latar adegan terpilih"):
                st.image(cv2.cvtColor(scene.background_bgr, cv2.COLOR_BGR2RGB),
                         caption=scene.id, use_container_width=True)
        st.divider()


if __name__ == "__main__":
    main()
