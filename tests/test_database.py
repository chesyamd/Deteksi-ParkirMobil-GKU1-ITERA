"""Unit test database adegan, build, pencocokan, dan penolakan foto."""

import numpy as np
import pytest

from src.config import Params
from src.database import SceneDatabase
from synthetic_utils import make_query, make_scene, make_texture

SIZE = (400, 300)


@pytest.fixture
def built_db(tmp_path):
    p = Params(resize=SIZE, min_good_matches=8, min_inliers=10)
    base, slots = make_scene(tmp_path / "database", "adegan_a", SIZE,
                             n_refs=4, seed=3)
    db = SceneDatabase(tmp_path / "database", params=p, size=SIZE)
    results = db.build()
    assert results[0]["built"], results[0]
    db = SceneDatabase(tmp_path / "database", params=p, size=SIZE)
    return db, base, slots, p


def test_build_menghasilkan_latar_dan_metadata(built_db):
    db, _base, _slots, _p = built_db
    assert len(db.scenes) == 1
    scene = db.scenes[0]
    assert scene.id == "adegan_a"
    assert len(scene.slots) == 1
    for name in ("canonical.jpg", "background.jpg", "valid_mask.png", "build.json"):
        assert (scene.directory / name).exists(), name
    assert scene.background_bgr.shape[:2] == (SIZE[1], SIZE[0])
    assert float(scene.background_bgr.mean()) > 0


def test_foto_dari_adegan_yang_sama_dikenali(built_db):
    db, base, slots, p = built_db
    query = make_query(base, slots, [False], SIZE, seed=5)
    match = db.match(query, p)
    assert match.recognized, match.reason
    assert match.scene.id == "adegan_a"
    assert match.result.inliers >= 10


def test_foto_di_luar_database_ditolak(built_db):
    db, _base, _slots, _p = built_db
    # ambang inlier diperketat: tekstur asing hanya menghasilkan belasan inlier
    strict = Params(resize=SIZE, min_good_matches=8, min_inliers=25)
    outside = make_texture(SIZE[0], SIZE[1], seed=999)
    match = db.match(outside, strict)
    assert not match.recognized
    assert ("tidak cocok" in match.reason
            or "perlu foto latar baru" in match.reason
            or "pasangan fitur" in match.reason)


def test_database_kosong_memberi_pesan_build(tmp_path):
    db = SceneDatabase(tmp_path / "kosong", autoload=True)
    assert db.scenes == []
    match = db.match(np.zeros((100, 100, 3), np.uint8))
    assert not match.recognized
    assert "database kosong" in match.reason
    assert "src.database build" in match.reason


def test_scene_tidak_siap_build_tidak_dimuat(tmp_path):
    scene = tmp_path / "db" / "belum_build"
    (scene / "refs").mkdir(parents=True)
    (scene / "roi.json").write_text('{"slots": []}', encoding="utf-8")
    (scene / "scene.json").write_text('{"id": "belum_build"}', encoding="utf-8")
    db = SceneDatabase(tmp_path / "db")
    assert db.scenes == []
