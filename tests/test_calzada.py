import math

import pytest

from calzada import Camera, bbox_footprint, polygon_area

# Celular en el parabrisas de un camion: 1080p, ~70 grados horizontales, 2.2 m de alto, 12 grados hacia abajo.
CAM = Camera(width=1920, height=1080, hfov_deg=70, height_m=2.2, pitch_deg=12)


@pytest.mark.parametrize("x,z", [(0.0, 5.0), (-1.5, 8.0), (2.0, 12.0), (0.3, 4.2)])
def test_ida_y_vuelta_suelo_pixel_suelo(x, z):
    u, v = CAM.ground_to_pixel(x, z)
    gx, gz = CAM.pixel_to_ground(u, v)
    assert gx == pytest.approx(x, abs=1e-9)
    assert gz == pytest.approx(z, abs=1e-9)


def test_por_encima_del_horizonte_no_toca_el_suelo():
    assert CAM.pixel_to_ground(960, CAM.horizon_v - 1) is None
    assert CAM.pixel_to_ground(960, CAM.horizon_v + 1) is not None


def test_mas_abajo_en_la_imagen_es_mas_cerca():
    zs = [CAM.pixel_to_ground(960, v)[1] for v in (700, 800, 900, 1000, 1079)]
    assert zs == sorted(zs, reverse=True)


def test_area_de_un_cuadrado_de_un_metro():
    # Cuadrado de 1 m x 1 m en el suelo entre Z=6 y Z=7 -> su caja en la imagen -> de vuelta al suelo.
    pts = [CAM.ground_to_pixel(x, z) for x, z in ((-0.5, 6), (0.5, 6), (0.5, 7), (-0.5, 7))]
    us, vs = [p[0] for p in pts], [p[1] for p in pts]
    fp = bbox_footprint(CAM, (min(us), min(vs), max(us), max(vs)))
    assert fp is not None
    assert fp.distance_m == pytest.approx(6.0, abs=1e-6)
    assert fp.length_m == pytest.approx(1.0, abs=1e-6)
    # La caja del cuadrado es algo mas ancha arriba que el cuadrado (perspectiva): area >= 1 m2.
    assert 1.0 <= fp.area_m2 < 1.25


def test_lejos_no_se_mide():
    pts = [CAM.ground_to_pixel(x, z) for x, z in ((-0.5, 30), (0.5, 31))]
    box = (pts[0][0], pts[1][1], pts[1][0], pts[0][1])
    assert bbox_footprint(CAM, box, max_distance_m=15) is None


def test_polygon_area():
    assert polygon_area([(0, 0), (2, 0), (2, 3), (0, 3)]) == pytest.approx(6.0)
    assert polygon_area([(0, 0), (1, 0), (0, 1)]) == pytest.approx(0.5)


def test_focal_desde_fov():
    cam = Camera(width=1000, height=500, hfov_deg=90, height_m=1, pitch_deg=0)
    assert cam.f == pytest.approx(500.0)
    assert cam.horizon_v == pytest.approx(250.0)
    assert math.isclose(cam.pixel_to_ground(500, 500)[1], 1 * 500 / 250)
