import pytest

from preview import codigo_det


@pytest.mark.parametrize("nombre,codigo", [
    ("D40", "D40"), ("D40 pothole", "D40"), ("d40_pothole", "D40"), ("D00", "D00"),
    ("pothole", "D40"), ("Pothole", "D40"), ("bache", "D40"),
    ("Pothole (D40)", "D40"), ("Longitudinal Crack (D00)", "D00"), ("Transverse Crack (D10)", "D10"),
    ("Alligator Crack (D20)", "D20"), ("crack", None),
    ("Repair", None), ("manhole", None), ("", None),
])
def test_codigo_det(nombre, codigo):
    assert codigo_det(nombre) == codigo
