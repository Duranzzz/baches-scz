"""Arma el artículo del 2do parcial (formato Revista 3i) sobre docs/plantilla_articulo_2do_parcial.docx.
Todas las cifras salen de informes/ (entrenamiento_amd.md, eval_*.json, corridas/ y videos_scz_20261122/).
Antes: python docs/figuras/generar_figuras.py (las fotografías y el esquema; los gráficos con datos se arman aquí,
editables, con python-pptx: pip install python-docx python-pptx)
    python docs/armar_articulo.py
Deja docs/articulo_2do_parcial.docx."""
import copy
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from docx import Document
from docx.opc.constants import CONTENT_TYPE as CT
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.opc.packuri import PackURI
from docx.opc.part import Part
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm, Pt
from docx.text.paragraph import Paragraph
from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData, XyChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_MARKER_STYLE
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.util import Cm as PCm
from pptx.util import Pt as PPt

RAIZ = Path(__file__).resolve().parents[1]
DOCS = RAIZ / "docs"
FIG = DOCS / "figuras"
d = Document(DOCS / "plantilla_articulo_2do_parcial.docx")
CUERPO = d._body
COAUTORA = "Dana Gabriela Quisberth Sursi"
ORCID_COAUTORA = "https://orcid.org/0009-0005-4240-7452"
CARRERA_COAUTORA = "Ingeniería Mecatrónica"
CORREO_COAUTORA = "dquisberth@ucb.edu.bo"


def buscar(prefijo):
    for p in d.paragraphs:
        if p.text.strip().startswith(prefijo):
            return p
    raise SystemExit(f"no encontré: {prefijo!r}")


def poner(p, texto, desde=0):
    runs = p.runs
    if not runs:
        p.add_run(texto)
    else:
        runs[desde].text = texto
        for r in runs[desde + 1:]:
            if not r._r.findall(qn("w:drawing")):
                r._r.getparent().remove(r._r)
    for h in p._element.findall(qn("w:hyperlink")):
        p._element.remove(h)


def borrar_entre(a, b):
    e = a._element.getnext()
    while e is not None and e is not b._element:
        sig = e.getnext()
        e.getparent().remove(e)
        e = sig


def coma(x, dec=3):
    return f"{x:.{dec}f}".replace(".", ",")


def miles(n):
    return f"{n:,}".replace(",", ".")


# ---------------------------------------------------------------- prototipos de la plantilla
P_CUERPO = copy.deepcopy(buscar("[Introducción:")._element)
P_SUB = copy.deepcopy(buscar("[Subtítulo de resultados")._element)
P_CAPTION = copy.deepcopy(buscar("Tabla 1")._element)
P_TABLA = copy.deepcopy(d.tables[0]._tbl)
P_NOTA = copy.deepcopy(buscar("Nota. [Qué muestra la tabla")._element)
P_FIG_ETIQ = copy.deepcopy(buscar("Figura 1.")._element)
P_FIG_TIT = copy.deepcopy(buscar("[Título de la figura")._element)
P_FIG_IMG = copy.deepcopy(buscar("[Figura: imagen")._element)
P_NOTA_FIG = copy.deepcopy(buscar("Nota. [Fuente y explicación")._element)
P_REF = copy.deepcopy(buscar("Arya, D., Maeda, H., Ghosh")._element)


class Cursor:
    """Inserta elementos en orden después de un ancla; numera tablas y figuras solo."""
    n_tabla = 0
    n_figura = 0

    def __init__(self, ancla):
        self.e = ancla._element if hasattr(ancla, "_element") else ancla

    def _poner(self, elem):
        self.e.addnext(elem)
        self.e = elem
        return Paragraph(elem, CUERPO) if elem.tag == qn("w:p") else elem

    def parrafo(self, texto, proto=P_CUERPO):
        """Párrafo con el formato del prototipo; "_{x}" escribe x como subíndice (AP_{val})."""
        p = self._poner(copy.deepcopy(proto))
        partes = re.split(r"_\{([^}]*)\}", texto)
        poner(p, partes[0])
        base = p.runs[0]._r.rPr
        for i, parte in enumerate(partes[1:], 1):
            if parte:
                r = p.add_run(parte)
                if base is not None:
                    r._r.insert(0, copy.deepcopy(base))
                r.font.subscript = i % 2 == 1
        return p

    def sub(self, texto):
        return self.parrafo(texto, P_SUB)

    def tabla(self, titulo, encabezados, filas, anchos, nota, tam=9.5):
        Cursor.n_tabla += 1
        cap = self._poner(copy.deepcopy(P_CAPTION))   # "Tabla N" en negrita, salto, título en cursiva
        cap.runs[0].text = f"Tabla {Cursor.n_tabla}"
        cap.runs[2].text = titulo
        cap.paragraph_format.keep_with_next = True
        t = copy.deepcopy(P_TABLA)
        ncol = len(encabezados)
        grid = t.find(qn("w:tblGrid"))
        cols = grid.findall(qn("w:gridCol"))
        for c in cols[ncol:]:
            grid.remove(c)
        while len(grid.findall(qn("w:gridCol"))) < ncol:
            grid.append(copy.deepcopy(cols[-1]))
        for c, w in zip(grid.findall(qn("w:gridCol")), anchos):
            c.set(qn("w:w"), str(round(w * 567)))
        tblw = t.find(qn("w:tblPr")).find(qn("w:tblW"))
        if tblw is not None:
            tblw.set(qn("w:w"), str(round(sum(anchos) * 567)))
            tblw.set(qn("w:type"), "dxa")
        filas_xml = t.findall(qn("w:tr"))
        cab, medio, ultima = filas_xml[0], filas_xml[1], filas_xml[-1]
        for fx in filas_xml[1:]:
            t.remove(fx)
        nuevas = [copy.deepcopy(ultima if i == len(filas) - 1 else medio) for i in range(len(filas))]
        for fx in nuevas:
            t.append(fx)
        for fx, valores in zip([cab] + nuevas, [encabezados] + filas):
            for tc in fx.findall(qn("w:tc"))[ncol:]:
                fx.remove(tc)
            while len(fx.findall(qn("w:tc"))) < ncol:
                fx.append(copy.deepcopy(fx.findall(qn("w:tc"))[-1]))
            for j, (tc, w, v) in enumerate(zip(fx.findall(qn("w:tc")), anchos, valores)):
                tcw = tc.find(qn("w:tcPr")).find(qn("w:tcW"))
                tcw.set(qn("w:w"), str(round(w * 567)))
                tcw.set(qn("w:type"), "dxa")
                p = Paragraph(tc.find(qn("w:p")), CUERPO)
                poner(p, v)
                p.runs[0].font.size = Pt(tam)
                p.alignment = 0 if j == 0 else 1
        self._poner(t)
        poner(self._poner(copy.deepcopy(P_NOTA)), "Nota. " + nota)
        return Cursor.n_tabla

    def figura(self, titulo, archivo, ancho_cm, nota):
        Cursor.n_figura += 1
        et = self._poner(copy.deepcopy(P_FIG_ETIQ))
        poner(et, f"Figura {Cursor.n_figura}.")
        tit = self._poner(copy.deepcopy(P_FIG_TIT))
        poner(tit, titulo)
        img = self._poner(copy.deepcopy(P_FIG_IMG))
        for r in list(img.runs):
            r._r.getparent().remove(r._r)
        img.add_run().add_picture(str(FIG / archivo), width=Cm(ancho_cm))
        img.alignment = 1
        for q in (et, tit, img):   # etiqueta, título, imagen y nota en la misma página
            q.paragraph_format.keep_with_next = True
        poner(self._poner(copy.deepcopy(P_NOTA_FIG)), "Nota. " + nota)
        return Cursor.n_figura

    def grafico(self, titulo, graficos, nota):
        """Como figura(), con gráficos editables de Office: graficos = [(gráfico de pptx, ancho_cm, alto_cm), ...]."""
        Cursor.n_figura += 1
        et = self._poner(copy.deepcopy(P_FIG_ETIQ))
        poner(et, f"Figura {Cursor.n_figura}.")
        tit = self._poner(copy.deepcopy(P_FIG_TIT))
        poner(tit, titulo)
        img = self._poner(copy.deepcopy(P_FIG_IMG))
        for r in list(img.runs):
            r._r.getparent().remove(r._r)
        for i, (gf, ancho, alto) in enumerate(graficos):
            if i:
                img.add_run(" ")
            img._p.append(incrustar(gf, ancho, alto))
        img.alignment = 1
        for q in (et, tit, img):
            q.paragraph_format.keep_with_next = True
        poner(self._poner(copy.deepcopy(P_NOTA_FIG)), "Nota. " + nota)
        return Cursor.n_figura

    n_ecuacion = 0

    def ecuacion(self, *hijos):
        """Ecuación del editor de Word (OMML) en su propio párrafo, centrada y con el número entre corchetes al
        margen derecho: es la numeración nativa de Word (escribir "#[1]" al final de la ecuación)."""
        Cursor.n_ecuacion += 1
        p = self._poner(copy.deepcopy(P_CUERPO))
        for hijo in list(p._p):
            if hijo.tag != qn("w:pPr"):
                p._p.remove(hijo)
        p.paragraph_format.first_line_indent = p.paragraph_format.left_indent = Cm(0)
        numero = mpar(mt(str(Cursor.n_ecuacion)), abre="[", cierra="]")
        arreglo = _m("eqArr", _m("eqArrPr", _m("maxDist", val="1")), _m("e", *hijos, mt("#"), numero))
        p._p.append(_m("oMathPara", _m("oMath", arreglo)))
        return Cursor.n_ecuacion


# ---------------------------------------------------------------- ecuaciones (OMML)
def _m(etiqueta, *hijos, **atributos):
    e = etree.Element(qn(f"m:{etiqueta}"))
    for k, v in atributos.items():
        e.set(qn(f"m:{k}"), v)
    for h in hijos:
        e.append(h)
    return e


def mt(texto, recto=False, texto_normal=False):
    """Texto de la ecuación: variables en cursiva; recto=True para nombres (IoU, máx); texto_normal=True para
    números con coma decimal, que el editor separaría como si fuera una lista."""
    r = _m("r")
    if texto_normal:
        r.append(_m("rPr", _m("nor")))
    elif recto:
        r.append(_m("rPr", _m("sty", val="p")))
    t = _m("t")
    t.text = texto
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    r.append(t)
    return r


def msub(base, sub):
    return _m("sSub", _m("e", *base), _m("sub", *sub))


def msup(base, sup):
    return _m("sSup", _m("e", *base), _m("sup", *sup))


def mfrac(num, den):
    return _m("f", _m("num", *num), _m("den", *den))


def mpar(*hijos, abre="(", cierra=")"):
    return _m("d", _m("dPr", _m("begChr", val=abre), _m("endChr", val=cierra)), _m("e", *hijos))


def mfunc(nombre, limite, argumento):
    """Función con límite debajo, como máx o argmáx."""
    return _m("func", _m("fName", _m("limLow", _m("e", mt(nombre, recto=True)), _m("lim", *limite))),
              _m("e", *argumento))


def msuma(sub, cuerpo):
    return _m("nary", _m("naryPr", _m("chr", val="∑"), _m("limLoc", val="undOvr"), _m("supHide", val="1")),
              _m("sub", *sub), _m("sup"), _m("e", *cuerpo))


# ---------------------------------------------------------------- gráficos editables (python-pptx)
# La plantilla pide gráficos de Office editables ("copie de Excel la figura y pegue con formato de origen"): se
# arman con python-pptx, con su hoja de Excel incrustada, y se pasan al .docx.
# Word escribe los números de un gráfico con el separador decimal de Windows (punto en una PC en inglés). Para que
# la coma salga en cualquier computadora, los valores van multiplicados por 100 y el formato pone la coma como texto:
# 61 con el formato 0\,00 se ve 0,61.
X100, COMA2 = 100, "0\\,00"
COL = dict(azul="2A78D6", naranja="EB6834", agua="1BAF7A", gris="8A8A8A", tinta="222222", rejilla="E2E2E0",
           nota="555555")
rgb = RGBColor.from_string
_PRS = Presentation()
_n_graf = 0


def grafico_nuevo(tipo, datos, ancho, alto, leyenda=XL_LEGEND_POSITION.BOTTOM, titulo=None):
    gf = _PRS.slides.add_slide(_PRS.slide_layouts[6]).shapes.add_chart(tipo, 0, 0, PCm(ancho), PCm(alto), datos)
    ch = gf.chart
    ch.font.name, ch.font.size = "Arial", PPt(8.5)
    ch.font.color.rgb = rgb(COL["tinta"])
    ch.has_title = titulo is not None
    if titulo:
        tf = ch.chart_title.text_frame
        tf.text = titulo
        f = tf.paragraphs[0].runs[0].font
        f.size, f.bold = PPt(9), True
    ch.has_legend = leyenda is not None
    if leyenda is not None:
        ch.legend.position, ch.legend.include_in_layout = leyenda, False
        ch.legend.font.size = PPt(8)
    return gf, ch


def ejes(ch, x_titulo, y_titulo, y, x=None, fmt_x=None):
    """y y x: (mínimo, máximo, paso), en las unidades de los datos (×100). Rejilla horizontal suave, sin línea del
    eje vertical."""
    va, ca = ch.value_axis, ch.category_axis
    va.minimum_scale, va.maximum_scale = y[0], y[1]
    va.major_unit = y[2]
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = rgb(COL["rejilla"])
    va.major_gridlines.format.line.width = PPt(0.6)
    va.format.line.fill.background()
    va.tick_labels.number_format, va.tick_labels.number_format_is_linked = COMA2, False
    ca.format.line.color.rgb = rgb(COL["gris"])
    ca.has_major_gridlines = False
    if x:
        ca.minimum_scale, ca.maximum_scale = x[0], x[1]
        ca.major_unit = x[2]
    if fmt_x:
        ca.tick_labels.number_format, ca.tick_labels.number_format_is_linked = fmt_x, False
    for eje, texto in ((ca, x_titulo), (va, y_titulo)):
        eje.has_title = texto is not None
        if texto is None:
            continue
        tf = eje.axis_title.text_frame
        tf.text = texto
        f = tf.paragraphs[0].runs[0].font
        f.size, f.bold = PPt(8.5), False


def linea(s, color, ancho=1.6, guion=None, marca=None, tam=5):
    s.smooth = False
    ln = s.format.line
    ln.color.rgb, ln.width = rgb(color), PPt(ancho)
    if guion is not None:
        ln.dash_style = guion
    if marca is None:
        s.marker.style = XL_MARKER_STYLE.NONE
    else:
        s.marker.style, s.marker.size = marca, tam
        s.marker.format.fill.solid()
        s.marker.format.fill.fore_color.rgb = rgb(color)
        s.marker.format.line.color.rgb = rgb(color)


def rotular(s, i, texto, pos=XL_LABEL_POSITION.ABOVE, color=None, tam=6):
    """Rótulo en un punto y, si se da el color, una marca circular llena en ese punto."""
    pt = s.points[i]
    dl = pt.data_label
    dl.has_text_frame = True
    dl.text_frame.text = texto
    dl.position = pos
    f = dl.text_frame.paragraphs[0].runs[0].font
    f.size = PPt(7)
    f.color.rgb = rgb(COL["nota"])
    if color:
        pt.marker.style, pt.marker.size = XL_MARKER_STYLE.CIRCLE, tam
        pt.marker.format.fill.solid()
        pt.marker.format.fill.fore_color.rgb = rgb(color)
        pt.marker.format.line.color.rgb = rgb(color)


def incrustar(gf, ancho, alto):
    """Pasa el gráfico (y su hoja de Excel) al .docx y devuelve el w:r con el dibujo."""
    global _n_graf
    _n_graf += 1
    pkg, cp = d.part.package, gf.chart_part
    graf = Part(PackURI(f"/word/charts/grafico{_n_graf}.xml"), CT.DML_CHART, b"", pkg)
    hoja = Part(PackURI(f"/word/embeddings/grafico{_n_graf}.xlsx"), CT.SML_SHEET, cp.chart_workbook.xlsx_part.blob, pkg)
    cs = copy.deepcopy(cp._element)
    cs.find(qn("c:externalData")).set(qn("r:id"), graf.relate_to(hoja, RT.PACKAGE))
    # Sin el marco redondeado que Word dibuja por defecto alrededor del gráfico
    cs.insert(0, parse_xml(f'<c:roundedCorners {nsdecls("c")} val="0"/>'))
    cs.find(qn("c:chart")).addnext(parse_xml(f'<c:spPr {nsdecls("c", "a")}><a:noFill/><a:ln><a:noFill/></a:ln></c:spPr>'))
    graf._blob = etree.tostring(cs, xml_declaration=True, encoding="UTF-8", standalone=True)
    rid = d.part.relate_to(graf, RT.CHART)
    n = d.part.next_id
    return parse_xml(
        f'<w:r {nsdecls("w", "wp", "a", "c", "r")}><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{round(ancho * 360000)}" cy="{round(alto * 360000)}"/>'
        f'<wp:effectExtent l="0" t="0" r="0" b="0"/><wp:docPr id="{n}" name="Gráfico {n}"/><wp:cNvGraphicFramePr/>'
        '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/chart">'
        f'<c:chart r:id="{rid}"/></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>')


# ---------------------------------------------------------------- datos de los informes
INF = RAIZ / "informes"


def ap(nombre, split, lado, dataset, clase="D40"):
    r = json.loads((INF / f"eval_{nombre}_{split}_{lado}.json").read_text(encoding="utf-8"))
    return next(f["AP50_det"] for f in r["filas"] if f["dataset"] == dataset and f["clase"] == clase)


videos = {}
for p in sorted((INF / "videos_scz_20261122").glob("*/resumen.json")):
    r = json.loads(p.read_text(encoding="utf-8"))
    videos[r["video"].replace(".mp4", "")] = r
ORDEN = [("scz_01_lateral_auto_hormigon", "scz_01 (lateral, auto)"),
         ("scz_02_frontal_avenida", "scz_02 (frontal)"),
         ("scz_03_frontal_doble_via", "scz_03 (frontal)"),
         ("scz_04_lateral_micro_losas", "scz_04 (lateral, micro)"),
         ("scz_05_lateral_micro_hormigon", "scz_05 (lateral, micro)")]
MODELOS_V = ["yolo12s_rdd2022", "rdd_yolo26s", "wise4_a25", "a1_y26s", "a2_y26s_hrp4k"]
tot = {k: [0, 0, 0, 0] for k in MODELOS_V}   # cuadros con bache, eventos, cuadros con grieta, eventos de grieta
tot_cuadros = 0
filas_v = []
for clave, nombre in ORDEN:
    r = videos[clave]
    tot_cuadros += r["cuadros"]
    fila = [nombre, miles(r["cuadros"])]
    for k in MODELOS_V:
        m = r["modelos"][k]
        for i, campo in enumerate(("cuadros_con_bache", "eventos_bache", "cuadros_con_grieta", "eventos_grieta")):
            tot[k][i] += m[campo]
        fila.append(f"{coma(100 * m['cuadros_con_bache'] / r['cuadros'], 1)} %; {m['eventos_bache']}")
    filas_v.append(fila)
filas_v.append(["Total", miles(tot_cuadros)] + [
    f"{coma(100 * tot[k][0] / tot_cuadros, 1)} %; {tot[k][1]}" for k in MODELOS_V])
pct = {k: coma(100 * tot[k][0] / tot_cuadros, 1) for k in MODELOS_V}
pctg = {k: coma(100 * tot[k][2] / tot_cuadros, 1) for k in MODELOS_V}
v3 = videos["scz_03_frontal_doble_via"]["modelos"]
# Revisión a ciegas de los eventos de bache (herramientas/revisar_eventos.py)
rev = json.loads((INF / "videos_scz_20261122" / "revision" / "resultados.json").read_text(encoding="utf-8"))
REV = {f["modelo"]: f for f in rev["modelos"]}
REV_ORDEN = [("YOLO12s (referencia)", "Referencia"), ("B0 (pesos iniciales)", "Iniciales"),
             ("Baches SCZ v1 (final)", "Final"), ("a1 (entrenado)", "Entrenado"), ("a2 (con HRP4K)", "Con HRP4K")]
rfin, ra1, ra2, rref = (REV[k] for k in ("Baches SCZ v1 (final)", "a1 (entrenado)", "a2 (con HRP4K)",
                                         "YOLO12s (referencia)"))
n_rev = sum(f["eventos"] for f in rev["modelos"])
p100 = lambda x: f"{round(100 * x)} %"
for cota in (lambda f: f["B"] / f["eventos"], lambda f: (f["B"] + f["D"]) / f["eventos"]):
    assert max(rev["modelos"], key=cota) is rfin, "el modelo final ya no es el primero con alguna cota de los dudosos"
# Cuadros de Santa Cruz etiquetados con cajas (entrenamiento/scz_cuadros.py) y ajuste v2 (informes/ajuste_scz.md)
SCZ = INF / "scz_cuadros"
leer = lambda nombre: json.loads((SCZ / f"{nombre}.json").read_text(encoding="utf-8"))
scz50, scz30 = leer("eval_todos_640"), leer("eval_todos_640_iou30")
scz50p, scz30p = leer("eval_prueba_640"), leer("eval_prueba_640_iou30")
AJ = INF / "ajuste_scz.json"
ajuste = json.loads(AJ.read_text(encoding="utf-8")) if AJ.exists() else None
SCZ_ORDEN = [("referencia", "Referencia"), ("B0", "Iniciales"), ("v1", "Final"), ("a1", "Entrenado"), ("a2", "Con HRP4K")]
c3 = lambda x: coma(x, 3)
assert max(scz50["ap50"], key=scz50["ap50"].get) == "a2", "a2 ya no es el que más encuentra en los cuadros"
a2_01 = videos["scz_01_lateral_auto_hormigon"]["modelos"]["a2_y26s_hrp4k"]

# ---------------------------------------------------------------- título, autores y resúmenes
poner(buscar("[Título del manuscrito"), "Cartografía de baches y grietas con inteligencia artificial en el teléfono móvil")
poner(buscar("[Título traducido al inglés"),
      "Cartography of potholes and cracks with artificial intelligence on the mobile phone")
poner(buscar("[Título em português"), "Cartografia de buracos e fissuras com inteligência artificial no telefone celular")
recibido = buscar("Recibido:")
poner(recibido, "28-11-2026", desde=1)   # fecha de envío, con el formato de la plantilla (31-10-2025)
for color in recibido.runs[1]._r.findall(".//" + qn("w:color")):   # la plantilla marca en rojo lo que falta llenar
    color.getparent().remove(color)

# Segunda autora: el mismo bloque que el primero (nombre, carrera, departamento, universidad, ORCID y correo)
nombre1 = buscar("Leonardo Duran Cuenca")
bloque = [nombre1._element]
e = nombre1._element.getnext()
while len(bloque) < 6:
    bloque.append(e)
    e = e.getnext()
ancla = bloque[-1]
for i, original in enumerate(bloque):
    nuevo = copy.deepcopy(original)
    ancla.addnext(nuevo)
    ancla = nuevo
    p = Paragraph(nuevo, CUERPO)
    if i == 0:
        poner(p, COAUTORA)
        p.paragraph_format.space_before = Pt(10)
    elif i == 1:
        poner(p, CARRERA_COAUTORA)
    elif i == 4:   # ORCID: el mismo ícono y un enlace nuevo
        h = nuevo.find(qn("w:hyperlink"))
        h.set(qn("r:id"), d.part.relate_to(ORCID_COAUTORA, RT.HYPERLINK, is_external=True))
        ts = h.findall(".//" + qn("w:t"))
        ts[0].text = ORCID_COAUTORA
        for extra in ts[1:]:
            extra.text = ""
    elif i == 5:
        poner(p, CORREO_COAUTORA)

# Los resúmenes dicen que ninguna variante v2 cumplió la regla (informes/ajuste_scz.md)
assert ajuste and not ajuste["ganador"], "los resúmenes suponen que ninguna v2 cumplió la regla"
v1d = ajuste["v1"]
mejor = max(ajuste["candidatos"].values(), key=lambda d: d["scz_prueba50"])
n_aj = scz50["cuadros"] - scz50p["cuadros"]
assert scz50["ap50"]["a1"] > scz50["ap50"]["v1"], "el modelo sin promediar ya no halla más baches en los cuadros"
RESUMEN = (
    "El objetivo es implementar y evaluar una aplicación web que detecta baches y grietas con inteligencia artificial "
    "en el teléfono móvil y los registra en la cartografía de Santa Cruz de la Sierra. Se recopilaron 12.204 "
    "fotografías de seis conjuntos públicos y, con 6.507 depuradas de entrenamiento, se entrenó en una GPU AMD MI300X "
    "un detector YOLO26s; el modelo final promedia sus pesos con los iniciales según una regla prefijada. Se "
    f"evalúa con una prueba congelada de 2.721 fotografías, {n_rev} marcas revisadas a ciegas en cinco videos y "
    f"{scz50['cuadros']} cuadros etiquetados de la ciudad. El modelo final obtiene un AP50 de 0,276 en baches, frente a "
    "0,128 del detector anterior; sin promediar llega a 0,496, pero baja de 0,048 a 0,017 en un conjunto no visto. En "
    f"los videos, el {p100(rfin['precision_bache'])} de las marcas del modelo final son baches, frente al "
    f"{p100(ra1['precision_bache'])} del modelo sin promediar, que en los cuadros halla más baches. Ajustar con {n_aj} "
    f"cuadros locales sube el AP50 en otros {scz50p['cuadros']} de {c3(v1d['scz_prueba50'])} a "
    f"{c3(mejor['scz_prueba50'])}, pero resta robustez: no se adopta. La aplicación procesa de 8 a 12 cuadros por "
    "segundo en una laptop. Promediar pesos da robustez; la ciudad requiere datos propios.")
ABSTRACT = (
    "The objective is to implement and evaluate a web application that detects potholes and cracks with artificial "
    "intelligence on the mobile phone and records them on the cartography of Santa Cruz de la Sierra. We collected "
    "12,204 photographs from six public datasets and, with 6,507 curated training photographs, trained a YOLO26s "
    "detector on an AMD MI300X GPU; the final model averages its weights with the initial ones by a pre-registered "
    f"rule. The evaluation uses a frozen test set of 2,721 photographs, {n_rev} detections reviewed blindly in "
    f"five videos and {scz50['cuadros']} labeled frames of the city. The final model obtains an AP50 of 0.276 for "
    "potholes, compared with 0.128 for the previous detector; without averaging it reaches 0.496 but drops from 0.048 "
    f"to 0.017 on an unseen dataset. In the videos, {round(100 * rfin['precision_bache'])}% of the final model's "
    f"detections are potholes, compared with {round(100 * ra1['precision_bache'])}% for the model without averaging, "
    f"which finds more potholes in the labeled frames. Fine-tuning with {n_aj} local frames raises the AP50 on another "
    f"{scz50p['cuadros']} from {v1d['scz_prueba50']:.3f} to {mejor['scz_prueba50']:.3f} but reduces robustness and is "
    "not adopted. The application processes 8 to 12 frames per second on a laptop. Weight averaging provides "
    "robustness; the city requires its own data.")
RESUMO = (
    "O objetivo é implementar e avaliar uma aplicação web que detecta buracos e fissuras com inteligência artificial "
    "no telefone celular e os registra na cartografia de Santa Cruz de la Sierra. Foram reunidas 12.204 fotografias de "
    "seis conjuntos públicos e, com 6.507 depuradas de treinamento, foi treinado em uma GPU AMD MI300X um detector "
    "YOLO26s; o modelo final faz a média de seus pesos com os iniciais segundo uma regra prefixada. A avaliação "
    f"usa um teste congelado de 2.721 fotografias, {n_rev} marcações revisadas às cegas em cinco vídeos e "
    f"{scz50['cuadros']} quadros rotulados da cidade. O modelo final obtém AP50 de 0,276 em buracos, frente a 0,128 do "
    "detector anterior; sem média chega a 0,496, mas cai de 0,048 para 0,017 em um conjunto não visto. Nos vídeos, "
    f"{p100(rfin['precision_bache'])} das marcações do modelo final são buracos, frente a "
    f"{p100(ra1['precision_bache'])} do modelo sem média, que encontra mais buracos nos quadros rotulados. Ajustar com "
    f"{n_aj} quadros locais eleva o AP50 em outros {scz50p['cuadros']} de {c3(v1d['scz_prueba50'])} para "
    f"{c3(mejor['scz_prueba50'])}, mas reduz a robustez: não é adotado. A aplicação processa de 8 a 12 quadros por "
    "segundo em um laptop. A média de pesos dá robustez; a cidade requer dados próprios.")
poner(buscar("[Resumen:"), RESUMEN)
poner(buscar("Palabras clave:"), " inteligencia artificial, cartografía, teléfono móvil", desde=1)
poner(buscar("[Abstract:"), ABSTRACT)
poner(buscar("Keywords:"), " artificial intelligence, cartography, mobile phone", desde=1)
poner(buscar("[Resumo:"), RESUMO)
poner(buscar("Palavras-chave:"), " inteligência artificial, cartografia, telefone celular", desde=1)

# ---------------------------------------------------------------- Introducción
INTRO = [
    "Los baches y las grietas del pavimento afectan la seguridad y el costo de mantenimiento de las vías, y su "
    "inspección suele hacerse recorriendo las calles y anotando cada daño. La detección automática mediante visión "
    "por computadora permite hacer esa revisión con cámaras de bajo costo (Ma et al., 2022): Maeda et al. (2018) "
    "mostraron que bastan las imágenes de un teléfono montado en un vehículo, y el conjunto RDD2022 extendió esa idea "
    "a 47.420 imágenes de seis países (Arya et al., 2024).",
    "El desempeño de estos detectores depende de la cámara, del tipo de vía y de las condiciones de captura (Ma et al., "
    "2022), por lo que Arya et al. (2021) proponen entrenar con imágenes de varios países para que un detector funcione "
    "en todos ellos. Además, ajustar un modelo con datos propios suele mejorarlo en esos datos, pero puede empeorarlo "
    "fuera de ellos, porque el ajuste deforma lo que el modelo había aprendido antes (Kumar et al., 2022); promediar "
    "los pesos iniciales y los ajustados es una forma conocida de conservar ambas cosas (Wortsman et al., 2021). En "
    "consecuencia, un buen resultado en los datos de entrenamiento no garantiza el mismo resultado en una ciudad "
    "distinta.",
    "En el primer parcial de la materia se integró un detector de baches en un prototipo de escritorio y se "
    "reprodujeron sus métricas; quedaron pendientes la ejecución en el teléfono, la vinculación con un mapa y un "
    "entrenamiento propio y documentado. Llevar las detecciones a un mapa facilita su uso en la gestión urbana (Sahoo "
    "et al., 2025), y ejecutar el detector en el propio teléfono evita enviar video de la calle, con personas y "
    "placas, a un servidor.",
    "El objetivo de este trabajo es implementar y evaluar una aplicación web que detecta baches y grietas en el "
    "teléfono móvil, sin servidor, y registra cada daño como un punto en la cartografía de Santa Cruz de la Sierra. El "
    "estudio es una investigación de la materia Inteligencia Artificial, del cuarto semestre de la licenciatura en "
    "Ingeniería en Inteligencia Artificial de la Universidad Católica Boliviana “San Pablo”. El aporte propio comprende "
    "la recopilación y depuración de 12.204 fotografías de seis fuentes, el entrenamiento en una GPU de AMD, una regla "
    "de selección del modelo fijada antes de conocer los resultados, la prueba en videos de la ciudad y la aplicación "
    "con su mapa.",
]
p = buscar("[Introducción:")
poner(p, INTRO[0])
c = Cursor(p)
for t in INTRO[1:]:
    c.parrafo(t)

# ---------------------------------------------------------------- Metodología
p = buscar("[Metodología:")
poner(p, "Se sigue un enfoque cuantitativo: se recopilan y depuran datos, se entrena y selecciona un detector, se "
         "desarrolla la aplicación y se evalúa todo en datos etiquetados y en videos sin etiquetas. El entrenamiento y "
         "la medición se ejecutan en una GPU AMD Instinct MI300X de AMD Developer Cloud, con ROCm, PyTorch 2.15 y "
         "Ultralytics 8.4.191; la aplicación se prueba en una laptop con procesador Intel Core i7-1355U, gráficos "
         "integrados, 7,7 GB de memoria y Windows 11.")
c = Cursor(p)
c.parrafo("Se recopilaron 12.204 fotografías de seis conjuntos públicos de daños viales, tomadas con cámaras y en "
          "países distintos (tabla 1). Las etiquetas de cada fuente se convirtieron a un esquema común: grietas "
          "longitudinales (D00), transversales (D10) y de cocodrilo (D20), baches (D40), grieta sin tipo y tapa o "
          "rejilla. Cada conjunto se evalúa solo en las clases que tiene etiquetadas completas; por ejemplo, Bisman y "
          "Bucko solo marcan baches, y de RDD2022 solo se evalúan grietas.")
c.parrafo("La partición agrupa las fotografías por video o tramo de ruta y separa los casi duplicados entre "
          "entrenamiento, validación y prueba mediante una huella visual de 256 bits; la prueba queda congelada desde "
          "el inicio. La depuración retira del entrenamiento 619 fotografías (596 montajes de Egipto, 11 duplicados "
          "exactos, 11 con superposiciones y 1 sin pavimento), corrige 14 etiquetas, 9 de entrenamiento y 5 de "
          "validación, y deja 243 fotografías dudosas, porque no está claro que retirarlas mejore.")
c.tabla("Conjuntos de datos y partición",
        ["Conjunto", "Cámara", "Entrenamiento", "Validación", "Prueba", "Clases evaluadas"],
        [["Bisman", "Celular, de cerca", "3.259", "686", "677", "Baches"],
         ["Brasil", "Desde vehículo", "1.517", "358", "359", "Baches y grietas"],
         ["Bucko (Eslovaquia)", "Cámara de tablero", "1.261", "133", "395", "Baches"],
         ["Egipto", "Celular al frente", "470", "45", "91", "Baches y grietas"],
         ["HRP4K", "4K, en perspectiva", "3.186*", "–", "899", "Baches"],
         ["RDD2022", "Varias, seis países", "–", "–", "300", "Grietas"],
         ["Total", "", "6.507", "1.222", "2.721", ""]],
        [2.9, 3.3, 2.8, 2.1, 1.7, 3.2],
        "Fotografías después de la depuración; el entrenamiento incluye fotografías que no se usan para evaluar, como los "
        "aumentos de Egipto. *El entrenamiento y la validación oficiales de HRP4K solo se suman en la segunda corrida y "
        "no entran en el total. Fuentes: Bisman-Singh-Dev (2026), Passos (2020), Bučko et al. (2022), Hanno100 (s. f.), "
        "Chen et al. (2026), dronefreak (2026) y Arya et al. (2024).")
c.parrafo("El detector es un YOLO26s de 9,47 millones de parámetros. Se entrena con transferencia de aprendizaje: "
          "parte de pesos públicos de la misma arquitectura, preentrenados con las clases de RDD2022, y se ajusta como "
          "detector de baches con las 6.507 fotografías de entrenamiento (tabla 2). A los aumentos de datos habituales "
          "se suman compresión JPEG, reducción de resolución y desenfoque de movimiento, para imitar el video que se "
          "envía por WhatsApp. Una segunda corrida repite la receta sumando al entrenamiento las 3.186 fotografías de "
          "entrenamiento y validación de HRP4K.")
c.tabla("Configuración del entrenamiento",
        ["Parámetro", "Valor"],
        [["Tamaño de entrada y lote", "1.280 píxeles; 32 imágenes"],
         ["Optimizador", "SGD, tasa inicial 0,005 con caída coseno hasta el 5 %, momento 0,937"],
         ["Regularización", "Decaimiento de pesos 0,0005; 3 épocas de calentamiento"],
         ["Épocas", "Hasta 150, con parada tras 30 sin mejora en validación"],
         ["Aumentos habituales", "Mosaico (sin él las últimas 15 épocas), volteo horizontal, escala ±50 %, traslación 10 %"],
         ["Aumentos tipo WhatsApp", "JPEG de calidad 20 a 60 (30 %), resolución al 40-80 % (25 %), desenfoque (15 %)"],
         ["Cómputo", "AMD Instinct MI300X, precisión mixta, caché de imágenes en disco"]],
        [4.6, 11.4],
        "Entre paréntesis, la probabilidad de aplicar cada aumento. Pesos iniciales: TamAko783 (s. f.); biblioteca: "
        "Ultralytics (2026).", tam=9)
θ_ = lambda sub: msub([mt("θ")], [mt(sub, recto=True)])
c.parrafo("Para conservar la robustez de los pesos iniciales, se promedian peso por peso con los ajustados, sin "
          "entrenar de nuevo (ecuación [1]), donde α es la proporción del modelo ajustado; se prueban α = 0,25, 0,50 "
          "y 0,75.")
assert c.ecuacion(mt("θ"), mpar(mt("α")), mt("="), mt("α"), mt("·"), θ_("ajustado"), mt("+"),
                  mpar(mt("1"), mt("−"), mt("α")), mt("·"), θ_("inicial")) == 1
c.parrafo("La regla de selección se fija antes de ver estos resultados: HRP4K, que no participa del ajuste, se divide "
          "en dos mitades fijas de 467 y 432 fotografías, A para elegir y B para informar, y a 640 píxeles, el tamaño "
          "de la aplicación, se elige el modelo de mayor AP50 de validación entre los que no empeoran más de 0,005 en "
          "la mitad A (ecuación [2]):")
AP_ = lambda sub, arg: [msub([mt("AP", recto=True)], [mt(sub, recto=True)]), mpar(*arg)]
assert c.ecuacion(msup([mt("θ")], [mt("*")]), mt("="),
                  mfunc("argmáx", [mt("θ"), mt("∈"), mt("C")], AP_("val", [mt("θ")])), mt(","), mt("  "),
                  mt("C"), mt("="),
                  mpar(mt("θ"), mt(":"), *AP_("A", [mt("θ")]), mt("≥"), *AP_("A", [θ_("inicial")]), mt("−"),
                       mt("0,005", texto_normal=True), abre="{", cierra="}")) == 2
c.parrafo("donde AP_{val} es el AP50 de validación, AP_{A} el de la mitad A y θ recorre los pesos iniciales, los "
          "promedios y el modelo entrenado. El modelo final conserva las cuatro clases iniciales: las filas de grietas "
          "quedan intactas y solo se promedia la de baches.")
c.parrafo("Una detección es correcta si su caja coincide al menos en un 50 % con una etiqueta de la misma clase, medido "
          "como intersección sobre unión (IoU, ecuación [3]), donde B_{p} es la caja predicha y B_{e}, la etiquetada.")
B_ = lambda sub: msub([mt("B")], [mt(sub)])
assert c.ecuacion(mt("IoU", recto=True), mt("="),
                  mfrac([mt("área", recto=True), mpar(B_("p"), mt("∩"), B_("e"))],
                        [mt("área", recto=True), mpar(B_("p"), mt("∪"), B_("e"))])) == 3
c.parrafo("Con esos emparejamientos, ordenados por confianza, se calcula el AP con 101 puntos de exhaustividad r "
          "(ecuación [4]), donde p(r′) es la precisión cuando la exhaustividad es r′; el AP50 exige una IoU de 0,5.")
assert c.ecuacion(mt("AP", recto=True), mt("="), mfrac([mt("1")], [mt("101")]),
                  msuma([mt("r"), mt("∈"), mpar(mt("0; 0,01; …; 1", texto_normal=True), abre="{", cierra="}")],
                        [mfunc("máx", [mt("r′"), mt("≥"), mt("r")], [mt("p"), mpar(mt("r′"))])])) == 4
c.parrafo("Se comparan el modelo final, el entrenado sin promediar, la segunda corrida, los promedios con otras "
          "proporciones, los pesos iniciales y el detector que usaba la aplicación, como referencia. Los umbrales de "
          "confianza de la aplicación se eligen en validación por F1, la media armónica de la precisión P y la "
          "exhaustividad R (ecuación [5]), y la prueba se mide una sola vez, salvo la mitad de HRP4K destinada a "
          "elegir.")
assert c.ecuacion(msub([mt("F")], [mt("1")]), mt("="),
                  mfrac([mt("2"), mt("·"), mt("P"), mt("·"), mt("R")], [mt("P"), mt("+"), mt("R")])) == 5
for t in [
    "La aplicación es una página web instalable que ejecuta el modelo, exportado a ONNX, dentro del teléfono: en su GPU "
    "mediante WebGPU o, si no la hay, en su CPU. Un daño se registra cuando aparece en 2 de 3 cuadros seguidos y se "
    "dibuja como un punto en un mapa de OpenStreetMap, en la posición del GPS o en la de un recorrido de prueba que "
    "simula un vehículo sobre una ruta real. La severidad se estima por el área de la caja. Los puntos quedan en el "
    "dispositivo y se exportan en GeoJSON o CSV.",
    "Se analizan cinco videos de calles de Santa Cruz de la Sierra grabados con celular y enviados por "
    "WhatsApp, con 10.162 cuadros de unos 848 × 478 píxeles: dos con la cámara al frente desde un auto y tres al costado "
    "de un auto o de un micro. Cada modelo procesa todos los cuadros a 864 × 512 píxeles con sus umbrales. Como no "
    "tienen etiquetas, se cuentan los cuadros con marca y los eventos, tramos de al menos 3 cuadros con marca. "
    "Además, la marca más segura de cada evento se revisa a ojo, sin saber qué modelo la hizo, y se clasifica como "
    "bache, otro daño del pavimento (grieta, parche roto o desgaste), algo que no es daño (junta, mancha, sombra, "
    "vehículo o marca vial) o dudosa.",
]:
    c.parrafo(t)
c.parrafo(
    f"Para medir la detección con cajas en la ciudad, se etiquetaron {scz50['cuadros']} cuadros de los cinco videos, "
    "tomados de las versiones con caras y placas difuminadas. En cada video, el 60 % inicial sirve para ajustar y el "
    f"40 % final, después de 2 s de separación, para probar: {scz50['cuadros'] - scz50p['cuadros']} y "
    f"{scz50p['cuadros']} cuadros. La mitad se eligió al azar y la otra mitad entre los cuadros que algún modelo "
    "marcó. Un solo revisor vio en cada cuadro las cajas propuestas por los cinco modelos, sin saber de cuál era cada "
    "una, aceptó las que eran baches y agregó los que ninguno marcó. Como las cajas aceptadas vienen de los modelos y "
    "la extensión de un bache es ambigua, además del AP50 se informa el AP con una superposición mínima de 0,3 (AP30).")
if ajuste:
    c.parrafo(
        "Con los cuadros de ajuste se entrenó una variante del modelo final, v2, que los suma, repetidos cinco veces, a "
        "las fotografías de entrenamiento, con la misma receta, y se promedió con el modelo final (α de 0,25 a 1). Antes "
        "de etiquetar se fijó la regla para reemplazar al modelo de la aplicación: al menos 0,03 más de AP50 en los "
        "cuadros de prueba, sin perder más de 0,01 en Egipto ni más de 0,005 en HRP4K-A. Además, la receta del modelo "
        "entrenado se repitió desde los pesos iniciales con otra semilla, con el mismo promedio, para estimar la "
        "variación por el azar del entrenamiento.")
c.parrafo("El código, el modelo y los informes están disponibles en https://github.com/Duranzzz/baches-scz; la "
          "aplicación, en https://duranzzz.github.io/baches-scz-app/ (código en "
          "https://github.com/Duranzzz/baches-scz-app), y los datos procesados, en "
          "https://huggingface.co/datasets/Duranishhh/baches-scz.").alignment = 0   # sin justificar: las URL abren huecos

# ---------------------------------------------------------------- Resultados y discusión
res_h = buscar("Resultados y discusión")
borrar_entre(res_h, buscar("Conclusiones"))
c = Cursor(res_h)

c.sub("Datos recopilados")
c.parrafo(
    "Las cuatro fuentes con baches en el entrenamiento aportan cámaras muy distintas (figura 1): fotografías de cerca "
    "en Bisman, vehículo en ruta en Brasil, cámara de tablero en Bucko y celular al frente en Egipto. El entrenamiento "
    "reúne 6.507 fotografías con 8.432 baches etiquetados, y la validación, 1.222 fotografías con 1.474 baches. La "
    "depuración encuentra que 596 de las 1.951 fotografías de Egipto (31 %) son montajes de varias imágenes, un "
    "artefacto de su redistribución que no se parece a la calle, por lo que se retiran del entrenamiento.")
c.figura("Fotografías de cuatro fuentes con sus etiquetas", "figura_muestras.png", 16.0,
         "Cajas naranjas: baches; azules: grietas. Fuentes: Passos (2020), Bučko et al. (2022), Hanno100 (s. f.) y Chen "
         "et al. (2026).")

c.sub("Entrenamiento")
c.parrafo(
    "La primera corrida termina tras 91 épocas, en 1,18 horas (unos 47 segundos por época), al cumplirse 30 épocas sin "
    "mejora; la segunda, con HRP4K, termina tras 82 épocas, en 1,58 horas. En ambas, el mAP50 de validación sube "
    "rápido durante las primeras 20 épocas y luego se estabiliza entre 0,72 y 0,78 (figura 2). Estos valores son altos "
    "porque la validación comparte caminos con el entrenamiento en algunos conjuntos; por eso los modelos se comparan "
    "en la prueba congelada y en un conjunto no visto.")
def grafico_curvas():
    datos, estilos = XyChartData(), []
    for nombre, etiqueta, color, guion in (
            ("a1_y26s", "Corrida 1: 6.507 fotografías", COL["azul"], None),
            ("a2_y26s_hrp4k", "Corrida 2: + HRP4K (9.693)", COL["naranja"], MSO_LINE_DASH_STYLE.DASH)):
        with open(INF / "corridas" / nombre / "results.csv", encoding="utf-8") as f:
            filas = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
        ys = [float(r["metrics/mAP50(B)"]) for r in filas]
        s = datos.add_series(etiqueta, number_format="0.0")
        for r, y in zip(filas, ys):
            s.add_data_point(int(float(r["epoch"])), y * X100)
        estilos.append((color, guion, ys))
    gf, ch = grafico_nuevo(XL_CHART_TYPE.XY_SCATTER_LINES_NO_MARKERS, datos, 16.0, 7.0)
    ejes(ch, "Época", "mAP50 de baches en validación", (50, 85, 5), x=(0, 100, 10), fmt_x="0")
    for s, (color, guion, ys) in zip(ch.plots[0].series, estilos):
        linea(s, color, 1.5, guion)
        i = max(range(len(ys)), key=ys.__getitem__)
        rotular(s, i, f"máximo {coma(ys[i])}", color=color)
    return [(gf, 16.0, 7.0)]


c.grafico("mAP50 de baches en validación durante el entrenamiento", grafico_curvas(),
          "Valor que calcula la biblioteca de entrenamiento en cada época. Gráfico editable. Elaboración propia.")

c.sub("Efecto del entrenamiento y del promedio de pesos")
n_t3 = Cursor.n_tabla + 1
c.parrafo(
    "En la prueba a 640 píxeles, el entrenamiento eleva el AP50 de los conjuntos que vio, por ejemplo de 0,306 a 0,629 "
    f"en Bucko y de 0,356 a 0,442 en Egipto, pero en HRP4K, que no vio, lo baja de 0,048 a 0,017 (tabla {n_t3}). Estos "
    "datos indican que el entrenamiento especializa el modelo en las cámaras y en el etiquetado de sus datos, una "
    "ganancia que no se traslada a un dominio nuevo, condición que también cumplen las calles cruceñas.")
c.tabla("AP50 de baches según la proporción del modelo entrenado, a 640 píxeles",
        ["Modelo", "Val.", "Bisman", "Brasil", "Bucko", "Egipto", "Prueba", "HRP4K-A", "HRP4K-B"],
        [["Detector de referencia", "0,142", "0,153", "0,472", "0,150", "0,151", "0,128", "0,023", "0,044"],
         ["Pesos iniciales (α = 0)", "0,191", "0,187", "0,588", "0,306", "0,356", "0,196", "0,034", "0,065"],
         ["Promedio α = 0,25", "0,299", "0,366", "0,613", "0,409", "0,471", "0,282", "0,029", "0,059"],
         ["Promedio α = 0,50", "0,478", "0,623", "0,653", "0,509", "0,534", "0,393", "0,025", "0,040"],
         ["Promedio α = 0,75", "0,680", "0,791", "0,683", "0,598", "0,484", "0,472", "0,015", "0,019"],
         ["Entrenado (α = 1)", "0,732", "0,834", "0,662", "0,629", "0,442", "0,496", "0,011", "0,019"]],
        [3.6, 1.4, 1.5, 1.4, 1.4, 1.5, 1.5, 1.85, 1.85],
        "Val.: validación con los cuatro conjuntos con baches; las demás columnas son de la prueba congelada (Prueba: "
        "total, con HRP4K). Bucko, Brasil y Bisman comparten caminos o fotografías espejadas con el entrenamiento, por lo "
        "que sus valores son optimistas para los modelos entrenados. HRP4K-A (467 fotografías) se usa para elegir y "
        "HRP4K-B (432), solo para informar.", tam=8.5)
n_f3 = Cursor.n_figura + 1
c.parrafo(
    f"El promedio de pesos ordena ese intercambio (figura {n_f3}). Con α = 0,25, el AP50 de validación sube de 0,191 a "
    "0,299 y el de la prueba de 0,196 a 0,282, mientras que en la mitad de HRP4K para elegir pasa de 0,034 a 0,029, "
    "dentro del margen; con α = 0,50 o más, la caída fuera de los datos conocidos supera el margen. La regla elige "
    "α = 0,25, que en la mitad para informar obtiene 0,059, frente a 0,065 de los pesos iniciales y 0,044 del detector "
    "de referencia; a 1.280 píxeles, este promedio incluso supera a los pesos iniciales en HRP4K (0,098 frente a 0,095).")
def grafico_promedio():
    """Los mismos valores de la tabla anterior (informes/entrenamiento_amd.md), en dos paneles."""
    categorias = ["0", "0,25", "0,50", "0,75", "1"]
    paneles = [
        ("Conjuntos usados en el ajuste", (0, 80, 20), [
            ("Validación, total", [0.191, 0.299, 0.478, 0.680, 0.732]),
            ("Prueba, Egipto", [0.356, 0.471, 0.534, 0.484, 0.442])]),
        ("HRP4K, no visto al entrenar", (0, 8, 2), [
            ("Mitad A (elegir)", [0.034, 0.029, 0.025, 0.015, 0.011]),
            ("Mitad B (informar)", [0.065, 0.059, 0.040, 0.019, 0.019]),
            ("Límite de la regla", [0.029] * 5)]),
    ]
    estilos = [(COL["azul"], None, XL_MARKER_STYLE.CIRCLE), (COL["naranja"], MSO_LINE_DASH_STYLE.DASH,
               XL_MARKER_STYLE.SQUARE), (COL["gris"], MSO_LINE_DASH_STYLE.ROUND_DOT, None)]
    graficos = []
    for titulo, y, series in paneles:
        datos = CategoryChartData(number_format="0.0")
        datos.categories = categorias
        for nombre, valores in series:
            datos.add_series(nombre, [v * X100 for v in valores])
        gf, ch = grafico_nuevo(XL_CHART_TYPE.LINE_MARKERS, datos, 8.0, 7.4, titulo=titulo)
        ejes(ch, "Proporción α del modelo entrenado", "AP50 de baches", y)
        for s, (color, guion, marca) in zip(ch.plots[0].series, estilos):
            linea(s, color, 1.6 if marca else 1.0, guion, marca)
        graficos.append((gf, 8.0, 7.4))
    return graficos


c.grafico("AP50 de baches según la proporción del modelo entrenado en el promedio de pesos", grafico_promedio(),
          "A 640 píxeles. α = 0: pesos iniciales; α = 1: modelo entrenado; la regla eligió α = 0,25. La línea "
          "punteada gris marca el límite de la regla de selección (ecuación [2]). Gráficos editables. Elaboración "
          "propia.")

c.sub("Modelo final")
n_f4 = Cursor.n_figura + 1
c.parrafo(
    f"El modelo final mejora frente al detector de referencia y a los pesos iniciales en todos los conjuntos, salvo en "
    f"HRP4K, donde queda entre ambos (figura {n_f4}): en el total de la prueba pasa de 0,128 y 0,196 a 0,276, y en "
    "Egipto, de 0,151 y 0,356 a 0,472. También mantiene la detección de grietas, con un AP50 de 0,199 en Egipto y de "
    "0,063 en Brasil, frente a 0,114 y 0,070 de los pesos iniciales y 0,075 y 0,061 del detector de referencia.")
def grafico_barras():
    conjuntos = ["bisman", "brazil", "bucko", "egypt", "hrp4k", "todos"]
    series = [("Detector de referencia", "yolo12s_RDD2022_best", COL["agua"]),
              ("Pesos iniciales", "rdd_yolo26s_dist", COL["naranja"]),
              ("Modelo final (aplicación)", "wise4_a25", COL["azul"])]
    datos = CategoryChartData(number_format="0.0")
    datos.categories = ["Bisman", "Brasil", "Bucko", "Egipto", "HRP4K", "Total"]
    for etiqueta, nombre, _ in series:
        datos.add_series(etiqueta, [ap(nombre, "test", 640, x) * X100 for x in conjuntos])
    gf, ch = grafico_nuevo(XL_CHART_TYPE.COLUMN_CLUSTERED, datos, 16.0, 7.5, XL_LEGEND_POSITION.TOP)
    ejes(ch, None, "AP50 de baches (prueba, 640 px)", (0, 70, 10))
    plot = ch.plots[0]
    plot.gap_width, plot.overlap = 60, -5
    for s, (_, _, color) in zip(plot.series, series):
        s.format.fill.solid()
        s.format.fill.fore_color.rgb = rgb(color)
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format, dl.number_format_is_linked = COMA2, False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size = PPt(6.5)
    return [(gf, 16.0, 7.5)]


c.grafico("AP50 de baches por conjunto en la prueba congelada", grafico_barras(),
          "A 640 píxeles. Gráfico editable. Elaboración propia con los resultados de la prueba congelada.")
n_f5 = Cursor.n_figura + 1
c.parrafo(
    f"Los umbrales de la aplicación se eligen en validación (figura {n_f5}). Para baches, el F1 cae a medida que sube "
    "el umbral, y se usa 0,10, el mínimo que admite la aplicación, con un F1 de 0,338. Para grietas, el F1 apenas "
    "cambia entre 0,10 y 0,25, de modo que se usa 0,25, el valor más alto de ese tramo, para reducir las falsas "
    "alarmas.")
def grafico_umbral():
    """Los valores de docs/figuras/generar_figuras.py (F1 medio en validación del modelo final, 640 px)."""
    conf = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
    curvas_f1 = [("Baches", [0.365, 0.338, 0.329, 0.307, 0.294, 0.277, 0.242]),
                 ("Grietas", [0.156, 0.149, 0.149, 0.144, 0.142, 0.125, 0.121])]
    datos = XyChartData()
    for nombre, ys in curvas_f1:
        s = datos.add_series(nombre, number_format="0.0")
        for x, y in zip(conf, ys):
            s.add_data_point(x * X100, y * X100)
    gf, ch = grafico_nuevo(XL_CHART_TYPE.XY_SCATTER_LINES, datos, 16.0, 6.8)
    ejes(ch, "Umbral de confianza", "F1 medio entre conjuntos (validación)", (10, 40, 5), x=(0, 45, 5), fmt_x=COMA2)
    s_b, s_g = ch.plots[0].series
    linea(s_b, COL["azul"], 1.6, None, XL_MARKER_STYLE.CIRCLE)
    linea(s_g, COL["naranja"], 1.6, MSO_LINE_DASH_STYLE.DASH, XL_MARKER_STYLE.SQUARE)
    rotular(s_b, conf.index(0.10), "umbral de la aplicación: 0,10", color=COL["azul"], tam=9)
    rotular(s_g, conf.index(0.25), "umbral de la aplicación: 0,25", color=COL["naranja"], tam=9)
    return [(gf, 16.0, 6.8)]


c.grafico("F1 en validación según el umbral de confianza del modelo final", grafico_umbral(),
          "Media del F1 de los conjuntos con cada clase, con el detector solo y a 640 píxeles. Gráfico editable. "
          "Elaboración propia.")

c.sub("Segunda corrida con HRP4K")
a2_cuadros = tot["a2_y26s_hrp4k"][0]
c.parrafo(
    "La segunda corrida suma al entrenamiento las 3.186 fotografías del entrenamiento y la validación de HRP4K. A 640 "
    "píxeles obtiene un AP50 de baches de 0,634 en el total de la prueba y de 0,367 en HRP4K, frente a 0,496 y 0,017 de "
    "la primera, aunque en Egipto baja de 0,442 a 0,385. Como HRP4K deja de ser un dominio no visto, este modelo no "
    "compite en la regla de selección. En los videos marca baches en el "
    f"{coma(100 * a2_cuadros / tot_cuadros, 1)} % de los cuadros ({miles(a2_cuadros)}), y en scz_01 llega al "
    f"{coma(100 * a2_01['cuadros_con_bache'] / videos['scz_01_lateral_auto_hormigon']['cuadros'], 1)} % "
    f"({miles(a2_01['cuadros_con_bache'])}), lo que sugiere falsas alarmas sobre el hormigón.")

c.sub("Aplicación")
n_f6 = Cursor.n_figura + 1
c.parrafo(
    f"La aplicación se ejecuta completa en el teléfono (figura {n_f6}). El modelo corre en segundo plano con ONNX "
    "Runtime Web (Microsoft, 2026), en la GPU mediante WebGPU o en la CPU, y el mapa usa MapLibre GL JS (MapLibre, 2026) "
    "con datos de OpenStreetMap (Colaboradores de OpenStreetMap, 2026). En seis casos de prueba, sus cajas, clases y "
    "confianzas coinciden con las de la biblioteca de entrenamiento, con una diferencia máxima de 0,18 píxeles.")
c.figura("Funcionamiento de la aplicación en el teléfono", "figura_arquitectura.png", 15.0, "Elaboración propia.")
n_f7 = Cursor.n_figura + 1
c.parrafo(
    "En la laptop, la aplicación procesa entre 8 y 12 cuadros por segundo con los gráficos integrados, y entre 0,39 y "
    "0,62 segundos por cuadro con la CPU y un solo hilo; el modelo, de 38,5 MB, se descarga una vez y la aplicación "
    f"funciona sin conexión, salvo el fondo del mapa. La figura {n_f7} muestra una demostración con el video scz_03 y el "
    "recorrido de prueba por la avenida Cristo Redentor: en dos ejecuciones registra entre 9 y 15 baches y entre 15 y "
    "18 grietas, y al tocar un punto muestra el recorte del daño, su confianza, el segundo del video y la calle. Los "
    "puntos se exportan en GeoJSON o CSV para abrirlos en un sistema de información geográfica.")
c.figura("Demostración de la aplicación con un video de Santa Cruz de la Sierra y el recorrido de prueba",
         "figura2_app.jpg", 13.0,
         "De izquierda a derecha: inicio, segundo 75, final del video y detalle de un punto. Con el recorrido de prueba, "
         "el punto no es la ubicación real del daño, y la aplicación lo advierte. Elaboración propia.")

c.sub("Videos de Santa Cruz de la Sierra")
n_t4 = Cursor.n_tabla + 1
n_f8 = Cursor.n_figura + 1
c.parrafo(
    f"En los cinco videos, el modelo final marca baches en el {pct['wise4_a25']} % de los cuadros "
    f"({miles(tot['wise4_a25'][0])} de {miles(tot_cuadros)}) y forma {tot['wise4_a25'][1]} eventos, frente al "
    f"{pct['yolo12s_rdd2022']} % ({miles(tot['yolo12s_rdd2022'][0])}) y {tot['yolo12s_rdd2022'][1]} eventos del "
    f"detector de referencia (tabla {n_t4}); en scz_03, tomado al frente, pasa de "
    f"{v3['yolo12s_rdd2022']['eventos_bache']} a {v3['wise4_a25']['eventos_bache']} eventos. Sus marcas más seguras "
    f"son baches reales sobre el pavimento (figura {n_f8}). El modelo entrenado sin promediar marca el {pct['a1_y26s']} % "
    f"de los cuadros ({miles(tot['a1_y26s'][0])}): a ojo, también acierta daños reales, pero marca manchas y desgaste "
    "menor con confianzas de 0,19 a 0,31.")
c.tabla("Cuadros con marca de bache y eventos en los videos de Santa Cruz de la Sierra",
        ["Video (cámara)", "Cuadros", "Referencia", "Iniciales", "Final", "Entrenado", "Con HRP4K"],
        filas_v, [3.8, 1.4, 2.2, 2.2, 2.2, 2.2, 2.2],
        "Porcentaje de cuadros con al menos una marca de bache y, después del punto y coma, los eventos: tramos de 3 cuadros o más con marca. Final: el modelo de la aplicación; Entrenado: sin "
        "promediar; Con HRP4K: la segunda corrida. Todos los cuadros a 864 × 512 píxeles, con los umbrales de cada "
        "modelo. Sin etiquetas, son marcas y no aciertos.", tam=8)
c.figura("Baches detectados por el modelo final en los videos de Santa Cruz de la Sierra", "figura_videos.png", 16.0,
         "Recortes del pavimento alrededor de cada detección, con su confianza. Elaboración propia.")
c.parrafo(
    f"En grietas, el modelo final marca el {pctg['wise4_a25']} % de los cuadros ({miles(tot['wise4_a25'][2])}), frente "
    f"al {pctg['yolo12s_rdd2022']} % ({miles(tot['yolo12s_rdd2022'][2])}) del detector de referencia y al "
    f"{pctg['rdd_yolo26s']} % ({miles(tot['rdd_yolo26s'][2])}) de los pesos iniciales. Las juntas de las losas de "
    "hormigón y la carrocería del propio vehículo, visible en los videos laterales, aparecen a veces como grietas.")
n_t5 = Cursor.n_tabla + 1
c.parrafo(
    f"Para estimar la precisión en la ciudad, se revisó la marca más segura de cada uno de los {n_rev} eventos de los "
    f"cinco modelos, mezclados al azar y sin saber qué modelo los había marcado (tabla {n_t5}). El "
    f"{p100(rfin['precision_bache'])} de los eventos del modelo final son baches, frente al "
    f"{p100(ra1['precision_bache'])} del modelo entrenado sin promediar, el {p100(ra2['precision_bache'])} de la "
    f"segunda corrida y el {p100(rref['precision_bache'])} del detector de referencia, que marca sobre todo grietas "
    f"como si fueran baches. Entre todos los modelos aparecen {rev['baches_distintos']} baches distintos: el modelo "
    f"final encuentra {rfin['baches_hallados']}; el modelo sin promediar, {ra1['baches_hallados']}, con casi el doble "
    f"de eventos ({ra1['eventos']} frente a {rfin['eventos']}), y la segunda corrida, {ra2['baches_hallados']}. En "
    "esta revisión, las marcas adicionales del modelo sin promediar son sobre todo grietas, manchas y juntas.")
c.tabla("Revisión a ciegas de los eventos de bache en los videos de Santa Cruz de la Sierra",
        ["Modelo", "Eventos", "Bache", "Otro daño", "No es daño", "Dudoso", "Precisión", "Baches hallados"],
        [[nombre, str(REV[k]["eventos"]), str(REV[k]["B"]), str(REV[k]["O"]), str(REV[k]["N"]), str(REV[k]["D"]),
          p100(REV[k]["precision_bache"]), str(REV[k]["baches_hallados"])] for k, nombre in REV_ORDEN],
        [2.6, 1.6, 1.5, 1.9, 2.1, 1.7, 2.0, 2.6],
        "Precisión: eventos con un bache entre los juzgados, sin contar los dudosos. Si los dudosos se cuentan como "
        f"errores o como baches, el modelo final sigue primero ({p100(rfin['B'] / rfin['eventos'])} a "
        f"{p100((rfin['B'] + rfin['D']) / rfin['eventos'])}). Baches hallados: de los {rev['baches_distintos']} "
        "baches distintos que marcó al menos un modelo, cuántos marcó cada uno; los que ningún modelo marcó no "
        "cuentan. Un solo revisor, con video de 848 × 478 píxeles.", tam=8.5)

c.sub("Cuadros etiquetados de Santa Cruz de la Sierra")
n_t6 = Cursor.n_tabla + 1
a50, a30 = scz50["ap50"], scz30["ap50"]
c.parrafo(
    f"En los {scz50['cuadros']} cuadros etiquetados, con {scz50['baches']} baches, la segunda corrida es la que más "
    f"baches encuentra, con un AP50 de {c3(a50['a2'])}, seguida del modelo sin promediar ({c3(a50['a1'])}) y del "
    f"modelo final ({c3(a50['v1'])}), que apenas supera al detector de referencia ({c3(a50['referencia'])}) y a los "
    f"pesos iniciales ({c3(a50['B0'])}) (tabla {n_t6}). Con una superposición mínima de 0,3, el modelo final sube a "
    f"{c3(a30['v1'])} y se acerca al modelo sin promediar ({c3(a30['a1'])}): muchas de sus cajas encuentran el bache, "
    "pero con otra extensión. Así, ordenadas por confianza, las marcas de los modelos ajustados por completo encuentran "
    f"más baches de la ciudad, aunque con el umbral de la aplicación son baches con menos frecuencia (tabla {n_t5}).")
filas6 = [[nombre, c3(a50[k]), c3(a30[k]), c3(scz50p["ap50"][k]), c3(scz30p["ap50"][k])] for k, nombre in SCZ_ORDEN]
cand = list(ajuste["candidatos"].values())
filas6 += [[f"v2, α = {coma(v['alfa'], 2)}", "–", "–", c3(v["scz_prueba50"]), c3(v["scz_prueba30"])] for v in cand]
c.tabla("AP de baches en los cuadros etiquetados de Santa Cruz de la Sierra",
        ["Modelo", "AP50, 200 cuadros", "AP30, 200 cuadros", "AP50, prueba", "AP30, prueba"], filas6,
        [3.6, 3.1, 3.1, 3.1, 3.1],
        f"A 640 píxeles. AP50 y AP30: acierto con una superposición (IoU) de al menos 0,5 y 0,3. Prueba: los "
        f"{scz50p['cuadros']} cuadros del 40 % final de cada video, con {scz50p['baches']} baches. Etiquetas de un solo "
        "revisor, que aceptó o descartó las cajas propuestas por los cinco modelos y agregó las que faltaban. Las "
        f"variantes v2 se ajustaron con los otros {n_aj} cuadros, por eso solo se miden en la prueba, y no propusieron "
        "cajas para las etiquetas.",
        tam=8.5)
# Ninguna v2 cumple: todas mejoran en Santa Cruz y bajan en HRP4K-A (y en HRP4K-B); una baja además en Egipto
assert all(v["condiciones"]["santa_cruz"] and not v["condiciones"]["hrp4k_a"] for v in cand)
assert all(v["hrp4k_b"] < v1d["hrp4k_b"] for v in cand)
egipto_falla = [v for v in cand if not v["condiciones"]["egipto"]]
assert len(egipto_falla) == 1
c.parrafo(
    f"Todas las variantes v2 mejoran en los cuadros de prueba: la mejor, con α = {coma(mejor['alfa'], 2)}, obtiene "
    f"un AP50 de {c3(mejor['scz_prueba50'])}, frente a {c3(v1d['scz_prueba50'])} del modelo final. También suben en "
    f"la prueba congelada, hasta {c3(max(v['test'] for v in cand))} frente a {c3(v1d['test'])}, pero bajan en "
    f"HRP4K-A, de {c3(v1d['hrp4k_a'])} a entre {c3(min(v['hrp4k_a'] for v in cand))} y "
    f"{c3(max(v['hrp4k_a'] for v in cand))}, y lo mismo ocurre en HRP4K-B, que no entra en la regla; la de "
    f"α = {coma(egipto_falla[0]['alfa'], 2)} baja además en Egipto, de {c3(v1d['egipto'])} a "
    f"{c3(egipto_falla[0]['egipto'])}. Ninguna cumple la regla, de modo que el modelo final sigue en la aplicación.")

sem = ajuste["semillas"]
assert c3(sem["wise4_s1"]["hrp4k"]["A"]) == "0,027" and round(sem["a1_s1"]["test"]["todos"], 3) in (0.48, 0.481)
assert c3(sem["B0"]["hrp4k"]["A"] - 0.005) == "0,029"
assert sem["wise4_s1"]["hrp4k"]["A"] < sem["B0"]["hrp4k"]["A"] - 0.005 <= v1d["hrp4k_a"]
falta = sorted(v1d["hrp4k_a"] - 0.005 - v["hrp4k_a"] for v in cand)   # cuánto le falta a cada v2 para el mínimo
assert falta[0] < 0.003 < falta[1], "ya no es solo α = 0,25 la que queda cerca del mínimo"
c.sub("Discusión")
for t in [
    "La caída del modelo entrenado en HRP4K coincide con Kumar et al. (2022), para quienes ajustar todos los pesos "
    "deforma lo aprendido y empeora el desempeño fuera de la distribución del ajuste. El promedio de pesos recupera buena "
    "parte de esa robustez sin entrenar de nuevo, como reportan Wortsman et al. (2021), y aquí se observa en detección de objetos. "
    "En Egipto, el conjunto de prueba más limpio, los promedios incluso superan al modelo entrenado: 0,471 con α = 0,25 "
    "y 0,534 con α = 0,50, frente a 0,442, en línea con la observación de que promediar pesos puede mejorar también "
    "dentro de la distribución (Wortsman et al., 2022). Las variantes v2 repiten el patrón: el ajuste con cuadros de "
    "la ciudad mejora en ella y en los conjuntos que el entrenamiento ya veía, pero empeora en HRP4K, incluso "
    "promediado.",
    "La ventaja de partir de pesos entrenados con imágenes de varios países concuerda con la propuesta de Arya et al. "
    "(2021). En HRP4K, con fotografías 4K en perspectiva (Chen et al., 2026), los baches lejanos ocupan pocos píxeles a "
    "640 y todos los modelos quedan por debajo de 0,05, de modo que las diferencias pequeñas deben leerse con cautela; "
    "a 1.280 píxeles los valores suben hasta 0,098. La repetición con otra semilla reproduce el patrón, con 0,481 "
    "frente a 0,496 en el total de la prueba, pero su promedio con α = 0,25 obtiene 0,027 en HRP4K-A, por debajo del "
    "margen de la regla (0,029). Así, la elección del modelo final, como el descarte de la variante v2 con α = 0,25, "
    "quedó dentro de la variación por el azar del entrenamiento; las demás variantes v2 quedan claramente por debajo "
    "del mínimo. Una aplicación en el teléfono debe equilibrar ese tamaño con la "
    "velocidad, y 640 píxeles mantiene entre 8 y 12 cuadros por segundo.",
    "La métrica que se informa es el AP50, que resume precisión y exhaustividad en un solo valor (Lin et al., 2014; "
    "Padilla et al., 2021). Su valor depende de que la prueba sea independiente del entrenamiento: Barz y Denzler "
    "(2020) muestran que los casi duplicados entre partes inflan las métricas, y por eso la partición usa huellas "
    "visuales (Krawetz, 2013) y grupos por video o ruta. Aun así, Bucko, Brasil y Bisman comparten caminos o "
    "fotografías espejadas entre partes, y documentar estas condiciones junto con los datos, como proponen Gebru et "
    "al. (2021), permite interpretar los resultados con sus límites.",
    "La aplicación detecta en el propio teléfono, sin enviar video, y lleva cada daño confirmado al mapa, en la línea "
    "de la visualización geoespacial de baches de Sahoo et al. (2025) y de la idea de Maeda et al. (2018) de usar el "
    "teléfono como sensor. La confirmación en 2 de 3 cuadros reduce las marcas aisladas, y el recorrido de prueba "
    "permite mostrar el mapa con videos grabados, aunque entonces la posición del punto es simulada.",
    "Las dos medidas en la ciudad no se contradicen: la revisión mira las marcas con el umbral de la aplicación, donde "
    "el modelo final es el más preciso, y el AP recorre todos los umbrales, donde los modelos ajustados por completo, "
    "sobre todo el entrenado con HRP4K, encuentran más baches. La regla de selección, que protegía la robustez fuera de "
    "los datos de entrenamiento, favoreció la precisión; elegir con cuadros de la ciudad el umbral de esos modelos "
    "podría reunir las dos ventajas. Además, los cuadros de prueba vienen de los mismos cinco videos que los de "
    "ajuste, con las mismas cámaras y días, así que la mejora de las variantes v2 puede ser menor en otras calles o "
    "con otros teléfonos.",
    "Las limitaciones son que los cuadros de Santa Cruz de la Sierra los etiquetó un solo revisor, a partir de las "
    "cajas de los modelos, y que vienen de solo cinco videos; el uso de una "
    "mitad de HRP4K para elegir, con márgenes del orden de la variación entre semillas, estimada con una sola "
    "repetición; las 243 fotografías dudosas que siguen en "
    "el entrenamiento; las etiquetas de Brasil, que marcan solo el carril propio; la severidad, estimada por el área de "
    "la caja, y el error de varios metros del GPS del teléfono. Etiquetar más fotografías de calles cruceñas, por "
    "ejemplo las de los puntos que la aplicación ya registra, y con más de un revisor, daría una medida más firme y "
    "más datos para ajustar.",
]:
    c.parrafo(t)

# ---------------------------------------------------------------- Conclusiones
p = buscar("[Conclusiones:")
poner(p, "El trabajo cumple su objetivo: la aplicación detecta baches y grietas en el teléfono, sin servidor, y registra "
         "cada daño como un punto en la cartografía de Santa Cruz de la Sierra, con los datos guardados en el "
         "dispositivo. El modelo final, entrenado con fotografías de seis fuentes, duplica el AP50 de baches del "
         "detector que usaba la aplicación en la prueba congelada, de 0,128 a 0,276, y lo triplica en Egipto, de 0,151 a "
         "0,472, sin perder detección de grietas.")
cc = Cursor(p)
cc.parrafo("El entrenamiento con datos propios especializa el modelo: mejora en los conjuntos que vio y empeora en uno "
           "que no vio. Promediar sus pesos con los iniciales, con una regla fijada de antemano, aprovecha el "
           "entrenamiento con poca pérdida de robustez, algo útil para una ciudad sin datos etiquetados propios. En los videos de la "
           "ciudad, sus marcas son baches con más frecuencia que las de los demás modelos; en cambio, en los cuadros "
           "etiquetados, los modelos ajustados por completo encuentran más baches, de modo que la elección depende de si "
           "importa más evitar falsas alarmas o no perder baches. Sumar más "
           "fotografías de otra fuente, como en la segunda corrida, mejora esa fuente, pero no garantiza una mejora en "
           f"las demás. Lo mismo ocurre al ajustar con cuadros de la ciudad: el AP50 en sus cuadros de prueba sube de "
           f"{c3(v1d['scz_prueba50'])} a {c3(mejor['scz_prueba50'])}, pero baja en HRP4K, y la regla mantuvo el "
           "modelo final.")
cc.parrafo("La aplicación funciona como propuesta de un sistema de inspección: cualquier teléfono con navegador puede "
           "recorrer la ciudad, marcar daños y exportarlos a un sistema de información geográfica. Los cuadros "
           "etiquetados permiten ya medir los modelos en las calles cruceñas; con más fotografías locales, la misma "
           "regla podrá elegir el modelo directamente para la ciudad.")

# ---------------------------------------------------------------- Referencias (APA 7.ª edición)
I = True   # cursiva
REFS = [
    ([("Arya, D., Maeda, H., Ghosh, S. K., Toshniwal, D., Mraz, A., Kashiyama, T., & Sekimoto, Y. (2021). Deep "
       "learning-based road damage detection and classification for multiple countries. ", 0),
      ("Automation in Construction, 132", I), (", 103935. ", 0)], "https://doi.org/10.1016/j.autcon.2021.103935"),
    ([("Arya, D., Maeda, H., Ghosh, S. K., Toshniwal, D., & Sekimoto, Y. (2024). RDD2022: A multi-national image dataset "
       "for automatic road damage detection. ", 0), ("Geoscience Data Journal", I), (". ", 0)],
     "https://doi.org/10.1002/gdj3.260"),
    ([("Barz, B., & Denzler, J. (2020). Do we train on test data? Purging CIFAR of near-duplicates. ", 0),
      ("Journal of Imaging, 6", I), ("(6), 41. ", 0)], "https://doi.org/10.3390/jimaging6060041"),
    ([("Bisman-Singh-Dev. (2026). [Código fuente, datos y pesos de un detector de baches; versión bd8238b]. GitHub. ", 0)],
     "https://github.com/Bisman-Singh-Dev/YOLO-Pothole-Detection-Model/tree/bd8238b0f87083a1cea658336307d15b54eb7500"),
    ([("Bučko, B., Lieskovská, E., Zábovská, K., & Zábovský, M. (2022). Computer vision based pothole detection under "
       "challenging conditions. ", 0), ("Sensors, 22", I), ("(22), 8878. ", 0)], "https://doi.org/10.3390/s22228878"),
    ([("Chen, H., Tu, Z., Zhao, Y., & Ye, J. (2026). A high-resolution perspective-view road image dataset for pothole "
       "detection. ", 0), ("Scientific Data, 13", I), (", Artículo 961. ", 0)],
     "https://doi.org/10.1038/s41597-026-07317-w"),
    ([("Colaboradores de OpenStreetMap. (2026). ", 0), ("OpenStreetMap", I), (" [Base de datos]. ", 0)],
     "https://www.openstreetmap.org"),
    ([("dronefreak. (2026). ", 0), ("HRP4K", I), (" [Conjunto de datos]. Hugging Face. ", 0)],
     "https://huggingface.co/datasets/dronefreak/HRP4K"),
    ([("Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., & Crawford, K. (2021). "
       "Datasheets for datasets. ", 0), ("Communications of the ACM, 64", I), ("(12), 86–92. ", 0)],
     "https://doi.org/10.1145/3458723"),
    ([("Hanno100. (s. f.). ", 0), ("RoadDamageDetection-Egypt", I), (" [Conjunto de datos]. Hugging Face. ", 0)],
     "https://huggingface.co/datasets/Hanno100/RoadDamageDetection-Egypt"),
    ([("Krawetz, N. (2013, 21 de enero). Kind of like that. ", 0), ("The Hacker Factor Blog", I), (". ", 0)],
     "https://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html"),
    ([("Kumar, A., Raghunathan, A., Jones, R., Ma, T., & Liang, P. (2022). ", 0),
      ("Fine-tuning can distort pretrained features and underperform out-of-distribution", I),
      (" [Preimpresión]. arXiv. ", 0)], "https://doi.org/10.48550/arXiv.2202.10054"),
    ([("Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., & Zitnick, C. L. (2014). "
       "Microsoft COCO: Common objects in context. En ", 0), ("Computer Vision – ECCV 2014", I),
      (" (pp. 740–755). Springer. ", 0)], "https://doi.org/10.1007/978-3-319-10602-1_48"),
    ([("Ma, N., Fan, J., Wang, W., Wu, J., Jiang, Y., Xie, L., & Fan, R. (2022). Computer vision for road imaging and "
       "pothole detection: A state-of-the-art review of systems and algorithms. ", 0),
      ("Transportation Safety and Environment, 4", I), ("(4), Artículo tdac026. ", 0)], "https://doi.org/10.1093/tse/tdac026"),
    ([("Maeda, H., Sekimoto, Y., Seto, T., Kashiyama, T., & Omata, H. (2018). Road damage detection and classification "
       "using deep neural networks with smartphone images. ", 0),
      ("Computer-Aided Civil and Infrastructure Engineering, 33", I), ("(12), 1127–1141. ", 0)],
     "https://doi.org/10.1111/mice.12387"),
    ([("MapLibre. (2026). ", 0), ("MapLibre GL JS", I), (" (versión 6.19.0) [Software]. GitHub. ", 0)],
     "https://github.com/maplibre/maplibre-gl-js"),
    ([("Microsoft. (2026). ", 0), ("ONNX Runtime", I), (" (versión 1.31.1) [Software]. ", 0)], "https://onnxruntime.ai"),
    ([("Padilla, R., Passos, W. L., Dias, T. L. B., Netto, S. L., & da Silva, E. A. B. (2021). A comparative analysis of "
       "object detection metrics with a companion open-source toolkit. ", 0), ("Electronics, 10", I), ("(3), 279. ", 0)],
     "https://doi.org/10.3390/electronics10030279"),
    ([("Passos, B. T. (2020). ", 0), ("Cracks and potholes in road images", I),
      (" (versión 4) [Conjunto de datos]. Mendeley Data. ", 0)], "https://doi.org/10.17632/t576ydh9v8.4"),
    ([("Sahoo, R. R., Mohanty, S. S., & Mishra, S. (2025). ", 0),
      ("iWatchRoad: Scalable detection and geospatial visualization of potholes for smart cities", I),
      (" [Preimpresión]. arXiv. ", 0)], "https://doi.org/10.48550/arXiv.2508.10945"),
    ([("TamAko783. (s. f.). ", 0), ("YOLO26s_RDD_FRDC_Distilled_v2", I),
      (" [Modelo de aprendizaje automático]. Hugging Face. ", 0)],
     "https://huggingface.co/TamAko783/YOLO26s_RDD_FRDC_Distilled_v2"),
    ([("Ultralytics. (2026). ", 0), ("Ultralytics YOLO", I), (" (versión 8.4.191) [Software]. GitHub. ", 0)],
     "https://github.com/ultralytics/ultralytics"),
    ([("Wortsman, M., Ilharco, G., Gadre, S. Y., Roelofs, R., Gontijo-Lopes, R., Morcos, A. S., Namkoong, H., Farhadi, "
       "A., Carmon, Y., Kornblith, S., & Schmidt, L. (2022). ", 0),
      ("Model soups: Averaging weights of multiple fine-tuned models improves accuracy without increasing inference "
       "time", I), (" [Preimpresión]. arXiv. ", 0)], "https://doi.org/10.48550/arXiv.2203.05482"),
    ([("Wortsman, M., Ilharco, G., Kim, J. W., Li, M., Kornblith, S., Roelofs, R., Gontijo-Lopes, R., Hajishirzi, H., "
       "Farhadi, A., Namkoong, H., & Schmidt, L. (2021). ", 0), ("Robust fine-tuning of zero-shot models", I),
      (" [Preimpresión]. arXiv. ", 0)], "https://doi.org/10.48550/arXiv.2109.01903"),
]
run_proto = copy.deepcopy(P_REF.find(qn("w:r")))
rpr0 = run_proto.find(qn("w:rPr"))
if rpr0 is not None:
    for i_ in rpr0.findall(qn("w:i")):
        rpr0.remove(i_)
link_proto = copy.deepcopy(P_REF.find(qn("w:hyperlink")))
ref_h = buscar("Referencias")
borrar_entre(ref_h, buscar("Sobre el autor"))
ancla = ref_h._element
for segmentos, url in REFS:
    e = copy.deepcopy(P_REF)
    for hijo in list(e):
        if hijo.tag != qn("w:pPr"):
            e.remove(hijo)
    for texto, cursiva in segmentos:
        r = copy.deepcopy(run_proto)
        for t in r.findall(qn("w:t")):
            r.remove(t)
        rpr = r.find(qn("w:rPr"))
        if cursiva:
            if rpr is None:
                rpr = r.makeelement(qn("w:rPr"), {})
                r.insert(0, rpr)
            rpr.append(rpr.makeelement(qn("w:i"), {}))
        t = r.makeelement(qn("w:t"), {})
        t.text = texto
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        r.append(t)
        e.append(r)
    h = copy.deepcopy(link_proto)
    h.set(qn("r:id"), d.part.relate_to(url, RT.HYPERLINK, is_external=True))
    hr = h.findall(qn("w:r"))
    for extra in hr[1:]:
        h.remove(extra)
    ts = hr[0].findall(qn("w:t"))
    ts[0].text = url
    for extra in ts[1:]:
        hr[0].remove(extra)
    e.append(h)
    ancla.addnext(e)
    ancla = e

# ---------------------------------------------------------------- autores y declaraciones
poner(buscar("Sobre el autor"), "Sobre los autores")
autor = buscar("Leonardo Duran Cuenca es estudiante")
poner(autor,
      "Leonardo Duran Cuenca es estudiante de Ingeniería en Inteligencia Artificial de la Universidad Católica "
      "Boliviana “San Pablo”. Su experiencia en investigación comprende un estudio del clima de Santa Cruz de la Sierra "
      "de los últimos 26 años (Adquisición, Procesamiento y Análisis de Datos, 2026), con datos históricos de "
      "Open-Meteo, y un prototipo de detección de baches con inteligencia artificial (Inteligencia Artificial, primer "
      "parcial, 2026).")
Cursor(autor).parrafo(f"{COAUTORA} es estudiante de {CARRERA_COAUTORA} de la Universidad Católica Boliviana "
                      "“San Pablo”.")
poner(buscar("Con recursos propios"), "Con recursos propios. El cómputo en GPU se realizó con créditos gratuitos de AMD "
                                       "Developer Cloud.")
poner(buscar("Declara no tener"), "Declaran no tener ningún conflicto de intereses, que puedan haber influido en los "
                                   "resultados obtenidos o las interpretaciones propuestas.")
poner(buscar("El estudio se realizó respetando el Código"),
      "El estudio se realizó respetando el Código de ética y buenas prácticas editoriales de publicación. Los cinco "
      "videos de Santa Cruz de la Sierra fueron grabados con celular por terceros y cedidos a los autores para este "
      "estudio; como muestran personas y placas de vehículos, en el artículo solo aparecen imágenes reducidas o "
      "recortes del pavimento.")
copy_p = buscar("Copyright")
for t in copy_p._element.iter(qn("w:t")):
    if "Leonardo Duran Cuenca" in t.text:
        t.text = t.text.replace("Leonardo Duran Cuenca", f"Leonardo Duran Cuenca y {COAUTORA}")
        break
else:
    raise SystemExit("no encontré el nombre en los derechos de uso")

# ---------------------------------------------------------------- metadatos y partes sobrantes de la plantilla
cp = d.core_properties
cp.title = "Cartografía de baches y grietas con inteligencia artificial en el teléfono móvil"
cp.subject = "Artículo científico, formato Revista 3i"
cp.author = f"Leonardo Duran Cuenca; {COAUTORA}"
cp.last_modified_by = "Leonardo Duran Cuenca"
cp.keywords = "inteligencia artificial, cartografía, teléfono móvil"
cp.comments = cp.category = cp.content_status = ""
cp.language = "es-BO"
cp.created = cp.modified = datetime.now(timezone.utc).replace(microsecond=0, tzinfo=None)
cp.revision = 1
for viejo in cp._element.findall(qn("cp:lastPrinted")):
    cp._element.remove(viejo)
# El gráfico de ejemplo, imágenes que la plantilla ya no muestra y su lista vacía de fuentes bibliográficas: partes
# relacionadas desde el documento que nada referencia (r:id, r:embed o r:link).
usados = set(d.part._element.xpath("//@r:id | //@r:embed | //@r:link"))
sobrantes = [r for r in d.part.rels.values() if not r.is_external and r.rId not in usados
             and r.reltype in (RT.IMAGE, RT.CHART, RT.CUSTOM_XML)]
for r in sobrantes:
    d.part.rels.pop(r.rId)
print("partes sobrantes quitadas:", sorted(str(r.target_part.partname) for r in sobrantes))
# docProps/app.xml trae las estadísticas del último guardado de la plantilla (12 páginas, 35 minutos de edición):
# se quitan, y Word las vuelve a calcular al abrir el archivo.
EXT = "{http://schemas.openxmlformats.org/officeDocument/2006/extended-properties}"
for parte in d.part.package.iter_parts():
    if str(parte.partname) == "/docProps/app.xml":
        app = etree.fromstring(parte.blob)
        for campo in ("TotalTime", "Pages", "Words", "Characters", "Lines", "Paragraphs", "CharactersWithSpaces",
                      "Company", "Manager", "HyperlinkBase"):
            for e in app.findall(EXT + campo):
                app.remove(e)
        parte._blob = etree.tostring(app, xml_declaration=True, encoding="UTF-8", standalone=True)

salida = DOCS / "articulo_2do_parcial.docx"
d.save(salida)
print(f"guardado: {salida}")
print(f"resumen: {len(RESUMEN.split())} palabras | abstract: {len(ABSTRACT.split())} | resumo: {len(RESUMO.split())}")
print(f"referencias: {len(REFS)} | tablas: {len(d.tables)} | figuras: {Cursor.n_figura} | imágenes: {len(d.inline_shapes)}")
