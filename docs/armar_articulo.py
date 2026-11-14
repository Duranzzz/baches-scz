"""Arma el artículo del 2do parcial (formato Revista 3i) sobre docs/plantilla_articulo_2do_parcial.docx.
Todas las cifras salen de informes/ (entrenamiento_amd.md, eval_*.json, corridas/ y videos_scz_20261122/).
Antes: python docs/figuras/figura1_wise.py y python docs/figuras/generar_figuras.py
    python docs/armar_articulo.py
Deja docs/articulo_2do_parcial.docx."""
import copy
import json
from pathlib import Path

from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from docx.text.paragraph import Paragraph

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
        p = self._poner(copy.deepcopy(proto))
        poner(p, texto)
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
a2_01 = videos["scz_01_lateral_auto_hormigon"]["modelos"]["a2_y26s_hrp4k"]

# ---------------------------------------------------------------- título, autores y resúmenes
poner(buscar("[Título del manuscrito"), "Cartografía de baches y grietas con inteligencia artificial en el teléfono móvil")
poner(buscar("[Título traducido al inglés"),
      "Cartography of potholes and cracks with artificial intelligence on the mobile phone")
poner(buscar("[Título em português"), "Cartografia de buracos e fissuras com inteligência artificial no telefone celular")

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

RESUMEN = (
    "El objetivo es implementar y evaluar una aplicación web que detecta baches y grietas con inteligencia artificial "
    "en el teléfono móvil y registra cada daño como un punto en la cartografía de Santa Cruz de la Sierra. Se "
    "recopilaron 12.204 fotografías de seis conjuntos públicos de daños viales, se depuraron y, con 6.507 de "
    "entrenamiento, se entrenó en una GPU AMD MI300X un detector YOLO26s; el modelo final promedia sus pesos con los "
    "iniciales en una proporción elegida con una regla fijada antes de conocer los resultados. La evaluación usa una "
    "prueba congelada de 2.721 fotografías y cinco videos de la ciudad sin etiquetas. El modelo final obtiene un AP50 "
    "de 0,276 en baches, frente a 0,128 del detector que usaba la aplicación, y de 0,472 frente a 0,151 en el conjunto "
    "más limpio; el modelo entrenado sin promediar llega a 0,496, pero baja de 0,048 a 0,017 en un conjunto que no "
    "vio. La aplicación procesa entre 8 y 12 cuadros por segundo en una laptop sin tarjeta gráfica dedicada y guarda "
    "los puntos en el dispositivo. En conclusión, el promedio de pesos mejora la detección sin perder robustez, aunque "
    "la precisión en calles cruceñas no está medida.")
ABSTRACT = (
    "The objective is to implement and evaluate a web application that detects potholes and cracks with artificial "
    "intelligence on the mobile phone and records each damage as a point on the cartography of Santa Cruz de la Sierra. "
    "We collected 12,204 photographs from six public road damage datasets, curated them and, with 6,507 training "
    "photographs, trained a YOLO26s detector on an AMD MI300X GPU; the final model averages its weights with the "
    "initial ones in a proportion chosen with a rule fixed before the results were known. The evaluation uses a "
    "frozen test set of 2,721 photographs and five unlabeled videos of the city. The final model obtains an AP50 of "
    "0.276 for potholes, compared with 0.128 for the detector the application used, and 0.472 compared with 0.151 on "
    "the cleanest dataset; the model trained without averaging reaches 0.496 but drops from 0.048 to 0.017 on a "
    "dataset it did not see. The application processes between 8 and 12 frames per second on a laptop without a "
    "dedicated graphics card and stores the points on the device. In conclusion, weight averaging improves detection "
    "without losing robustness, although precision on the streets of Santa Cruz has not been measured.")
RESUMO = (
    "O objetivo é implementar e avaliar uma aplicação web que detecta buracos e fissuras com inteligência artificial "
    "no telefone celular e registra cada dano como um ponto na cartografia de Santa Cruz de la Sierra. Foram reunidas "
    "12.204 fotografias de seis conjuntos públicos de danos viários, que foram depuradas e, com 6.507 de treinamento, "
    "foi treinado em uma GPU AMD MI300X um detector YOLO26s; o modelo final faz a média de seus pesos com os iniciais "
    "em uma proporção escolhida com uma regra fixada antes de conhecer os resultados. A avaliação usa um teste "
    "congelado de 2.721 fotografias e cinco vídeos da cidade sem rótulos. O modelo final obtém AP50 de 0,276 em "
    "buracos, frente a 0,128 do detector que a aplicação usava, e de 0,472 frente a 0,151 no conjunto mais limpo; o "
    "modelo treinado sem média chega a 0,496, mas cai de 0,048 para 0,017 em um conjunto que não viu. A aplicação "
    "processa entre 8 e 12 quadros por segundo em um laptop sem placa de vídeo e guarda os pontos no dispositivo. Em "
    "conclusão, a média de pesos melhora a detecção sem perder robustez, embora a precisão nas ruas de Santa Cruz "
    "não tenha sido medida.")
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
for t in [
    "Para conservar la robustez de los pesos iniciales, se promedian con los ajustados en proporciones α del ajustado "
    "de 0,25, 0,50 y 0,75. La regla de selección se fija antes de ver estos resultados: HRP4K, que no participa del "
    "ajuste, se divide en dos mitades fijas de 467 y 432 fotografías, una para elegir y otra para informar, y a 640 "
    "píxeles, el tamaño de la aplicación, se elige el modelo de mayor AP50 de validación entre los que no empeoran más "
    "de 0,005 en la primera mitad. El modelo final conserva las cuatro clases iniciales: las filas de grietas quedan "
    "intactas y solo se promedia la de baches.",
    "Una detección es correcta si su caja coincide al menos en un 50 % con una etiqueta de la misma clase, medido como "
    "intersección sobre unión; con esos emparejamientos se calcula el AP50 con 101 puntos de exhaustividad. Se "
    "comparan el modelo final, el entrenado sin promediar, la segunda corrida, los promedios con otras proporciones, "
    "los pesos iniciales y el detector que usaba la aplicación, como referencia. Los umbrales de confianza de la "
    "aplicación se eligen en validación por F1, y la prueba se mide una sola vez, salvo la mitad de HRP4K destinada a "
    "elegir.",
    "La aplicación es una página web instalable que ejecuta el modelo, exportado a ONNX, dentro del teléfono: en su GPU "
    "mediante WebGPU o, si no la hay, en su CPU. Un daño se registra cuando aparece en 2 de 3 cuadros seguidos y se "
    "dibuja como un punto en un mapa de OpenStreetMap, en la posición del GPS o en la de un recorrido de prueba que "
    "simula un vehículo sobre una ruta real. La severidad se estima por el área de la caja. Los puntos quedan en el "
    "dispositivo y se exportan en GeoJSON o CSV.",
    "Se analizan cinco videos de calles de Santa Cruz de la Sierra grabados con celular y enviados por "
    "WhatsApp, con 10.162 cuadros de unos 848 × 478 píxeles: dos con la cámara al frente desde un auto y tres al costado "
    "de un auto o de un micro. Cada modelo procesa todos los cuadros a 864 × 512 píxeles con sus umbrales. Como no "
    "tienen etiquetas, se cuentan los cuadros con marca y los eventos, tramos de al menos 3 cuadros con marca, y se "
    "revisan a ojo las marcas de mayor confianza.",
]:
    c.parrafo(t)
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
c.figura("mAP50 de baches en validación durante el entrenamiento", "figura_curvas.png", 16.0,
         "Valor que calcula la biblioteca de entrenamiento en cada época. Elaboración propia.")

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
c.figura("AP50 de baches según la proporción del modelo entrenado en el promedio de pesos", "figura1_wise.png", 16.0,
         "A 640 píxeles. La franja marca el modelo elegido y la línea punteada, el límite de la regla de selección. "
         "Elaboración propia.")

c.sub("Modelo final")
n_f4 = Cursor.n_figura + 1
c.parrafo(
    f"El modelo final mejora frente al detector de referencia y a los pesos iniciales en todos los conjuntos, salvo en "
    f"HRP4K, donde queda entre ambos (figura {n_f4}): en el total de la prueba pasa de 0,128 y 0,196 a 0,276, y en "
    "Egipto, de 0,151 y 0,356 a 0,472. También mantiene la detección de grietas, con un AP50 de 0,199 en Egipto y de "
    "0,063 en Brasil, frente a 0,114 y 0,070 de los pesos iniciales y 0,075 y 0,061 del detector de referencia.")
c.figura("AP50 de baches por conjunto en la prueba congelada", "figura_barras.png", 16.0,
         "A 640 píxeles. Elaboración propia con los resultados de la prueba congelada.")
n_f5 = Cursor.n_figura + 1
c.parrafo(
    f"Los umbrales de la aplicación se eligen en validación (figura {n_f5}). Para baches, el F1 cae a medida que sube "
    "el umbral, y se usa 0,10, el mínimo que admite la aplicación, con un F1 de 0,338. Para grietas, el F1 apenas "
    "cambia entre 0,10 y 0,25, de modo que se usa 0,25, el valor más alto de ese tramo, para reducir las falsas "
    "alarmas.")
c.figura("F1 en validación según el umbral de confianza del modelo final", "figura_umbral.png", 16.0,
         "Media del F1 de los conjuntos con cada clase, con el detector solo y a 640 píxeles. Elaboración propia.")

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

c.sub("Discusión")
for t in [
    "La caída del modelo entrenado en HRP4K coincide con Kumar et al. (2022), para quienes ajustar todos los pesos "
    "deforma lo aprendido y empeora el desempeño fuera de la distribución del ajuste. El promedio de pesos recupera esa "
    "robustez sin entrenar de nuevo, como reportan Wortsman et al. (2021), y aquí se observa en detección de objetos. "
    "En Egipto, el conjunto de prueba más limpio, los promedios incluso superan al modelo entrenado: 0,471 con α = 0,25 "
    "y 0,534 con α = 0,50, frente a 0,442, en línea con la observación de que promediar pesos puede mejorar también "
    "dentro de la distribución (Wortsman et al., 2022).",
    "La ventaja de partir de pesos entrenados con imágenes de varios países concuerda con la propuesta de Arya et al. "
    "(2021). En HRP4K, con fotografías 4K en perspectiva (Chen et al., 2026), los baches lejanos ocupan pocos píxeles a "
    "640 y todos los modelos quedan por debajo de 0,05, de modo que las diferencias pequeñas deben leerse con cautela; "
    "a 1.280 píxeles los valores suben hasta 0,098. Una aplicación en el teléfono debe equilibrar ese tamaño con la "
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
    "Las limitaciones son la falta de fotografías etiquetadas de Santa Cruz de la Sierra, por lo que en los videos se "
    "cuentan marcas y no aciertos; el uso de una mitad de HRP4K para elegir; las 243 fotografías dudosas que siguen en "
    "el entrenamiento; las etiquetas de Brasil, que marcan solo el carril propio; la severidad, estimada por el área de "
    "la caja, y el error de varios metros del GPS del teléfono. Etiquetar fotografías de calles cruceñas, por ejemplo "
    "las de los puntos que la aplicación ya registra, permitiría medir la precisión local y entrenar con ese dominio.",
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
           "entrenamiento sin perder robustez, algo útil para una ciudad sin datos etiquetados propios. Sumar más "
           "fotografías de otra fuente, como en la segunda corrida, mejora esa fuente, pero no garantiza una mejora en "
           "las demás.")
cc.parrafo("La aplicación funciona como propuesta de un sistema de inspección: cualquier teléfono con navegador puede "
           "recorrer la ciudad, marcar daños y exportarlos a un sistema de información geográfica. Con fotografías "
           "locales etiquetadas, la misma regla de selección permitirá elegir el modelo directamente para las calles "
           "cruceñas.")

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

salida = DOCS / "articulo_2do_parcial.docx"
d.save(salida)
print(f"guardado: {salida}")
print(f"resumen: {len(RESUMEN.split())} palabras | abstract: {len(ABSTRACT.split())} | resumo: {len(RESUMO.split())}")
print(f"referencias: {len(REFS)} | tablas: {len(d.tables)} | figuras: {Cursor.n_figura} | imágenes: {len(d.inline_shapes)}")
