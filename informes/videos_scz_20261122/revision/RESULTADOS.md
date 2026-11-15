# Revisión a ciegas de las marcas de bache en los videos de Santa Cruz

355 eventos de 5 modelos, juzgados sin saber de qué modelo era cada uno (ver `herramientas/revisar_eventos.py`). Hay 47 baches reales distintos entre todos los modelos.

| Modelo | Eventos | Bache | Otro daño | No es daño | Dudoso | Precisión (bache) | Precisión (cualquier daño) | Baches hallados | Cobertura |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO12s (referencia) | 36 | 12 | 14 | 3 | 7 | 41 % | 90 % | 10 | 21 % |
| B0 (pesos iniciales) | 51 | 18 | 11 | 7 | 15 | 50 % | 81 % | 18 | 38 % |
| Baches SCZ v1 (final) | 63 | 23 | 8 | 5 | 27 | 64 % | 86 % | 21 | 45 % |
| a1 (entrenado) | 110 | 25 | 29 | 23 | 33 | 32 % | 70 % | 21 | 45 % |
| a2 (con HRP4K) | 95 | 22 | 27 | 22 | 24 | 31 % | 69 % | 22 | 47 % |

- **Precisión:** de los eventos juzgados (sin los dudosos), cuántos son un bache (o cualquier daño).
- **Cobertura:** de los baches reales distintos que encontró al menos un modelo, cuántos encontró este. Es una cota: un bache que no marcó ningún modelo no cuenta.
- Juzga un solo revisor, con video de WhatsApp de 848 × 478. Las hojas con cada juicio quedan en salidas/revision_scz/juzgadas/.
