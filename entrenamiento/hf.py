"""Mueve el proyecto entre tu PC, un repositorio privado de Hugging Face y la máquina de GPU.

En tu PC, una vez, ANTES de prender la GPU (la subida desde casa es lo lento; así no se paga GPU esperando):
    python entrenamiento/hf.py paquete --repo TU_USUARIO/baches-scz
En la GPU, después de preparar_maquina.sh (para no repetir descargas e importación en otra sesión):
    python entrenamiento/hf.py datos --repo TU_USUARIO/baches-scz
En la GPU, al terminar cada corrida:
    python entrenamiento/hf.py subir --repo TU_USUARIO/baches-scz --runs a1
En tu PC, para traer los resultados (o en una GPU nueva, --que datos):
    python entrenamiento/hf.py bajar --repo TU_USUARIO/baches-scz --que resultados

Necesita un token de Hugging Face con permiso de escritura: `hf auth login`, o la variable HF_TOKEN.
Repositorio: paquete/{fotos,etiquetas,codigo}.tar, datos/preparados.tar, resultados/<corrida>/...
Las fotos van reducidas a 1280 px de lado mayor (como se entrena): el paquete pesa menos de la mitad.
"""
import argparse
import io
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comun  # noqa: E402

# "carpeta/**/*" y no "carpeta/**": en Python < 3.13, "**" al final solo devuelve carpetas.
CODIGO = ["*.py", "splits.csv", "requirements.txt", "README.md", "informes/**/*", "entrenamiento/**/*",
          "tests/**/*", "modelos/**/*"]


def api():
    from huggingface_hub import HfApi
    return HfApi()


def armar_paquete(raiz, carpeta):
    carpeta.mkdir(parents=True, exist_ok=True)
    fotos = sorted((raiz / "fotos").glob("*.jpg"))
    t0, reducidas = time.time(), 0
    with tarfile.open(carpeta / "fotos.tar", "w") as tar:
        for i, p in enumerate(fotos):
            datos = p.read_bytes()
            salida, w, h = comun.reducir(datos, comun.LADO, p.name)
            reducidas += salida is not datos
            info = tarfile.TarInfo(f"fotos/{p.name}")
            info.size, info.mtime = len(salida), int(p.stat().st_mtime)
            tar.addfile(info, io.BytesIO(salida))
            if (i + 1) % 2000 == 0:
                print(f"  fotos {i + 1}/{len(fotos)}  {time.time() - t0:.0f} s", flush=True)
        for extra in (raiz / "fotos").glob("*.txt"):
            tar.add(extra, arcname=f"fotos/{extra.name}")
    print(f"fotos.tar: {len(fotos)} fotos ({reducidas} reducidas a {comun.LADO} px)")
    with tarfile.open(carpeta / "etiquetas.tar", "w") as tar:
        tar.add(raiz / "etiquetas", arcname="etiquetas")
    with tarfile.open(carpeta / "codigo.tar", "w") as tar:
        vistos = set()
        for patron in CODIGO:
            for p in sorted(raiz.glob(patron)):
                if p.is_file() and "__pycache__" not in p.parts and p not in vistos:
                    vistos.add(p)
                    tar.add(p, arcname=p.relative_to(raiz).as_posix())
    for t in ("fotos.tar", "etiquetas.tar", "codigo.tar"):
        print(f"  {t}: {(carpeta / t).stat().st_size / 1e9:.2f} GB")


def subir_archivo(repo, ruta, en_repo):
    t0 = time.time()
    print(f"subiendo {ruta.name} ({ruta.stat().st_size / 1e9:.2f} GB) ...", flush=True)
    api().upload_file(path_or_fileobj=str(ruta), path_in_repo=en_repo, repo_id=repo, repo_type="dataset",
                      commit_message=f"{en_repo}")
    print(f"  listo en {(time.time() - t0) / 60:.1f} min")


def bajar(repo, que, destino):
    from huggingface_hub import snapshot_download
    patrones = {"paquete": ["paquete/*"], "datos": ["datos/*"], "resultados": ["resultados/**"]}[que]
    local = snapshot_download(repo, repo_type="dataset", local_dir=str(destino / "_hf"), allow_patterns=patrones)
    for tar in sorted(Path(local).glob(f"{que}/*.tar")):
        print(f"extrayendo {tar.name} en {destino} ...", flush=True)
        with tarfile.open(tar) as t:
            try:
                t.extractall(destino, filter="data")
            except TypeError:  # Python < 3.11.4 no tiene filter; el tar es nuestro
                t.extractall(destino)
    if que == "resultados":
        print(f"resultados en {Path(local) / 'resultados'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("accion", choices=["paquete", "datos", "subir", "bajar"])
    ap.add_argument("--repo", required=True, help="USUARIO/nombre (se crea privado si no existe)")
    ap.add_argument("--runs", nargs="*", default=[], help="corridas de runs/ a subir")
    ap.add_argument("--que", choices=["paquete", "datos", "resultados"], default="resultados")
    ap.add_argument("--solo-armar", action="store_true", help="paquete: armar los .tar sin subirlos")
    ap.add_argument("--raiz", help="raíz del proyecto (por defecto, este repo)")
    args = ap.parse_args()
    raiz = comun.raiz(args.raiz)

    if args.accion != "bajar" and not args.solo_armar:
        api().create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)

    if args.accion == "paquete":
        carpeta = raiz / "salidas" / "paquete"
        armar_paquete(raiz, carpeta)
        if not args.solo_armar:
            # Si se corta, volver a correrlo: los bloques ya subidos no se vuelven a mandar.
            for t in ("codigo.tar", "etiquetas.tar", "fotos.tar"):
                subir_archivo(args.repo, carpeta / t, f"paquete/{t}")
    elif args.accion == "datos":
        tar_path = raiz / "salidas" / "preparados.tar"
        tar_path.parent.mkdir(exist_ok=True)
        with tarfile.open(tar_path, "w") as tar:
            for nombre in ("fotos", "etiquetas", "splits.csv"):
                tar.add(raiz / nombre, arcname=nombre)
        subir_archivo(args.repo, tar_path, "datos/preparados.tar")
    elif args.accion == "subir":
        for run in args.runs:
            carpeta = raiz / "runs" / run
            print(f"subiendo runs/{run} (sin los checkpoints intermedios) ...", flush=True)
            api().upload_folder(repo_id=args.repo, repo_type="dataset", folder_path=str(carpeta),
                                path_in_repo=f"resultados/{run}", ignore_patterns=["weights/epoch*.pt", "*.cache"],
                                commit_message=f"resultados {run}")
        api().upload_folder(repo_id=args.repo, repo_type="dataset", folder_path=str(raiz / "informes"),
                            path_in_repo="resultados/informes", commit_message="informes")
        api().upload_file(path_or_fileobj=str(raiz / "splits.csv"), path_in_repo="resultados/splits.csv",
                          repo_id=args.repo, repo_type="dataset", commit_message="splits finales")
        print("listo")
    elif args.accion == "bajar":
        bajar(args.repo, args.que, raiz)


if __name__ == "__main__":
    main()
