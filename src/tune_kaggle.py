# src/tune_kaggle.py
#
# Versão do tune adaptada para o Kaggle Notebooks.
#
# DIFERENÇAS em relação ao tune_colab.py:
#   - Sem montagem de Drive — arquivos baixados via gdown direto do Google Drive
#   - Caminhos de entrada: /kaggle/working/ (após download com gdown)
#   - Caminhos de saída:   /kaggle/working/
#
# ANTES DE RODAR — checklist obrigatório:
#
#   1. No notebook Kaggle:
#      - Settings → "Internet" → ativar (necessário para gdown e ultralytics)
#      - Settings → "Accelerator" → GPU T4 x2
#
#   2. Instalar dependências e baixar arquivos (primeira célula do notebook):
#
#      !pip install ultralytics gdown -q
#
#      import gdown
#      gdown.download(id="1rf4RdXoaGWCPjxfv5BGG-7mFV6otLWLO",
#                     output="/kaggle/working/yj4Iu2-UFPR-ALPR.zip", quiet=False)
#      gdown.download(id="1s9M8HgJQ0o4-BCnWQ3imFn_Rhsx8r-Di",
#                     output="/kaggle/working/number-plate-yolo26s.pt", quiet=False)
#
#   3. Fazer upload deste script e rodar:
#      !python /kaggle/working/tune_kaggle.py

import zipfile
import shutil
from pathlib import Path
from ultralytics import YOLO


# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

# Arquivos baixados via gdown para /kaggle/working/
CAMINHO_ZIP    = Path("/kaggle/working/yj4Iu2-UFPR-ALPR.zip")
CAMINHO_MODELO = Path("/kaggle/working/number-plate-yolo26s.pt")

# Kaggle salva outputs em /kaggle/working/ — persiste durante a sessão
PASTA_DATASET_KAGGLE = Path("/kaggle/working/UFPR-ALPR dataset")
PASTA_YOLO           = Path("/kaggle/working/dataset")
CAMINHO_YAML         = Path("/kaggle/working/dataset.yaml")
PASTA_RESULTADOS     = "/kaggle/working/runs/tune"
NOME_EXPERIMENTO     = "busca_hiperparametros_v1"

EPOCHS_POR_CANDIDATO = 3    # epochs por candidato
ITERACOES            = 20   # 20 candidatos → ~4 horas na T4


# =============================================================================
# ESPAÇO DE BUSCA (idêntico ao tune_colab.py)
# =============================================================================

ESPACO_DE_BUSCA = {
    "lr0":          (1e-4, 5e-3),
    "lrf":          (0.001, 0.1),
    "momentum":     (0.8, 0.98),
    "weight_decay": (0.0001, 0.001),
    "box":          (5.0, 15.0),
    "dfl":          (0.5, 3.0),
    "degrees":      (0.0, 20.0),
    "perspective":  (0.0, 0.001),
    "scale":        (0.3, 0.7),
    "mosaic":       (0.5, 1.0),
    "fliplr":       (0.0, 0.5),
    "hsv_v":        (0.2, 0.6),
}


# =============================================================================
# PREPARAÇÃO DO DATASET
# =============================================================================

def descompactar_dataset():
    """Descompacta o zip do dataset UFPR-ALPR para o disco local do Kaggle."""
    if PASTA_DATASET_KAGGLE.exists():
        print("Dataset já descompactado. Pulando.")
        return
    if not CAMINHO_ZIP.exists():
        raise FileNotFoundError(
            f"Zip não encontrado em {CAMINHO_ZIP}\n"
            "Execute o gdown na célula anterior para baixar o dataset do Drive."
        )
    print(f"Descompactando {CAMINHO_ZIP} (pode levar alguns minutos)...")
    with zipfile.ZipFile(CAMINHO_ZIP, "r") as zf:
        zf.extractall(PASTA_DATASET_KAGGLE.parent)
    print(f"Dataset pronto em {PASTA_DATASET_KAGGLE}\n")


def preparar_estrutura_yolo():
    """Converte anotações UFPR → formato YOLO e organiza as pastas."""
    if PASTA_YOLO.exists():
        print("Estrutura YOLO já existe. Pulando.")
        return

    LARGURA, ALTURA = 1920, 1080
    splits = {"training": "train", "validation": "val", "testing": "test"}

    print("Criando estrutura de pastas YOLO...")
    for split_yolo in splits.values():
        (PASTA_YOLO / "images" / split_yolo).mkdir(parents=True, exist_ok=True)
        (PASTA_YOLO / "labels" / split_yolo).mkdir(parents=True, exist_ok=True)

    for split_ufpr, split_yolo in splits.items():
        pasta_imgs = PASTA_YOLO / "images" / split_yolo
        pasta_lbls = PASTA_YOLO / "labels" / split_yolo
        contagem = 0
        for pasta_track in sorted((PASTA_DATASET_KAGGLE / split_ufpr).iterdir()):
            if not pasta_track.is_dir():
                continue
            for png in sorted(pasta_track.glob("*.png")):
                txt = png.with_suffix(".txt")
                if not txt.exists():
                    continue
                shutil.copy(png, pasta_imgs / png.name)
                with open(txt, encoding="utf-8") as f:
                    for linha in f:
                        if linha.startswith("corners:"):
                            pares = linha.replace("corners:", "").strip().split()
                            xs = [int(p.split(",")[0]) for p in pares]
                            ys = [int(p.split(",")[1]) for p in pares]
                            cx = (min(xs) + max(xs)) / 2 / LARGURA
                            cy = (min(ys) + max(ys)) / 2 / ALTURA
                            w  = (max(xs) - min(xs)) / LARGURA
                            h  = (max(ys) - min(ys)) / ALTURA
                            label = f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
                            break
                with open(pasta_lbls / png.with_suffix(".txt").name, "w") as f:
                    f.write(label)
                contagem += 1
        print(f"  [{split_ufpr:>10}] → {contagem} imagens convertidas.")
    print(f"\nEstrutura YOLO pronta em {PASTA_YOLO}\n")


def gerar_dataset_yaml():
    """Gera o dataset.yaml com os caminhos do Kaggle."""
    if CAMINHO_YAML.exists():
        print("dataset.yaml já existe. Pulando.")
        return
    conteudo = f"""path: {PASTA_YOLO}
train: images/train
val:   images/val
test:  images/test
nc: 1
names:
  0: license_plate
"""
    CAMINHO_YAML.write_text(conteudo, encoding="utf-8")
    print(f"dataset.yaml gerado em {CAMINHO_YAML}\n")


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def main():
    """Prepara o dataset e executa a busca genética de hiperparâmetros."""

    # ── Preparação ────────────────────────────────────────────────────────────
    descompactar_dataset()
    preparar_estrutura_yolo()
    gerar_dataset_yaml()

    # ── Busca ─────────────────────────────────────────────────────────────────
    estimativa_h = (EPOCHS_POR_CANDIDATO * ITERACOES * 4.5) / 60

    print("=" * 60)
    print("  BUSCA DE HIPERPARÂMETROS — EVOLUÇÃO GENÉTICA (Kaggle)")
    print("=" * 60)
    print(f"  Modelo base      : {CAMINHO_MODELO}")
    print(f"  Dataset          : {CAMINHO_YAML}")
    print(f"  Epochs/candidato : {EPOCHS_POR_CANDIDATO}")
    print(f"  Iterações        : {ITERACOES}")
    print(f"  Parâmetros       : {len(ESPACO_DE_BUSCA)}")
    print(f"  Estimativa       : ~{estimativa_h:.0f} horas na T4")
    print(f"  Resultados em    : {PASTA_RESULTADOS}/{NOME_EXPERIMENTO}")
    print("=" * 60 + "\n")

    modelo = YOLO(str(CAMINHO_MODELO))

    modelo.tune(
        data=str(CAMINHO_YAML),
        epochs=EPOCHS_POR_CANDIDATO,
        iterations=ITERACOES,
        imgsz=1280,
        batch=8,
        optimizer="AdamW",
        freeze=10,
        space=ESPACO_DE_BUSCA,
        project=PASTA_RESULTADOS,
        name=NOME_EXPERIMENTO,
        exist_ok=True,
        plots=True,
        save=False,
        val=True,
    )

    caminho_yaml = f"{PASTA_RESULTADOS}/{NOME_EXPERIMENTO}/best_hyperparameters.yaml"

    print("\n" + "=" * 60)
    print("  BUSCA CONCLUÍDA!")
    print("=" * 60)
    print(f"  Melhores hiperparâmetros: {caminho_yaml}")
    print()
    print("  No Kaggle os resultados ficam em /kaggle/working/")
    print("  Faça download do best_hyperparameters.yaml pelo painel")
    print("  Output → busca_hiperparametros_v1 → best_hyperparameters.yaml")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
