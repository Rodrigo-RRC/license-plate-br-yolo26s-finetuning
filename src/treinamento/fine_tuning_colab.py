# src/fine_tuning_colab.py
#
# Versão do fine-tuning adaptada para rodar no Google Colab com GPU T4.
#
# DIFERENÇAS em relação ao fine_tuning.py original:
#   - Os dados vêm do Google Drive (não do devcontainer local)
#   - O script descompacta o zip automaticamente na sessão do Colab
#   - O dataset.yaml é gerado em tempo de execução com os caminhos corretos
#   - batch=8 (em vez de 16) para não estourar a memória do T4 (16GB VRAM)
#
# ANTES DE RODAR — checklist obrigatório:
#   1. Faça upload do arquivo yj4Iu2-UFPR-ALPR.zip para o seu Google Drive
#   2. Faça upload do arquivo number-plate-yolo26s.pt para o seu Google Drive
#   3. No Colab: Runtime → Change runtime type → T4 GPU
#   4. Execute as células na ordem (se estiver num notebook .ipynb)
#      ou rode: !python fine_tuning_colab.py (se preferir como script)

import zipfile
from pathlib import Path

from ultralytics import YOLO


# =============================================================================
# PASSO 0: MONTAR O GOOGLE DRIVE
#
# Esta etapa conecta o Colab à sua conta do Google Drive.
# Uma janela de autorização será aberta — clique em "Permitir".
# O Drive ficará disponível em /content/drive/MyDrive/
# =============================================================================

def montar_drive():
    """
    Monta o Google Drive no ambiente Colab.

    Se o Drive já estiver montado (ex: montado manualmente antes de rodar
    o script), pula esta etapa silenciosamente.
    """
    if Path("/content/drive/MyDrive").exists():
        print("Google Drive já está montado em /content/drive/")
        return
    try:
        from google.colab import drive   # biblioteca exclusiva do Colab
        drive.mount("/content/drive")
        print("Google Drive montado com sucesso em /content/drive/")
    except Exception:
        print("AVISO: não foi possível montar o Drive automaticamente.")
        print("Monte manualmente com: from google.colab import drive; drive.mount('/content/drive')")


# =============================================================================
# CONFIGURAÇÕES
#
# Ajuste PASTA_DRIVE se você salvou os arquivos em uma subpasta diferente.
# Exemplo: se salvou em "Meu Drive/projetos/placas/", use:
#   PASTA_DRIVE = Path("/content/drive/MyDrive/projetos/placas")
# =============================================================================

# Pasta raiz do seu Google Drive onde você fez upload dos arquivos
PASTA_DRIVE = Path("/content/drive/MyDrive")

# Nome do arquivo zip do dataset (como está no seu Drive)
NOME_ZIP = "yj4Iu2-UFPR-ALPR.zip"

# Nome do arquivo do modelo pré-treinado (como está no seu Drive)
NOME_MODELO = "number-plate-yolo26s.pt"

# Onde o Colab vai descompactar o dataset (pasta temporária da sessão)
# /content/ é o disco local do Colab — rápido, mas some ao fechar a sessão
PASTA_DATASET_COLAB = Path("/content/UFPR-ALPR dataset")

# Onde vamos montar a estrutura YOLO (images/train, labels/train, etc.)
PASTA_YOLO = Path("/content/dataset")

# Caminho completo para o modelo no Drive
CAMINHO_MODELO = PASTA_DRIVE / NOME_MODELO

# Caminho completo para o zip no Drive
CAMINHO_ZIP = PASTA_DRIVE / NOME_ZIP

# Onde salvar o dataset.yaml gerado dinamicamente
CAMINHO_YAML = Path("/content/dataset.yaml")


# =============================================================================
# PARÂMETROS DO FINE-TUNING
# =============================================================================

EPOCHS          = 50    # voltas no dataset de treino
BATCH           = 8     # reduzido de 16 para 8 — T4 tem 16GB, mas imgsz=1280 é pesado
TAMANHO_IMAGEM  = 1280  # maior resolução → mais detalhe para placas de moto
LEARNING_RATE   = 0.001 # menor que o treino original (0.01) — fine-tuning precisa de passos suaves
PATIENCE        = 10    # para se não melhorar em 10 epochs consecutivas
FREEZE          = 10    # congela as 10 primeiras camadas do backbone
PASTA_RESULTADOS = "/content/runs/fine_tuning"
NOME_EXPERIMENTO = "placas_brasileiras_colab_v1"


# =============================================================================
# PASSO 1: DESCOMPACTAR O DATASET
# =============================================================================

def descompactar_dataset():
    """
    Descompacta o zip do UFPR-ALPR do Google Drive para o disco local do Colab.

    Por que copiar para o disco local em vez de usar direto do Drive?
    O Google Drive é acessado via rede — lento para ler milhares de arquivos
    pequenos. O disco local do Colab (/content/) é muito mais rápido para I/O
    intenso, como o carregamento de imagens durante o treino.

    O zip ocupa ~9GB; a descompactação leva alguns minutos.
    """
    if PASTA_DATASET_COLAB.exists():
        print(f"Dataset já descompactado em {PASTA_DATASET_COLAB}. Pulando.")
        return

    print(f"Descompactando {CAMINHO_ZIP} → {PASTA_DATASET_COLAB.parent} ...")
    print("(isso pode levar alguns minutos — o zip tem ~9GB)\n")

    if not CAMINHO_ZIP.exists():
        raise FileNotFoundError(
            f"Zip não encontrado em {CAMINHO_ZIP}.\n"
            f"Certifique-se de ter feito upload do arquivo para o Google Drive "
            f"na pasta {PASTA_DRIVE}."
        )

    with zipfile.ZipFile(CAMINHO_ZIP, "r") as zf:
        zf.extractall(PASTA_DATASET_COLAB.parent)

    print(f"Dataset descompactado com sucesso em {PASTA_DATASET_COLAB}\n")


# =============================================================================
# PASSO 2: CONVERTER ANOTAÇÕES E MONTAR ESTRUTURA YOLO
# =============================================================================

def preparar_estrutura_yolo():
    """
    Converte as anotações UFPR para o formato YOLO e organiza as pastas.

    Reutiliza a mesma lógica do script preparar_dataset.py — apenas com
    os caminhos ajustados para o ambiente Colab.

    A estrutura gerada é:
        /content/dataset/
            images/train/   ← 1.800 imagens de treino
            images/val/     ← 900 imagens de validação
            images/test/    ← 1.800 imagens de teste
            labels/train/   ← labels YOLO de treino
            labels/val/     ← labels YOLO de validação
            labels/test/    ← labels YOLO de teste
    """
    import shutil

    LARGURA = 1920
    ALTURA  = 1080

    splits = {
        "training":   "train",
        "validation": "val",
        "testing":    "test",
    }

    if PASTA_YOLO.exists():
        print(f"Estrutura YOLO já existe em {PASTA_YOLO}. Pulando preparação.")
        return

    print("Criando estrutura de pastas YOLO...")
    for split_yolo in splits.values():
        (PASTA_YOLO / "images" / split_yolo).mkdir(parents=True, exist_ok=True)
        (PASTA_YOLO / "labels" / split_yolo).mkdir(parents=True, exist_ok=True)

    for split_ufpr, split_yolo in splits.items():
        pasta_origem       = PASTA_DATASET_COLAB / split_ufpr
        pasta_imgs_destino = PASTA_YOLO / "images" / split_yolo
        pasta_lbls_destino = PASTA_YOLO / "labels" / split_yolo

        contagem = 0
        for pasta_track in sorted(pasta_origem.iterdir()):
            if not pasta_track.is_dir():
                continue

            for arquivo_png in sorted(pasta_track.glob("*.png")):
                arquivo_txt = arquivo_png.with_suffix(".txt")
                if not arquivo_txt.exists():
                    continue

                # Copia a imagem
                shutil.copy(arquivo_png, pasta_imgs_destino / arquivo_png.name)

                # Converte a anotação UFPR → YOLO
                with open(arquivo_txt, encoding="utf-8") as f:
                    for linha in f:
                        if linha.startswith("corners:"):
                            numeros = linha.replace("corners:", "").strip()
                            pares   = numeros.split()
                            xs = [int(p.split(",")[0]) for p in pares]
                            ys = [int(p.split(",")[1]) for p in pares]
                            x_min, x_max = min(xs), max(xs)
                            y_min, y_max = min(ys), max(ys)
                            cx = (x_min + x_max) / 2 / LARGURA
                            cy = (y_min + y_max) / 2 / ALTURA
                            w  = (x_max - x_min) / LARGURA
                            h  = (y_max - y_min) / ALTURA
                            label_yolo = f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
                            break

                destino_label = pasta_lbls_destino / arquivo_png.with_suffix(".txt").name
                with open(destino_label, "w") as f:
                    f.write(label_yolo)

                contagem += 1

        print(f"  [{split_ufpr:>10}] → {contagem} imagens convertidas.")

    print(f"\nEstrutura YOLO pronta em {PASTA_YOLO}\n")


# =============================================================================
# PASSO 3: GERAR O dataset.yaml COM OS CAMINHOS DO COLAB
# =============================================================================

def gerar_dataset_yaml():
    """
    Cria o arquivo dataset.yaml com os caminhos corretos para o ambiente Colab.

    O YOLO usa este arquivo para saber onde estão as imagens de treino,
    validação e teste, e quais são as classes a detectar.

    Por que gerar dinamicamente em vez de usar o arquivo do repositório?
    O arquivo do repositório aponta para /workspaces/... (devcontainer local).
    No Colab, os dados estão em /content/... — caminhos completamente diferentes.
    """
    if CAMINHO_YAML.exists():
        print(f"dataset.yaml já existe em {CAMINHO_YAML}. Pulando.")
        return

    conteudo = f"""# dataset.yaml — gerado automaticamente para o ambiente Google Colab
# Caminhos válidos apenas durante a sessão ativa do Colab.

path: {PASTA_YOLO}

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
# PASSO 4: FINE-TUNING
# =============================================================================

def main():
    """
    Executa todas as etapas em sequência:
      0. Monta o Google Drive
      1. Descompacta o dataset
      2. Prepara a estrutura YOLO
      3. Gera o dataset.yaml
      4. Roda o fine-tuning
    """
    # ── Etapa 0 ───────────────────────────────────────────────────────────────
    montar_drive()

    # ── Etapa 1 ───────────────────────────────────────────────────────────────
    descompactar_dataset()

    # ── Etapa 2 ───────────────────────────────────────────────────────────────
    preparar_estrutura_yolo()

    # ── Etapa 3 ───────────────────────────────────────────────────────────────
    gerar_dataset_yaml()

    # ── Etapa 4: Fine-tuning ───────────────────────────────────────────────────
    print("=" * 55)
    print("  INICIANDO FINE-TUNING (Google Colab / T4)")
    print("=" * 55)
    print(f"  Modelo base    : {CAMINHO_MODELO}")
    print(f"  Dataset        : {CAMINHO_YAML}")
    print(f"  Epochs         : {EPOCHS}")
    print(f"  Batch size     : {BATCH}  (reduzido para T4)")
    print(f"  imgsz          : {TAMANHO_IMAGEM}")
    print(f"  Learning rate  : {LEARNING_RATE}")
    print(f"  Freeze layers  : {FREEZE}")
    print(f"  Early stopping : {PATIENCE} epochs sem melhora")
    print(f"  Resultados em  : {PASTA_RESULTADOS}/{NOME_EXPERIMENTO}")
    print("=" * 55 + "\n")

    modelo = YOLO(str(CAMINHO_MODELO))

    modelo.train(
        data=str(CAMINHO_YAML),
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=TAMANHO_IMAGEM,
        lr0=LEARNING_RATE,
        cos_lr=True,           # decaimento suave de LR ao longo do treino
        patience=PATIENCE,
        freeze=FREEZE,         # congela backbone — treina só a cabeça
        degrees=10,            # rotação: ajuda com placas em ângulo (motos)
        perspective=0.001,     # perspectiva: variações angulares reais
        project=PASTA_RESULTADOS,
        name=NOME_EXPERIMENTO,
        exist_ok=True,
        verbose=True,
    )

    caminho_best = f"{PASTA_RESULTADOS}/{NOME_EXPERIMENTO}/weights/best.pt"

    print("\n" + "=" * 55)
    print("  TREINO CONCLUÍDO!")
    print("=" * 55)
    print(f"  Melhor modelo : {caminho_best}")
    print("\n  IMPORTANTE: Copie o best.pt para o Google Drive antes de")
    print("  fechar a sessão — o /content/ é apagado ao desconectar.")
    print(f"\n  No Colab, rode:")
    print(f"  import shutil")
    print(f"  shutil.copy('{caminho_best}', '{PASTA_DRIVE}/best_placas_v1.pt')")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
