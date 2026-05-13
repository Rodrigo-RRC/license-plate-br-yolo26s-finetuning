
# O que falta fazer:: Preciso rodar este script dia 04/05/2026
#  Pedir ao Alex acessa a OCI para rodar na GPU
# Lembrar a Claude Code com esta mensagem: "Pode verificar sua memória sobre o projeto car_license_plate_detector?"
# Ou digitar isto: "Varra todo o repositório car_license_plate_detector e me diz onde paramos"

# src/fine_tuning.py
#
# O que este script faz:
#   Realiza o fine-tuning (ajuste fino) do modelo YOLO pré-treinado
#   usando o dataset UFPR-ALPR organizado na pasta "dataset/".
#
#   Fine-tuning significa: em vez de treinar do zero, partimos de um modelo
#   que já sabe detectar placas e ensinamos ele a ser melhor no nosso domínio
#   específico (placas brasileiras, motos, caminhões, diferentes condições).
#
#   O Ultralytics cuida de toda a complexidade do treino por baixo dos panos.
#   Nosso papel é configurar os parâmetros corretamente.

from ultralytics import YOLO   # framework que contém o modelo YOLO e o motor de treino


# =============================================================================
# CONFIGURAÇÕES DO FINE-TUNING
#
# Centralizar as configurações aqui (em vez de espalhá-las pelo código)
# facilita ajustes futuros sem precisar ler o script inteiro.
# =============================================================================

# Modelo de partida: nosso modelo pré-treinado
# O YOLO vai carregar os pesos existentes e continuar aprendendo a partir deles
MODELO_BASE = "number-plate-yolo26s.pt"

# Arquivo de configuração do dataset (criado na Etapa 2)
# Contém os caminhos das imagens e a lista de classes
DATASET_YAML = "/workspaces/car_license_plate_detector/dataset.yaml"

# Número de epochs (voltas completas pelo dataset de treino)
# 50 é um bom ponto de partida para fine-tuning — não muito pouco, não exagerado
# O early stopping vai interromper antes se o modelo parar de melhorar
EPOCHS = 50

# Batch size: quantas imagens o modelo processa de uma vez antes de atualizar os pesos
# 16 é um valor seguro para a maioria das máquinas sem GPU dedicada
# Se der erro de memória, reduza para 8
BATCH = 16

# Tamanho das imagens durante o treino (em pixels, sempre quadrado)
# 640x640 é o padrão do YOLO — as imagens são redimensionadas automaticamente
# As originais do UFPR são 1920x1080, mas o YOLO cuida dessa conversão
TAMANHO_IMAGEM = 640

# Learning rate inicial (taxa de aprendizado)
# Controla o tamanho do "passo" ao corrigir erros
# 0.001 é baixo — ideal para fine-tuning, para não apagar o conhecimento já adquirido
# No treino do zero usaríamos algo maior, como 0.01
LEARNING_RATE = 0.001

# Patience: quantas epochs sem melhora o YOLO aguenta antes de parar (early stopping)
# Exemplo: patience=10 significa que se o val não melhorar em 10 epochs seguidas, para
# Isso evita desperdício de tempo e overfitting
PATIENCE = 10

# Pasta onde os resultados serão salvos (gráficos, pesos, métricas)
# O YOLO cria automaticamente subpastas organizadas por execução
PASTA_RESULTADOS = "runs/fine_tuning"

# Nome desta execução — aparece como subpasta dentro de PASTA_RESULTADOS
# Útil para identificar experimentos diferentes (ex: "tentativa1", "com_augmentation")
NOME_EXPERIMENTO = "placas_brasileiras_v1"


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def main():
    """Executa o fine-tuning do modelo YOLO com o dataset UFPR-ALPR."""

    print("=" * 55)
    print("  INICIANDO FINE-TUNING")
    print("=" * 55)
    print(f"  Modelo base    : {MODELO_BASE}")
    print(f"  Dataset        : {DATASET_YAML}")
    print(f"  Epochs         : {EPOCHS}")
    print(f"  Batch size     : {BATCH}")
    print(f"  Learning rate  : {LEARNING_RATE}")
    print(f"  Early stopping : {PATIENCE} epochs sem melhora")
    print(f"  Resultados em  : {PASTA_RESULTADOS}/{NOME_EXPERIMENTO}")
    print("=" * 55 + "\n")

    # ── PASSO 1: Carregar o modelo pré-treinado ────────────────────────────────
    #
    # Esta linha carrega o arquivo .pt (PyTorch model) com todos os pesos
    # que o modelo aprendeu anteriormente. É aqui que o fine-tuning começa:
    # partimos de um modelo que já "sabe" detectar placas, não do zero.
    print("Carregando modelo pré-treinado...")
    modelo = YOLO(MODELO_BASE)

    # ── PASSO 2: Iniciar o treino ──────────────────────────────────────────────
    #
    # modelo.train() é o coração do script. O Ultralytics cuida de:
    #   - Carregar as imagens em batches
    #   - Calcular os erros (loss)
    #   - Atualizar os pesos do modelo
    #   - Avaliar no val ao final de cada epoch
    #   - Salvar o melhor modelo (best.pt) e o último (last.pt)
    #   - Aplicar early stopping se configurado
    #   - Gerar gráficos de evolução do treino
    print("Iniciando treino...\n")
    modelo.train(
        data=DATASET_YAML,           # onde estão os dados e as classes
        epochs=EPOCHS,               # quantas voltas no dataset
        batch=BATCH,                 # tamanho do lote
        imgsz=TAMANHO_IMAGEM,        # tamanho das imagens
        lr0=LEARNING_RATE,           # learning rate inicial
        patience=PATIENCE,           # early stopping
        project=PASTA_RESULTADOS,    # pasta raiz dos resultados
        name=NOME_EXPERIMENTO,       # nome desta execução
        exist_ok=True,               # se a pasta já existir, não dá erro
        verbose=True,                # mostra o progresso detalhado no terminal
    )

    # ── PASSO 3: Informar onde encontrar os resultados ─────────────────────────
    #
    # Ao final do treino, o YOLO salva automaticamente:
    #   best.pt  → o melhor modelo encontrado durante todo o treino
    #   last.pt  → o modelo do último epoch (pode ter sofrido overfitting)
    #
    # Sempre use o best.pt para inferência e avaliação final.
    caminho_best = f"{PASTA_RESULTADOS}/{NOME_EXPERIMENTO}/weights/best.pt"

    print("\n" + "=" * 55)
    print("  TREINO CONCLUÍDO!")
    print("=" * 55)
    print(f"  Melhor modelo salvo em:")
    print(f"  {caminho_best}")
    print("\n  Próximo passo: avaliar o modelo treinado com eval_por_tipo.py")
    print("=" * 55 + "\n")


# Ponto de entrada: só executa se rodar o script diretamente
if __name__ == "__main__":
    main()
