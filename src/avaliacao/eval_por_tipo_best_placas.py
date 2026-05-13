# src/eval_por_tipo.py
#
# O que este script faz:
#   Avalia o modelo YOLO em TODAS as 1.800 imagens de teste do UFPR-ALPR,
#   separando automaticamente os resultados por tipo de veículo (carro e moto).
#   Ao final, exibe uma tabela comparativa com as métricas de cada grupo.
#
#   Por que isso é útil?
#   Permite identificar se o modelo tem desempenho desigual entre tipos de veículo,
#   o que é uma informação essencial para decidir estratégias de fine-tuning.

from pathlib import Path          # manipulação de caminhos de arquivo sem concatenar strings manualmente

import torch                      # biblioteca de tensores, base do PyTorch
from torchmetrics.detection import IntersectionOverUnion
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from ultralytics import YOLO      # framework para carregar e rodar modelos YOLO


# =============================================================================
# CONSTANTES
# =============================================================================

# Caminho para a pasta de teste do dataset UFPR-ALPR
PASTA_TESTE = Path("/workspaces/car_license_plate_detector/UFPR-ALPR dataset/testing")

# Caminho para o modelo YOLO pré-treinado
CAMINHO_MODELO = "best_placas_v2.pt"


# =============================================================================
# FUNÇÃO: obter_tipo_veiculo
#
# Lê um arquivo .txt do UFPR e retorna o tipo do veículo como string.
#
# O campo "type:" pode conter, por exemplo:
#   "	type: car"  ou  "	type: motorcycle"
# Note que o campo vem com um TAB antes — por isso usamos strip() para limpar.
#
# Parâmetros:
#   caminho_txt (Path): caminho para o arquivo de anotação UFPR
#
# Retorna:
#   str: tipo do veículo ("car", "motorcycle", etc.) ou "desconhecido" se não encontrar
# =============================================================================

def obter_tipo_veiculo(caminho_txt: Path) -> str:
    """Lê o campo 'type:' do arquivo UFPR e retorna o tipo do veículo."""
    with open(caminho_txt, encoding="utf-8") as arquivo:
        for linha in arquivo:
            # strip() remove espaços e tabs do início e fim da linha
            if linha.strip().startswith("type:"):
                # Divide a linha em "type:" e o valor, pega só o valor e limpa espaços
                # Exemplo: "	type: motorcycle\n" → "motorcycle"
                return linha.split(":")[1].strip()
    return "desconhecido"


# =============================================================================
# FUNÇÃO: parse_ufpr_to_xyxy
#
# Lê o campo "corners:" de um arquivo .txt do UFPR e converte as coordenadas
# dos 4 cantos da placa para um bounding box no formato [x_min, y_min, x_max, y_max].
#
# O formato UFPR armazena os cantos assim:
#   corners: 856,504 919,504 919,525 856,525
#   (4 pares x,y representando os 4 vértices da placa)
#
# Parâmetros:
#   caminho_txt (Path): caminho para o arquivo de anotação UFPR
#
# Retorna:
#   list[int]: [x_min, y_min, x_max, y_max] em pixels
# =============================================================================

def parse_ufpr_to_xyxy(caminho_txt: Path) -> list[int]:
    """Extrai as coordenadas da placa do arquivo UFPR e retorna [x_min, y_min, x_max, y_max]."""
    with open(caminho_txt, encoding="utf-8") as arquivo:
        linhas = arquivo.readlines()

    # Procura a linha que começa com "corners:"
    linha_corners = ""
    for linha in linhas:
        if linha.startswith("corners:"):
            linha_corners = linha
            break

    # Remove o rótulo "corners:" e espaços extras
    # Resultado: "856,504 919,504 919,525 856,525"
    apenas_numeros = linha_corners.replace("corners:", "").strip()

    # Separa os pares pelo espaço → ["856,504", "919,504", "919,525", "856,525"]
    pares_xy = apenas_numeros.split()

    lista_x = []
    lista_y = []
    for par in pares_xy:
        x, y = par.split(",")          # separa pela vírgula
        lista_x.append(int(x))
        lista_y.append(int(y))

    # O menor e maior x e y formam o retângulo que engloba os 4 cantos
    return [min(lista_x), min(lista_y), max(lista_x), max(lista_y)]


# =============================================================================
# FUNÇÃO: criar_metricas
#
# Cria e retorna um par de objetos de métricas TorchMetrics prontos para uso.
# Separamos em função para não repetir o mesmo código ao criar métricas
# para cada tipo de veículo.
#
# Retorna:
#   tuple: (IntersectionOverUnion, MeanAveragePrecision)
# =============================================================================

def criar_metricas() -> tuple:
    """Instancia e retorna um par de métricas TorchMetrics (IoU e mAP)."""
    iou = IntersectionOverUnion()
    mapa = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    return iou, mapa


# =============================================================================
# FUNÇÃO: montar_pred_dict
#
# Recebe o resultado bruto do YOLO e monta o dicionário no formato
# que o TorchMetrics espera para registrar uma predição.
#
# Se o YOLO não detectou nenhuma placa, retorna um dicionário vazio —
# isso penaliza corretamente o modelo na métrica.
#
# Parâmetros:
#   caixas: objeto boxes retornado pelo YOLO (resultados[0].boxes)
#
# Retorna:
#   dict: dicionário com "boxes", "scores" e "labels"
# =============================================================================

def montar_pred_dict(caixas) -> dict:
    """Converte o resultado do YOLO para o formato de dicionário do TorchMetrics."""
    if len(caixas) == 0:
        # Nenhuma detecção: tensores vazios com as formas corretas
        return {
            "boxes":  torch.empty((0, 4)),
            "scores": torch.empty((0,)),
            "labels": torch.empty((0,), dtype=torch.int),
        }

    return {
        # xyxy[0:1] pega somente a detecção de maior confiança (a primeira)
        "boxes":  caixas.xyxy[0:1].cpu(),
        # conf[0:1] é o percentual de certeza do YOLO para essa detecção
        "scores": caixas.conf[0:1].cpu(),
        # classe 0 = "license_plate" (única classe neste projeto)
        "labels": torch.zeros(1, dtype=torch.int),
    }


# =============================================================================
# FUNÇÃO: imprimir_resultado
#
# Formata e imprime o bloco de resultados de um tipo de veículo.
#
# Parâmetros:
#   titulo   (str)  : título do bloco (ex: "CARROS")
#   iou_metric      : objeto IntersectionOverUnion já computado
#   map_metric      : objeto MeanAveragePrecision já computado
#   contagem (int)  : número de imagens avaliadas
# =============================================================================

def imprimir_resultado(titulo: str, iou_metric, map_metric, contagem: int) -> dict:
    """Calcula, imprime e retorna as métricas finais de um grupo de veículos."""
    res_iou = iou_metric.compute()
    res_map = map_metric.compute()

    iou       = res_iou["iou"].item()
    map50     = res_map["map_50"].item()
    map75     = res_map["map_75"].item()
    map_geral = res_map["map"].item()

    print(f"\n{'=' * 52}")
    print(f"  {titulo}  ({contagem} imagens)")
    print(f"{'=' * 52}")
    print(f"  IoU médio          : {iou:.4f}  ({iou * 100:.2f}%)")
    print(f"  {'─' * 48}")
    print(f"  mAP@50  (tolerante): {map50:.4f}  ({map50 * 100:.2f}%)")
    print(f"  mAP@75  (rigoroso) : {map75:.4f}  ({map75 * 100:.2f}%)")
    print(f"  mAP@50-95 (geral)  : {map_geral:.4f}  ({map_geral * 100:.2f}%)")
    print(f"{'=' * 52}")

    # Retorna os valores para usar na tabela comparativa final
    return {"iou": iou, "map50": map50, "map75": map75, "map_geral": map_geral}


# =============================================================================
# FUNÇÃO: imprimir_tabela_comparativa
#
# Imprime a tabela final lado a lado com os resultados de todos os grupos.
#
# Parâmetros:
#   resultados (dict): dicionário com os resultados de cada tipo de veículo
#                      Formato: {"carro": {...}, "moto": {...}, "geral": {...}}
# =============================================================================

def imprimir_tabela_comparativa(resultados: dict) -> None:
    """Exibe a tabela comparativa final com os resultados por tipo de veículo."""
    linha = f"{'─' * 54}"
    cabecalho = f"{'Tipo':<12} {'Imagens':>8}  {'IoU':>8}  {'mAP@50':>8}  {'mAP@50-95':>10}"

    print(f"\n\n{'=' * 54}")
    print("  COMPARATIVO FINAL POR TIPO DE VEÍCULO")
    print(f"{'=' * 54}")
    print(f"  {cabecalho}")
    print(f"  {linha}")

    for tipo, dados in resultados.items():
        print(
            f"  {tipo:<12} {dados['contagem']:>8}  "
            f"{dados['iou'] * 100:>7.2f}%  "
            f"{dados['map50'] * 100:>7.2f}%  "
            f"{dados['map_geral'] * 100:>10.2f}%"
        )

    print(f"  {linha}")
    print(f"{'=' * 54}\n")


# =============================================================================
# FUNÇÃO PRINCIPAL: main
# =============================================================================

def main():
    """Executa a avaliação completa do modelo, separando os resultados por tipo de veículo."""
    print("\nCarregando o modelo YOLO...")
    modelo = YOLO(CAMINHO_MODELO)

    # Cria métricas independentes para cada grupo — elas acumulam resultados separadamente
    iou_carro,  map_carro  = criar_metricas()
    iou_moto,   map_moto   = criar_metricas()
    iou_geral,  map_geral  = criar_metricas()

    # Contadores por tipo de veículo
    contagem = {"Carros": 0, "Motos": 0, "Geral": 0}

    # Lista todas as subpastas (tracks) em ordem alfabética
    subpastas = sorted([p for p in PASTA_TESTE.iterdir() if p.is_dir()])
    print(f"Processando {len(subpastas)} tracks de teste...\n")

    # ==========================================================================
    # LOOP PRINCIPAL — percorre todos os tracks e todas as imagens
    # ==========================================================================

    for pasta in subpastas:
        for arquivo_txt in sorted(pasta.glob("*.txt")):

            # Descobre o tipo do veículo lendo o campo "type:" do arquivo UFPR
            tipo = obter_tipo_veiculo(arquivo_txt)

            # Lê as coordenadas do gabarito humano em pixels
            x_min, y_min, x_max, y_max = parse_ufpr_to_xyxy(arquivo_txt)

            # Monta o dicionário de gabarito no formato TorchMetrics
            target_dict = {
                "boxes":  torch.tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float),
                "labels": torch.zeros(1, dtype=torch.int),
            }

            # Localiza a imagem e roda a predição do YOLO
            caminho_imagem = pasta / f"{arquivo_txt.stem}.png"
            resultados = modelo.predict(source=str(caminho_imagem), conf=0.001, verbose=False)

            # Monta o dicionário de predição no formato TorchMetrics
            pred_dict = montar_pred_dict(resultados[0].boxes)

            # ── Atualiza as métricas GERAIS (todos os veículos) ────────────────
            iou_geral.update([pred_dict], [target_dict])
            map_geral.update([pred_dict], [target_dict])
            contagem["Geral"] += 1

            # ── Atualiza as métricas do grupo correspondente ao tipo ────────────
            if tipo == "car":
                iou_carro.update([pred_dict], [target_dict])
                map_carro.update([pred_dict], [target_dict])
                contagem["Carros"] += 1

            elif tipo == "motorcycle":
                iou_moto.update([pred_dict], [target_dict])
                map_moto.update([pred_dict], [target_dict])
                contagem["Motos"] += 1

    # ==========================================================================
    # RESULTADOS DETALHADOS POR GRUPO
    # ==========================================================================

    print("\n\nProcessamento concluído. Calculando métricas...\n")

    res_carros = imprimir_resultado("CARROS",   iou_carro, map_carro, contagem["Carros"])
    res_motos  = imprimir_resultado("MOTOS",    iou_moto,  map_moto,  contagem["Motos"])
    res_geral  = imprimir_resultado("GERAL",    iou_geral, map_geral, contagem["Geral"])

    # ==========================================================================
    # TABELA COMPARATIVA FINAL
    # ==========================================================================

    imprimir_tabela_comparativa({
        "Carros": {**res_carros, "contagem": contagem["Carros"]},
        "Motos":  {**res_motos,  "contagem": contagem["Motos"]},
        "Geral":  {**res_geral,  "contagem": contagem["Geral"]},
    })


# Ponto de entrada: só executa se rodar o script diretamente
if __name__ == "__main__":
    main()
