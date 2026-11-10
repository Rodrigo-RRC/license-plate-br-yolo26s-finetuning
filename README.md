# Detecção de Placas Veiculares Brasileiras com YOLO26s

[![Testar online](https://img.shields.io/badge/▶%20Testar%20online-Playground-FF6F00?logo=huggingface&logoColor=white)](https://huggingface.co/spaces/RodrigoRRC/deteccao-placas-br)
[![Modelo no HuggingFace](https://img.shields.io/badge/🤗%20Modelo-HuggingFace-yellow)](https://huggingface.co/RodrigoRRC/license-plate-br-yolo26s)
[![Notebook no Kaggle](https://img.shields.io/badge/Notebook-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/rodrigorrc/03-avaliacao-base-v1-v2)
[![Licença AGPL-3.0](https://img.shields.io/badge/Licença-AGPL--3.0-blue)](#licença)

Modelo de visão computacional que localiza placas de carros e motos em imagens de trânsito brasileiro.

O ponto de partida foi o `number-plate-yolo26s`, um detector treinado com placas de vários países. Ele foi ajustado ao padrão brasileiro com o dataset **UFPR-ALPR**, em duas etapas:

1. **v1**: fine-tuning com hiperparâmetros escolhidos manualmente
2. **v2**: fine-tuning com hiperparâmetros encontrados por busca genética (`model.tune()` do Ultralytics)

O modelo final (v2) atinge **mAP@50-95 de 82,03%** no conjunto de teste, contra 60,24% do modelo base, e está disponível para uso no HuggingFace.

![Fluxo do treino: modelo base, v1 e v2](Docs/fluxo_treino.svg)

---

## Resultado

Os três modelos foram avaliados no **mesmo conjunto de teste** do UFPR-ALPR (1.800 imagens que não foram usadas no treino), com as mesmas configurações.

| Métrica | Modelo base | v1 | v2 |
|---|---|---|---|
| mAP@50 | 89,58% | **99,11%** | 98,33% |
| mAP@50-95 | 60,24% | 77,09% | **82,03%** |
| Precisão | 86,32% | 96,99% | **97,69%** |
| Recall | 89,77% | 96,17% | **98,78%** |

**O que cada métrica mede:**
- **Precisão:** das placas que o modelo apontou, quantas eram placas de verdade
- **Recall:** das placas que existiam nas imagens, quantas o modelo encontrou
- **mAP@50:** taxa de acerto quando basta a caixa prevista se sobrepor pelo menos 50% à caixa real (critério tolerante)
- **mAP@50-95:** média da taxa de acerto em critérios cada vez mais exigentes, de 50% até 95% de sobreposição. Mede o quão justa a caixa fica em volta da placa

**Leitura dos resultados:** o fine-tuning melhorou todas as métricas em relação ao modelo base. O v1 e o v2 encontram praticamente todas as placas (mAP@50 acima de 98%). O ganho do v2 está no **mAP@50-95 (+4,9 pontos)**, ou seja, as caixas ficam mais bem ajustadas à placa.

📓 Avaliação completa, com as saídas gravadas: [`notebooks/03_avaliacao_base_v1_v2.ipynb`](notebooks/03_avaliacao_base_v1_v2.ipynb) · [Kaggle](https://www.kaggle.com/code/rodrigorrc/03-avaliacao-base-v1-v2)

---

## Como usar o modelo

> [!TIP]
> **Teste online, sem instalar nada:** [playground no HuggingFace](https://huggingface.co/spaces/RodrigoRRC/deteccao-placas-br). Envie uma foto, escolha o modelo (v1 ou v2) e veja as placas detectadas.

Para usar o modelo no seu próprio código:

**1. Clonar o repositório e instalar as dependências**

```bash
git clone https://github.com/Rodrigo-RRC/license-plate-br-yolo26s-finetuning.git
cd license-plate-br-yolo26s-finetuning
pip install -r requirements.txt
```

**2. Baixar o modelo do HuggingFace**

```python
from huggingface_hub import hf_hub_download

caminho_modelo = hf_hub_download(
    repo_id="RodrigoRRC/license-plate-br-yolo26s",
    filename="best_placas_v2.pt",   # ou "best_placas_v1.pt"
)
```

**3. Detectar placas numa imagem**

```python
from ultralytics import YOLO

modelo = YOLO(caminho_modelo)
resultado = modelo.predict(source="sua_imagem.jpg", conf=0.5)[0]

resultado.save("resultado.jpg")   # salva a imagem com as placas marcadas

for caixa in resultado.boxes:
    x1, y1, x2, y2 = caixa.xyxy[0].tolist()
    print(f"Placa em ({x1:.0f}, {y1:.0f}) até ({x2:.0f}, {y2:.0f}) · confiança {caixa.conf[0]:.0%}")
```

- `conf=0.5` → só mostra detecções com pelo menos 50% de confiança. Diminua para achar mais placas, aumente para ter menos alarmes falsos
- O modelo **localiza** a placa (desenha a caixa). Ele **não lê** os caracteres

---

## Desenvolvimento do projeto

O desenvolvimento foi dividido em três etapas, cada uma documentada em um notebook com as saídas gravadas:

| Etapa | Notebook | Ambiente | Descrição |
|---|---|---|---|
| 1. Fine-tuning v1 | [`01_fine_tuning_v1_colab`](notebooks/01_fine_tuning_v1_colab.ipynb) | Google Colab · GPU T4 | Fine-tuning do modelo base com hiperparâmetros definidos manualmente |
| 2. Otimização e fine-tuning v2 | [`02_tune_genetico_e_treino_v2_kaggle`](notebooks/02_tune_genetico_e_treino_v2_kaggle.ipynb) | Kaggle · GPU T4 | Busca evolutiva de hiperparâmetros (20 iterações × 3 épocas) seguida do fine-tuning com a melhor configuração (50 épocas) |
| 3. Avaliação | [`03_avaliacao_base_v1_v2`](notebooks/03_avaliacao_base_v1_v2.ipynb) | Kaggle | Avaliação dos três modelos no conjunto de teste, sob as mesmas condições |

### Configuração de treino

Parâmetros comuns aos dois fine-tunings: 50 épocas, `imgsz=1280`, `batch=8`, `freeze=10` (backbone congelado) e `cos_lr=True`.

**Diferenças relevantes**

| Parâmetro&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | v1 (definição manual) | v2&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |
|---|---|---|
| Otimizador | auto | AdamW (manual) |
| `lr0` | 0,001 | 0,00285 |
| `degrees` | 10 | 0,005 |
| `perspective` | 0,001 | 0 |
| `mosaic` | 1,0 | 0,61 |
| `weight_decay` | 0,0005 | 0,0001 |
| `dfl` | 1,5 | 1,23 |
| `lrf` | 0,01 | 0,0145 |
| `patience` | 10 | 15 (manual) |

**Diferenças mínimas** (a busca praticamente manteve o valor padrão)

| Parâmetro&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | v1 (definição manual) | v2&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |
|---|---|---|
| `momentum` | 0,937 | 0,942 |
| `box` | 7,5 | 7,51 |
| `scale` | 0,5 | 0,495 |
| `hsv_v` | 0,4 | 0,407 |

A busca evolutiva convergiu para uma rotação próxima de zero (`degrees`), diferente dos 10° definidos manualmente no v1, e reduziu o `mosaic` de 1,0 para 0,61. O otimizador AdamW não fez parte da busca: foi definido manualmente para o v2.

---

## Dataset

Treinamento e avaliação com o **UFPR-ALPR** (Laroca et al., 2018): 4.500 imagens em 1.920×1.080 px de 150 veículos (carros e motos), capturadas em cenários reais de trânsito.

| Divisão | Imagens | Uso |
|---|---|---|
| Treino | 1.800 | Fine-tuning |
| Validação | 900 | Seleção de hiperparâmetros e early stopping |
| Teste | 1.800 | Avaliação final |

O dataset é restrito a pesquisa acadêmica e não é redistribuído neste repositório. O acesso deve ser solicitado aos autores: [instruções](UFPR-ALPR%20dataset/README.md) · [página oficial](https://web.inf.ufpr.br/vri/databases/ufpr-alpr/)

---

## Modelo base

[`number-plate-yolo26s`](https://platform.ultralytics.com/muhammadrizwanmunawar/datasets/number-plate), de Muhammad Rizwan Munawar: YOLO26s treinado em um dataset de placas de diversos países (licença AGPL-3.0).

---

## Licença

**AGPL-3.0**, herdada do modelo base.

⚠️ **Uso não comercial.** Os modelos foram treinados no UFPR-ALPR, cujo uso é restrito a fins acadêmicos e não comerciais. A restrição se estende aos modelos derivados.

---

## Citação

Se utilizar este trabalho, cite o dataset original:

> R. Laroca, E. Severo, L. A. Zanlorensi, L. S. Oliveira, G. R. Gonçalves, W. R. Schwartz, D. Menotti. *A Robust Real-Time Automatic License Plate Recognition Based on the YOLO Detector.* International Joint Conference on Neural Networks (IJCNN), 2018, pp. 1-10.

---

## Autor

**Rodrigo Ribeiro Carvalho**
[GitHub](https://github.com/Rodrigo-RRC) · [HuggingFace](https://huggingface.co/RodrigoRRC) · [LinkedIn](https://www.linkedin.com/in/rodrigo-ribeiro-datascience) · [Portfólio](https://rodrigo-rrc.github.io)
