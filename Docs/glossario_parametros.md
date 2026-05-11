# Glossário de Parâmetros e Métricas — Fine-Tuning YOLO

**Projeto:** Detecção de placas veiculares brasileiras — UFPR-ALPR  
**Modelo base:** `number-plate-yolo26s.pt`  
**Última atualização:** 2026-05-05

---

## 1. Painel do Treino em Tempo Real

Estas colunas aparecem a cada epoch durante o fine-tuning:

```
Epoch   GPU_mem   box_loss   cls_loss   dfl_loss   Instances   Size
 1/50    6.42G      1.622      1.321     0.002482          16   1280: 73%
```

| Coluna | Exemplo | O que significa |
|---|---|---|
| `Epoch` | 1/50 | Epoch atual / total de epochs |
| `GPU_mem` | 6.42G | Memória da GPU em uso no momento — no T4 o limite é 16GB |
| `box_loss` | 1.622 | Erro médio na posição e tamanho das bounding boxes — **deve cair ao longo das epochs** |
| `cls_loss` | 1.321 | Erro médio na classificação (acertar que é uma placa) — tende a cair rápido pois há só uma classe |
| `dfl_loss` | 0.002482 | Erro médio na distribuição dos cantos da caixa (precisão fina) — também deve cair |
| `Instances` | 16 | Número de placas presentes nas imagens do batch atual |
| `Size` | 1280: 73% | Resolução de entrada e progresso do batch atual |

**Como interpretar a evolução:**
- As três losses (`box_loss`, `cls_loss`, `dfl_loss`) devem diminuir ao longo das epochs
- Se estabilizarem por 10 epochs consecutivas, o early stopping vai parar o treino
- `GPU_mem` estável indica que não há risco de estouro de memória

---

## 2. Métricas de Avaliação

Estas métricas são usadas para medir a qualidade do modelo **depois** do treino,
rodando o script `eval_por_tipo.py`.

---

### IoU — Intersection over Union

**O que é:** mede o quanto a caixa prevista pelo modelo se sobrepõe à caixa correta (gabarito humano).

```
        Área de sobreposição
IoU = ─────────────────────────────────
       Área da união das duas caixas
```

**Escala:** 0% (sem sobreposição) a 100% (sobreposição perfeita).

**Nossos resultados atuais:**

| Tipo | IoU atual | Meta pós fine-tuning |
|---|---|---|
| Carros | 70,30% | > 78% |
| Motos | 61,78% | > 70% |
| Geral | 68,60% | > 75% |

---

### mAP@50 — Mean Average Precision com limiar 50%

**O que é:** o modelo considera uma detecção correta se o IoU for **≥ 50%**.
Depois calcula a média da precisão em todos os casos.

**Interpretação:** critério tolerante — mesmo caixas razoavelmente imprecisas
são contadas como acerto. Útil para saber se o modelo **localiza** o objeto.

**Resultado atual:** 83,11%

---

### mAP@75 — Mean Average Precision com limiar 75%

**O que é:** igual ao mAP@50, mas exige IoU **≥ 75%** para contar como acerto.

**Interpretação:** critério rigoroso — revela se as caixas são geometricamente
precisas. Uma queda grande entre mAP@50 e mAP@75 indica caixas mal ajustadas.

**Resultado atual:** 43,02% — queda de 40 pontos em relação ao mAP@50,
confirmando o problema geométrico (especialmente em motos).

---

### mAP@50-95 — Mean Average Precision médio

**O que é:** média do mAP calculado em 10 limiares diferentes:
50%, 55%, 60%, 65%, 70%, 75%, 80%, 85%, 90%, 95%.

**Interpretação:** métrica oficial do COCO — a mais completa. Penaliza tanto
detecções ausentes quanto caixas imprecisas.

**Resultado atual:** 45,00%

---

## 2. Parâmetros de Treino

Valores configurados em `src/fine_tuning.py` e `src/fine_tuning_colab.py`.

---

### data
**O que é:** caminho para o arquivo `dataset.yaml`, que aponta para as imagens
de treino, validação e teste, e define as classes.

**Valor:** `/workspaces/car_license_plate_detector/dataset.yaml`

---

### epochs
**O que é:** número de voltas completas pelo dataset de treino.
A cada epoch, o modelo vê todas as 1.800 imagens de treino uma vez.

**Valor original:** 100 | **Nosso valor:** 50

**Por quê 50:** fine-tuning converge mais rápido que treino do zero — o modelo
já sabe detectar objetos, só precisa se adaptar ao novo domínio.

---

### batch
**O que é:** quantas imagens o modelo processa antes de atualizar os pesos.
Lotes maiores = treino mais estável, mas exigem mais memória GPU.

**Valor original:** 512 (datacenter) | **OCI:** 16 | **Colab T4:** 8

---

### imgsz
**O que é:** resolução para a qual todas as imagens são redimensionadas antes
de entrar no modelo. O YOLO divide a imagem em uma grade de células — quanto
maior o imgsz, mais células, mais detalhe para objetos pequenos.

**Valor original:** 640 | **Nosso valor:** 1280

**Por quê 1280:** as imagens do UFPR são 1920×1080. Com imgsz=640, a placa de
moto ocupa ~1 célula. Com imgsz=1280, ocupa ~4 células — muito mais detalhe.

---

### lr0 — Learning Rate inicial
**O que é:** o tamanho do "passo" que o modelo dá ao corrigir um erro no início
do treino. Passos grandes = aprendizado rápido mas impreciso. Passos pequenos
= aprendizado lento mas preciso.

**Valor original:** 0,01 | **Nosso valor:** 0,001

**Por quê menor:** no fine-tuning, o modelo já tem conhecimento útil. Um lr0
alto destruiria esse conhecimento. Usamos 10× menor para ajustes suaves.

---

### lrf — Learning Rate final (multiplicador)
**O que é:** define o LR no final do treino como uma fração do lr0.
LR final real = lr0 × lrf.

**Valor:** 0,01 (padrão Ultralytics)

**Exemplo com nossos valores:** lr0=0,001 × lrf=0,01 → LR final = 0,00001.
O modelo começa dando passos de 0,001 e termina dando passos de 0,00001.

---

### cos_lr — Cosine Learning Rate
**O que é:** define como o LR diminui ao longo do treino.
- `False` (original): decaimento linear
- `True` (nosso): decaimento em curva de cosseno — começa rápido, desacelera
  suavemente, termina com passos muito pequenos para afinar os pesos.

**Nosso valor:** True

---

### patience — Early Stopping
**O que é:** se o modelo passar N epochs seguidas sem melhorar o desempenho
na validação, o treino para automaticamente.

**Valor original:** 100 (desligado na prática) | **Nosso valor:** 10

**Por quê 10:** evita desperdício de tempo e overfitting (memorização do treino).
Se não melhorar em 10 epochs seguidas, não vai melhorar mais.

---

### freeze
**O que é:** número de camadas iniciais do modelo que ficam congeladas durante
o treino — seus pesos não são atualizados.

**Nosso valor:** 10

**Por quê congelar:** as primeiras camadas (backbone) já aprenderam a detectar
bordas, formas e texturas gerais — conhecimento que serve para qualquer imagem.
Congelando-as, treinamos apenas a cabeça de detecção, que precisa aprender
o que é especificamente uma placa brasileira.

---

### degrees — Rotação
**O que é:** augmentation que rotaciona aleatoriamente as imagens de treino
em até ±N graus.

**Valor original:** 0 | **Nosso valor:** 10

**Por quê ativar:** placas aparecem em ângulos variados na vida real,
especialmente em motos fotografadas de diferentes posições.

---

### perspective — Perspectiva
**O que é:** augmentation que aplica distorção de perspectiva às imagens,
simulando ângulos de câmera diferentes.

**Valor original:** 0 | **Nosso valor:** 0,001

**Por quê ativar:** câmeras registram veículos de diferentes alturas e ângulos.
O modelo precisa aprender a detectar placas com leve distorção perspectiva.

---

## 3. Parâmetros de Loss (função de erro)

O YOLO calcula três erros simultaneamente durante o treino e os soma com pesos:

### box
**O que é:** peso dado ao erro de posição e tamanho da bounding box.
Quanto maior, mais o modelo prioriza acertar a caixa.

**Valor:** 7,5 (mantido do original — padrão Ultralytics)

---

### cls — Classification Loss
**O que é:** peso dado ao erro de classificação (acertar a classe do objeto).

**Valor:** 0,5 — pouco importante no nosso caso, pois temos **uma única classe**
(`license_plate`). O modelo quase nunca erra a classe.

---

### dfl — Distribution Focal Loss
**O que é:** peso dado à perda de distribuição dos cantos da bbox.
O YOLO moderno não prevê coordenadas exatas — prevê uma distribuição
de probabilidade sobre possíveis valores para cada canto. O DFL mede
o erro nessa distribuição.

**Valor:** 1,5 (mantido do original — padrão Ultralytics)

---

## 4. Augmentations (mantidos do original)

| Parâmetro | Valor | O que faz |
|---|---|---|
| `mosaic` | 1,0 | Combina 4 imagens em 1 — aumenta diversidade de contexto |
| `fliplr` | 0,5 | Espelha 50% das imagens horizontalmente |
| `scale` | 0,5 | Varia o zoom aleatoriamente — simula diferentes distâncias |
| `hsv_h` | 0,015 | Varia levemente o matiz das cores |
| `hsv_s` | 0,7 | Varia a saturação — simula diferentes condições de luz |
| `hsv_v` | 0,4 | Varia o brilho — simula dia/noite, sombras |
| `translate` | 0,1 | Desloca a imagem até 10% — simula enquadramentos diferentes |
| `flipud` | 0 | Flip vertical desligado — placas não aparecem de cabeça para baixo |
| `mixup` | 0 | Desligado |

---

## 5. Resumo: Original vs. Nosso Fine-Tuning

| Parâmetro | Original | Fine-Tuning |
|---|---|---|
| Dataset | Placas internacionais genéricas | UFPR-ALPR (placas brasileiras) |
| imgsz | 640 | **1280** |
| epochs | 100 | 50 |
| patience | 100 (desligado) | **10** |
| batch | 512 | 16 (OCI) / 8 (Colab) |
| lr0 | 0,01 | **0,001** |
| cos_lr | False | **True** |
| degrees | 0 | **10** |
| perspective | 0 | **0,001** |
| freeze | não usado | **10** |
| box / cls / dfl | 7,5 / 0,5 / 1,5 | 7,5 / 0,5 / 1,5 (mantidos) |
