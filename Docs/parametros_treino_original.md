# Parâmetros do Treino Original — number-plate-yolo26s.pt

Comando completo usado por Muhammad Rizwan Munawar para treinar o modelo base:

```
yolo train device=5 model=ul://ultralytics/yolo26/yolo26s
data=ul://muhammadrizwanmunawar/datasets/number-plate
project=muhammadrizwanmunawar/number-plate-detection
name=number-plate-yolo26s lr0=0.01 lrf=0.01 momentum=0.937
weight_decay=0.0005 warmup_epochs=3 warmup_momentum=0.8
warmup_bias_lr=0.1 optimizer=auto box=7.5 cls=0.5 dfl=1.5
pose=12 kobj=1 label_smoothing=0 hsv_h=0.015 hsv_s=0.7
hsv_v=0.4 degrees=0 translate=0.1 scale=0.5 shear=0
perspective=0 flipud=0 fliplr=0.5 mosaic=1 mixup=0
copy_paste=0 epochs=100 batch=512 imgsz=640 patience=100
seed=0 deterministic=True amp=True cos_lr=False
close_mosaic=10 save_period=-1 fraction=1 single_cls=False
rect=False multi_scale=0 val=True resume=False cache=false
workers=8 dropout=0 iou=0.7 max_det=300
```

---

## Infraestrutura

| Parâmetro | Valor | O que significa |
|---|---|---|
| `device` | 5 | ID da GPU usada (GPU número 5 no servidor) |
| `workers` | 8 | Threads paralelas para carregar imagens durante o treino |
| `cache` | false | Não armazena as imagens na RAM — lê do disco a cada epoch |
| `amp` | True | Automatic Mixed Precision — usa float16 onde possível, economizando memória GPU sem perder precisão |
| `seed` | 0 | Semente aleatória — garante que o treino seja reproduzível |
| `deterministic` | True | Força operações determinísticas na GPU — mesma semente = mesmo resultado sempre |

---

## Modelo e Dataset

| Parâmetro | Valor | O que significa |
|---|---|---|
| `model` | ul://ultralytics/yolo26/yolo26s | Arquitetura base (YOLO26 small) carregada do Ultralytics HUB |
| `data` | ul://muhammadrizwanmunawar/datasets/number-plate | Dataset de placas genéricas internacionais — **não são placas brasileiras** |
| `project` | muhammadrizwanmunawar/number-plate-detection | Pasta onde os resultados foram salvos |
| `name` | number-plate-yolo26s | Nome desta execução de treino |
| `fraction` | 1 | Usa 100% do dataset (valor 0.5 usaria metade) |
| `single_cls` | False | Mantém múltiplas classes se houver — no nosso caso há só uma |

---

## Configuração do Treino

| Parâmetro | Valor | O que significa |
|---|---|---|
| `epochs` | 100 | Voltas completas pelo dataset de treino |
| `batch` | 512 | Imagens processadas por iteração — exige GPU de datacenter |
| `imgsz` | 640 | Resolução de entrada: todas as imagens são redimensionadas para 640×640 |
| `patience` | 100 | Early stopping: para se não melhorar em 100 epochs — na prática desligado |
| `resume` | False | Não retoma um treino anterior interrompido |
| `val` | True | Avalia no dataset de validação ao final de cada epoch |
| `save_period` | -1 | Não salva checkpoints intermediários — só salva o melhor e o último |
| `rect` | False | Não usa batches retangulares (mantém imagens quadradas) |
| `multi_scale` | 0 | Não varia a resolução durante o treino |
| `close_mosaic` | 10 | Desativa o augmentation mosaic nas últimas 10 epochs para estabilizar o treino |

---

## Learning Rate e Otimizador

| Parâmetro | Valor | O que significa |
|---|---|---|
| `optimizer` | auto | Ultralytics escolhe automaticamente (SGD ou Adam conforme o caso) |
| `lr0` | 0.01 | Learning rate inicial — tamanho do passo no começo do treino |
| `lrf` | 0.01 | Multiplicador do LR final. LR final real = lr0 × lrf = 0.01 × 0.01 = 0.0001 |
| `cos_lr` | False | Usa decaimento linear de LR (não cosseno) |
| `momentum` | 0.937 | Inércia do otimizador SGD — quanto da direção anterior é mantida na atualização dos pesos |
| `weight_decay` | 0.0005 | Regularização L2 — penaliza pesos muito grandes, reduz overfitting |
| `dropout` | 0 | Sem dropout — nenhum neurônio é desativado aleatoriamente durante o treino |

---

## Warmup (aquecimento)

O warmup é um período inicial onde o LR começa muito baixo e sobe gradualmente
até o lr0. Evita instabilidade nos primeiros passos do treino.

| Parâmetro | Valor | O que significa |
|---|---|---|
| `warmup_epochs` | 3 | Nas primeiras 3 epochs o LR sobe progressivamente até lr0 |
| `warmup_momentum` | 0.8 | Momentum usado durante o warmup (menor que o valor definitivo 0.937) |
| `warmup_bias_lr` | 0.1 | LR específico para os parâmetros de bias durante o warmup |

---

## Loss (função de erro)

O erro total = (box × perda_bbox) + (cls × perda_classe) + (dfl × perda_distribuição)

| Parâmetro | Valor | O que significa |
|---|---|---|
| `box` | 7.5 | Peso da perda de bounding box — o quanto errar a caixa penaliza o modelo |
| `cls` | 0.5 | Peso da perda de classificação — pouco relevante com uma única classe |
| `dfl` | 1.5 | Peso da Distribution Focal Loss — precisão fina dos cantos da caixa |
| `pose` | 12 | Peso da perda de pose (keypoints) — **não usado neste projeto**, pois não detectamos poses |
| `kobj` | 1 | Peso da perda de objetividade de keypoints — **não usado neste projeto** |
| `label_smoothing` | 0 | Suavização de rótulos desligada — os labels são tratados como 0 ou 1 exatos |
| `iou` | 0.7 | Limiar de IoU para o NMS (Non-Maximum Suppression) — descarta detecções sobrepostas com IoU > 70% |
| `max_det` | 300 | Número máximo de detecções por imagem |

---

## Augmentations (modificações aleatórias nas imagens de treino)

Augmentations aumentam artificialmente a diversidade do dataset sem precisar
coletar mais imagens.

### Cor e brilho

| Parâmetro | Valor | O que significa |
|---|---|---|
| `hsv_h` | 0.015 | Varia levemente o matiz (cor) — simula diferentes iluminações coloridas |
| `hsv_s` | 0.7 | Varia bastante a saturação — simula desde imagens apagadas até vibrantes |
| `hsv_v` | 0.4 | Varia o brilho — simula dia, noite, sombra, sol forte |

### Geometria

| Parâmetro | Valor | O que significa |
|---|---|---|
| `degrees` | 0 | **Sem rotação** — o modelo não aprendeu placas inclinadas |
| `translate` | 0.1 | Desloca a imagem até 10% — simula enquadramentos levemente diferentes |
| `scale` | 0.5 | Varia o zoom até 50% — simula veículos em diferentes distâncias |
| `shear` | 0 | **Sem cisalhamento** — não aplica distorção de tesoura |
| `perspective` | 0 | **Sem perspectiva** — não simula ângulos de câmera diferentes |
| `flipud` | 0 | **Sem flip vertical** — correto, placas não aparecem de cabeça para baixo |
| `fliplr` | 0.5 | Espelha 50% das imagens horizontalmente — simula placas dos dois lados |

### Composição

| Parâmetro | Valor | O que significa |
|---|---|---|
| `mosaic` | 1 | Combina 4 imagens em uma — aumenta diversidade de contexto e escala |
| `mixup` | 0 | **Desligado** — não mistura pares de imagens sobrepostas |
| `copy_paste` | 0 | **Desligado** — não recorta e cola objetos de uma imagem em outra |

---

## O que este treino **não fez** — lacunas para o nosso fine-tuning

| Lacuna | Consequência | Nossa correção |
|---|---|---|
| Dataset de placas internacionais | Nunca viu placa brasileira | Usar UFPR-ALPR |
| `imgsz=640` | Placa de moto ocupa ~1 célula da grade | `imgsz=1280` |
| `degrees=0` e `perspective=0` | Não aprendeu placas em ângulo | `degrees=10`, `perspective=0.001` |
| `cos_lr=False` | Decaimento de LR menos suave | `cos_lr=True` |
| `patience=100` | Sem early stopping efetivo | `patience=10` |
| `lr0=0.01` (alto para fine-tuning) | Risco de destruir conhecimento existente | `lr0=0.001` |
| Backbone não congelado | Pode "esquecer" features gerais | `freeze=10` |
