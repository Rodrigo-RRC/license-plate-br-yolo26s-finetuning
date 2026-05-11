# Plano de Fine-Tuning — Detecção de Placas Brasileiras

**Projeto:** Detecção de placas veiculares com YOLO no dataset UFPR-ALPR  
**Modelo base:** `number-plate-yolo26s.pt` (Muhammad Rizwan Munawar)  
**Dataset alvo:** UFPR-ALPR — 150 veículos, 4.500 imagens, 1920×1080px  
**Última atualização:** 2026-05-04

---

## Por que fazer fine-tuning?

O modelo `number-plate-yolo26s.pt` foi treinado em um dataset genérico de placas
internacionais. Ao avaliá-lo contra o dataset UFPR-ALPR (placas brasileiras reais),
os resultados mostraram desempenho insuficiente para uso em produção:

| Métrica | Resultado | Interpretação |
|---|---|---|
| IoU médio geral | 68,60% | Caixas mal ajustadas em ~1/3 dos casos |
| mAP@50 | 83,11% | Parece bom, mas o critério de 50% é tolerante |
| mAP@75 | 43,02% | Colapso ao exigir precisão real |
| mAP@50-95 | 45,00% | Inaceitável para produção |
| IoU em motos | 61,78% | Pior ainda — problema geométrico evidente |

**Diagnóstico:** O modelo nunca viu uma placa brasileira. A diferença de fonte,
proporção, fundo e contexto visual explica o baixo desempenho.

---

## Análise do treino original (o que o modelo já fez)

Este é o comando com que `number-plate-yolo26s.pt` foi treinado originalmente:

```
yolo train device=5
  model=ul://ultralytics/yolo26/yolo26s
  data=ul://muhammadrizwanmunawar/datasets/number-plate
  epochs=100  batch=512  imgsz=640  patience=100
  lr0=0.01  lrf=0.01  cos_lr=False
  optimizer=auto
  box=7.5  cls=0.5  dfl=1.5
  hsv_h=0.015  hsv_s=0.7  hsv_v=0.4
  degrees=0  translate=0.1  scale=0.5
  shear=0  perspective=0
  flipud=0  fliplr=0.5
  mosaic=1  mixup=0  copy_paste=0
  warmup_epochs=3  warmup_momentum=0.8  warmup_bias_lr=0.1
  iou=0.7  max_det=300  dropout=0
```

### Parâmetro por parâmetro — o que isso significa

| Parâmetro | Valor original | O que faz | Implicação para o nosso caso |
|---|---|---|---|
| `data` | dataset genérico internacional | Define o conjunto de treino | **Nunca viu placa brasileira** — principal causa do problema |
| `imgsz` | 640 | Resolução de entrada | Placa de moto ocupa ~1 célula da grade — pouco detalhe |
| `epochs` | 100 | Voltas no dataset | Treino completo; não parou antes |
| `patience` | 100 | Early stopping | Desligado na prática — rodou todas as 100 epochs |
| `batch` | 512 | Imagens por iteração | Exigiu GPU de datacenter; impossível reproduzir localmente |
| `lr0` | 0.01 | Learning rate inicial | Padrão razoável para treino do zero |
| `lrf` | 0.01 | Learning rate final | **Igual ao inicial — sem decaimento de LR** |
| `cos_lr` | False | Agendamento cosseno de LR | Desligado — LR constante até o fim |
| `degrees` | 0 | Rotação das imagens | **Sem rotação** — modelo não aprendeu placas inclinadas |
| `perspective` | 0 | Distorção de perspectiva | **Sem perspectiva** — modelo não aprendeu variações angulares |
| `flipud` | 0 | Flip vertical | Desligado — razoável, placas não aparecem de cabeça para baixo |
| `fliplr` | 0.5 | Flip horizontal | Ativo — ajuda com placas em ambos os lados |
| `mosaic` | 1 | Combina 4 imagens em 1 | Ativo — aumenta diversidade visual |
| `mixup` | 0 | Mistura pares de imagens | Desligado |
| `scale` | 0.5 | Variação de escala | Ativo — ajuda com objetos em diferentes distâncias |
| `box` | 7.5 | Peso da loss de bbox | Padrão Ultralytics |
| `cls` | 0.5 | Peso da loss de classe | Padrão Ultralytics |
| `dfl` | 1.5 | Peso da Distribution Focal Loss | Padrão Ultralytics |

### Principais lacunas identificadas

1. **Domínio errado** — dataset de placas genéricas vs. placas brasileiras (UFPR-ALPR)
2. **Sem decaimento de LR** — `lr0=lrf=0.01` com `cos_lr=False` significa que o modelo
   aprendeu na mesma velocidade do começo ao fim, sem afinar os pesos no final do treino
3. **Sem rotação nem perspectiva** — placas em ângulo são um caso real frequente,
   especialmente em motos fotografadas de diferentes posições
4. **Resolução insuficiente para motos** — `imgsz=640` faz a placa de moto ocupar
   apenas 1–2 células na grade do YOLO

---

## Distribuição do dataset UFPR-ALPR

Antes de definir o fine-tuning, analisamos a distribuição de tipos de veículo
(script: `src/analisar_distribuicao.py`):

| Split | Carros (tracks) | Motos (tracks) | Total |
|---|---|---|---|
| Treino | 48 (80%) | 12 (20%) | 60 |
| Validação | 24 (80%) | 6 (20%) | 30 |
| Teste | 48 (80%) | 12 (20%) | 60 |
| **Total** | **120** | **30** | **150** |

**Conclusão:** A distribuição é perfeitamente proporcional em todos os splits.
O desempenho inferior em motos **não é** causado por falta de exemplos de treino.
A causa são as lacunas 3 e 4 identificadas acima (geometria e resolução).

---

## Nossa estratégia de fine-tuning

### Comparativo: treino original vs. nosso fine-tuning

| Parâmetro | Treino original | Nosso fine-tuning | Por quê mudamos |
|---|---|---|---|
| `data` | Placas genéricas internacionais | UFPR-ALPR (placas brasileiras) | Domínio correto |
| `imgsz` | 640 | **1280** | Mais detalhe para placas de moto |
| `epochs` | 100 | 50 | Fine-tuning converge mais rápido que treino do zero |
| `patience` | 100 (desligado) | **10** | Parar quando parar de melhorar |
| `batch` | 512 | 16 | Adequado à GPU disponível (OCI) |
| `lr0` | 0.01 | **0.001** | LR inicial menor para fine-tuning — passos suaves para não destruir o conhecimento já adquirido |
| `lrf` | 0.01 | 0.01 | Multiplicador do LR final (LR final real = lr0 × lrf). Mantido; o decaimento em si é controlado pelo `cos_lr` |
| `cos_lr` | False | **True** | Liga o agendamento cosseno: LR decai suavemente de lr0 (0.001) até lr0×lrf (0.00001) ao longo do treino |
| `degrees` | 0 | **10** | Placas em ângulo — especialmente motos |
| `perspective` | 0 | **0.001** | Variações de perspectiva reais |
| `box` | 7.5 | 7.5 | Mantido — padrão razoável |
| `freeze` | não usado | **10** | Congela backbone; treina só a cabeça de detecção |

### Por que `freeze=10`?

O backbone do YOLO (primeiras camadas) já aprendeu a detectar bordas, formas e
texturas gerais. Congelar essas camadas durante o fine-tuning evita que o modelo
"esqueça" esse conhecimento geral ao adaptar-se para placas brasileiras.
Treinamos apenas a cabeça de detecção, que é a parte responsável por fazer as
previsões finais de bbox e classe.

### Por que `lr0=0.001` e não `0.01`?

No treino do zero, um LR alto (0.01) é necessário para fazer os pesos saírem do
ponto de partida aleatório rapidamente. No fine-tuning, os pesos já estão em um
bom ponto — um LR alto os perturbaria demais, destruindo o conhecimento já adquirido.
LR 10× menor (0.001) permite ajustes suaves e precisos.

---

## Scripts do projeto

| Script | O que faz | Status |
|---|---|---|
| `src/preparar_dataset.py` | Converte anotações UFPR → formato YOLO e organiza pastas | ✅ Executado |
| `src/analisar_distribuicao.py` | Analisa distribuição de tipos de veículo por split | ✅ Executado |
| `src/eval_por_tipo.py` | Avalia o modelo separando métricas por tipo de veículo | ✅ Executado (modelo base) |
| `src/fine_tuning.py` | Executa o fine-tuning do modelo com o dataset UFPR-ALPR | ⏳ Aguardando GPU (OCI) |

---

## Próximos passos

- [ ] Rodar `fine_tuning.py` na OCI com GPU (aguardando acesso via Alex)
- [ ] Rodar `eval_por_tipo.py` com o modelo fine-tuned e comparar com o baseline
- [ ] Se IoU de motos não melhorar suficientemente, investigar geometria das placas
- [ ] Avaliar se o modelo fine-tuned é adequado para produção (serviço)
