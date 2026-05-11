# Evolução Genética de Hiperparâmetros — Como Funciona

**Contexto:** antes de treinar o modelo final, usamos o script `src/tune_colab.py`
para deixar o YOLO encontrar automaticamente os melhores hiperparâmetros usando
um algoritmo de busca evolucionária gerenciado pela classe `Tuner` do Ultralytics.

Referência oficial: https://docs.ultralytics.com/pt/guides/hyperparameter-tuning/

---

## Por que fazer isso?

No fine-tuning anterior, definimos manualmente `lr0=0.001`, `degrees=10`, etc.
Esses valores foram escolhas razoáveis — mas são os **melhores possíveis** para
o nosso dataset? Não sabemos.

O espaço de combinações é enorme. Com 12 hiperparâmetros variando em intervalos
contínuos, testar tudo na força bruta levaria milhões de anos.

A busca evolucionária resolve isso: em vez de testar às cegas, ela **aprende**
com cada iteração onde o espaço de hiperparâmetros é promissor e foca nessas regiões.

---

## As 6 Etapas do Processo Oficial (Ultralytics Tuner)

---

### Etapa 1 — Inicializar Hiperparâmetros

O processo começa com um conjunto inicial de hiperparâmetros. Esse ponto de
partida pode ser:

- Os **valores padrão** do Ultralytics YOLO
- Valores baseados em **conhecimento de domínio** (o que fizemos no fine-tuning v1)
- Resultados de **experimentos anteriores**

No nosso caso, usamos como ponto de partida os valores do fine-tuning v1
que já produziram bons resultados (mAP@50-95 = 73,8%).

Exemplo do espaço de busca que definimos em `tune_colab.py`:

```python
ESPACO_DE_BUSCA = {
    "lr0":          (1e-4, 5e-3),   # learning rate inicial
    "degrees":      (0.0,  20.0),   # rotação
    "box":          (5.0,  15.0),   # peso da loss de bbox
    "mosaic":       (0.5,  1.0),    # augmentation mosaic
    "perspective":  (0.0,  0.001),  # distorção de perspectiva
    ...
}
```

O `Tuner` vai explorar combinações dentro desses limites ao longo das iterações.

---

### Etapa 2 — Mutar Hiperparâmetros

Esta é a etapa central do algoritmo. O `Tuner` usa o método `_mutate` para
gerar um **novo conjunto de hiperparâmetros** a partir do conjunto atual.

**Como a mutação funciona:**

O YOLO usa **mutação log-normal** combinada com **cruzamento BLX-α**:

- **Mutação log-normal:** cada valor é perturbado de forma proporcional ao
  seu tamanho atual. Isso evita mudanças drásticas — um lr0 de 0.001 não
  vai saltar para 0.5 de uma iteração para outra.

- **Cruzamento BLX-α:** os melhores resultados anteriores "trocam genes"
  entre si para gerar novos candidatos. O filho herda características dos
  pais, com uma margem de exploração além dos limites dos pais.

Exemplo visual da mutação:

```
Iteração anterior (melhor resultado):
  lr0 = 0.0024   degrees = 11.8   box = 8.3   mosaic = 0.87

Após mutação (_mutate):
  lr0 = 0.0021   degrees = 13.2   box = 7.9   mosaic = 0.91
         ↑               ↑              ↑              ↑
    variação suave  variação suave  variação suave  variação suave
```

Os valores variam suavemente, não aleatoriamente do zero —
cada iteração se baseia no conhecimento acumulado das anteriores.

---

### Etapa 3 — Treinar o Modelo

O modelo é treinado usando o conjunto mutado de hiperparâmetros gerado
na Etapa 2.

No nosso caso (`tune_colab.py`):

```python
modelo.tune(
    data=CAMINHO_YAML,
    epochs=10,          # cada candidato treina por 10 epochs
    iterations=30,      # 30 candidatos ao todo
    imgsz=1280,
    batch=8,
    optimizer="AdamW",
    freeze=10,
    space=ESPACO_DE_BUSCA,
)
```

Cada iteração treina o modelo do zero (partindo do `number-plate-yolo26s.pt`)
por `epochs=10` com os hiperparâmetros mutados. Ao final das 10 epochs,
o modelo é avaliado.

---

### Etapa 4 — Avaliar o Modelo

Após o treino de cada candidato, o `Tuner` avalia o desempenho usando
as métricas do YOLO — principalmente o **mAP@50-95** no dataset de validação.

Esse valor é o **fitness** do candidato — quanto maior, melhor a combinação
de hiperparâmetros.

```
Iteração 1:  lr0=0.0031  degrees=14.2  →  mAP@50-95 = 0.741  ← melhor até aqui
Iteração 2:  lr0=0.0008  degrees=2.7   →  mAP@50-95 = 0.698
Iteração 3:  lr0=0.0019  degrees=8.5   →  mAP@50-95 = 0.728
Iteração 4:  lr0=0.0024  degrees=11.8  →  mAP@50-95 = 0.753  ← novo melhor
...
```

O `Tuner` sempre mantém registro de qual combinação produziu o maior fitness
até o momento.

---

### Etapa 5 — Registrar os Resultados

Após cada iteração, o `Tuner` salva automaticamente os resultados em formato
**NDJSON** (Newline Delimited JSON) — um registro por linha, permitindo
acompanhar a evolução completa da busca.

Exemplo do arquivo gerado (`tune_results.csv`):

```
fitness, lr0,    lrf,   momentum, degrees, box,  mosaic
0.741,  0.0031, 0.042,  0.934,   14.2,   8.3,  0.91
0.698,  0.0008, 0.031,  0.921,   2.7,   12.1,  0.67
0.728,  0.0019, 0.038,  0.928,   8.5,    6.7,  0.82
0.753,  0.0024, 0.041,  0.931,   11.8,   7.9,  0.88
```

Esses registros servem para:
- Acompanhar a evolução do fitness ao longo das iterações
- Analisar quais hiperparâmetros mais impactam o resultado

Ao final, o arquivo `best_hyperparameters.yaml` contém os valores
do melhor candidato encontrado em todas as iterações.

---

### Etapa 6 — Repetir

O ciclo das Etapas 2 a 5 se repete até atingir o número de iterações
definido (`iterations=20`).

A diferença fundamental entre a primeira e as últimas iterações:

```
Iteração 1–5:   exploração ampla — candidatos bem diferentes entre si
Iteração 10–15: convergência — candidatos cada vez mais parecidos com os melhores
Iteração 16–20: refinamento fino — pequenas variações em torno do ótimo encontrado
```

Cada iteração **se baseia no conhecimento acumulado** de todas as anteriores —
por isso é muito mais eficiente do que busca aleatória.

---

## Resultado Final

Ao terminar todas as iterações, o `Tuner` salva:

```
runs/tune/busca_hiperparametros_v1/
├── best_hyperparameters.yaml   ← usar no fine_tuning_colab.py
├── tune_results.csv            ← histórico completo da busca
└── tune_scatter_plots.png      ← gráficos de correlação entre parâmetros e fitness
```

Exemplo do `best_hyperparameters.yaml`:

```yaml
lr0: 0.00218
lrf: 0.042
momentum: 0.934
weight_decay: 0.00038
box: 9.3
dfl: 1.8
degrees: 12.4
perspective: 0.00067
scale: 0.51
mosaic: 0.88
fliplr: 0.42
hsv_v: 0.41
```

---

## Fluxo Completo do Projeto com Tune

```
ETAPA A — tune_colab.py
  20 iterações × 3 epochs = busca genética
  Resultado: best_hyperparameters.yaml

ETAPA B — fine_tuning_colab.py (com os valores do yaml)
  50 epochs com hiperparâmetros otimizados
  Resultado: best_placas_v2.pt

ETAPA C — eval_por_tipo_best_placas.py
  Comparativo: v1 (manual) vs v2 (otimizado geneticamente)
```

---

## Estimativa de Tempo no Colab T4

| Configuração | Tempo estimado |
|---|---|
| `epochs=3`,  `iterations=20` | ~4 horas (recomendado para Colab free) |
| `epochs=5`,  `iterations=20` | ~7 horas |
| `epochs=10`, `iterations=30` | ~22 horas (excede o limite do Colab free) |

Cálculo: cada epoch com 1.800 imagens, batch=8, imgsz=1280 no T4 leva ~4 minutos.
`epochs × iterations × 4min = tempo total`.

Para o Colab free (limite ~12h por sessão), use `epochs=3, iterations=20`.
