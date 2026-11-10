# Detecção de Placas Veiculares Brasileiras com YOLO26s

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
