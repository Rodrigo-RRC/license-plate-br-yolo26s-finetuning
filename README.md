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
