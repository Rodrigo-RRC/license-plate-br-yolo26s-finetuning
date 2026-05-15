# Guia: Usando o Modelo para Inferência

Instruções para carregar e rodar o modelo publicado no Hugging Face.

---

## Instalação

```bash
pip install ultralytics huggingface_hub
```

---

## Carregando o modelo

O modelo está publicado em `RodrigoRRC/license-plate-br-yolo26s` no Hugging Face.

Use `hf_hub_download` para baixar — o prefixo `hf://` requer ultralytics recente e pode não funcionar em todos os ambientes:

```python
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

path = hf_hub_download(
    repo_id="RodrigoRRC/license-plate-br-yolo26s",
    filename="best_placas_v2.pt"
)

modelo = YOLO(path)
```

---

## Rodando uma predição

```python
resultados = modelo.predict(source="caminho/ou/url/da/imagem.jpg", conf=0.25)
resultados[0].show()

print(f"Detecções: {len(resultados[0].boxes)}")
for box in resultados[0].boxes:
    print(f"  Confiança: {box.conf[0]:.2%}")
```

O parâmetro `conf` define o limiar de confiança (0.25 = 25%). Ajuste conforme necessário.

---

## Modelos disponíveis

| Arquivo | Descrição |
|---|---|
| `best_placas_v1.pt` | Primeira versão — fine-tuning com hiperparâmetros padrão |
| `best_placas_v2.pt` | Segunda versão — fine-tuning com hiperparâmetros otimizados pela busca genética |

Use sempre o `best_placas_v2.pt` para melhores resultados.

---

## Notebook de treinamento

O notebook completo usado para treinar o modelo está em:
```
fine-tuning-no-modelo-number-plate-yolo26s-pt (1).ipynb
```

Publicado também em: https://www.kaggle.com/code/rodrigorrc/fine-tuning-no-modelo-number-plate-yolo26s-pt
