# Guia: Rodando o Tune no Kaggle Notebooks

**Quando usar este guia:** alternativa ao Google Colab quando a cota de GPU
estiver esgotada. O Kaggle oferece GPU T4 gratuita com **30 horas/semana**,
com sessões mais estáveis que o Colab.

**Script:** `src/tune_kaggle.py`

---

## Diferenças entre Kaggle e Colab

| Aspecto | Google Colab | Kaggle |
|---|---|---|
| Cota de GPU | ~12h/dia (variável) | 30h/semana (fixo) |
| Dados | Google Drive | Datasets próprios do Kaggle |
| Caminho de entrada | `/content/drive/MyDrive/` | `/kaggle/input/nome-dataset/` |
| Caminho de saída | `/content/` | `/kaggle/working/` |
| Montar storage | `drive.mount()` | Automático — sem código extra |
| Baixar resultados | Copiar para Drive | Download direto pelo painel |

---

## PRÉ-REQUISITO: Criar os datasets no Kaggle

Faça isso **uma única vez** — ficam salvos na sua conta para sempre.

Você precisa criar **dois datasets** no Kaggle:

### Dataset 1 — O zip do UFPR-ALPR

1. Acesse kaggle.com → clique em seu avatar → **"Your Datasets"**
2. Clique em **"New Dataset"**
3. Nome: `ufpr-alpr`
4. Faça upload do arquivo `yj4Iu2-UFPR-ALPR.zip` (~9.5 GB)
5. Clique em **"Create"** e aguarde o upload

### Dataset 2 — O modelo pré-treinado

1. Clique em **"New Dataset"** novamente
2. Nome: `yolo26s-model`
3. Faça upload do arquivo `number-plate-yolo26s.pt` (~20 MB)
4. Clique em **"Create"**

---

## PASSO 1: Criar um novo Notebook no Kaggle

1. Acesse kaggle.com → **"Create"** → **"New Notebook"**
2. Um notebook vazio será aberto

---

## PASSO 2: Adicionar os datasets ao notebook

No painel direito do notebook:

1. Clique em **"Add data"**
2. Pesquise por `ufpr-alpr` (seu dataset) → clique em **"Add"**
3. Clique em **"Add data"** novamente
4. Pesquise por `yolo26s-model` → clique em **"Add"**

Os arquivos agora estarão disponíveis automaticamente em:
```
/kaggle/input/ufpr-alpr/yj4Iu2-UFPR-ALPR.zip
/kaggle/input/yolo26s-model/number-plate-yolo26s.pt
```

---

## PASSO 3: Configurar GPU e Internet

No painel direito do notebook:

1. **Accelerator** → selecione **"GPU T4 x2"**
2. **Internet** → ative **(necessário para instalar o ultralytics)**

---

## PASSO 4: Instalar dependências

Na primeira célula do notebook, rode:

```python
!pip install ultralytics -q
```

---

## PASSO 5: Fazer upload do script

No painel esquerdo do notebook, clique no ícone de pasta → botão de upload → selecione:
```
src/tune_kaggle.py
```

---

## PASSO 6: Rodar o script

```python
!python /kaggle/working/tune_kaggle.py
```

O script vai automaticamente:
1. Descompactar o zip para `/kaggle/working/`
2. Converter anotações UFPR → formato YOLO
3. Gerar o `dataset.yaml`
4. Iniciar a busca genética (20 iterações × 3 epochs ≈ 4 horas)

---

## PASSO 7: Baixar o resultado

Ao terminar, os resultados ficam em `/kaggle/working/runs/tune/busca_hiperparametros_v1/`.

Para baixar o `best_hyperparameters.yaml`:

1. No painel esquerdo → aba **"Output"**
2. Navegue até `runs/tune/busca_hiperparametros_v1/`
3. Clique em `best_hyperparameters.yaml` → **"Download"**

---

## PASSO 8: Usar os hiperparâmetros no fine-tuning

Com o `best_hyperparameters.yaml` em mãos, abra o [fine_tuning_colab.py](../src/fine_tuning_colab.py)
e substitua os valores das constantes pelos encontrados pelo tune:

```python
# Exemplo — substitua pelos valores do seu yaml
LEARNING_RATE = 0.00218   # ← lr0 do yaml
FREEZE        = 10
# adicione os demais parâmetros no modelo.train()
```

---

## Referências rápidas

| O que | Onde |
|---|---|
| Script do tune (Kaggle) | `src/tune_kaggle.py` |
| Script do fine-tuning (Colab) | `src/fine_tuning_colab.py` |
| Script do fine-tuning (OCI/local) | `src/fine_tuning.py` |
| Guia do Colab | `Docs/guia_colab.md` |
| Como funciona o algoritmo genético | `Docs/evolucao_genetica_hiperparametros.md` |
