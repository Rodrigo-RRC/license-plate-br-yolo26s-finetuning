# Guia: Rodando o Fine-Tuning no Google Colab

**Quando usar este guia:** quando não tiver acesso à OCI (Alex) e quiser rodar
o fine-tuning na GPU gratuita do Google Colab (T4).

---

## PRÉ-REQUISITO: Subir os arquivos para o Google Drive

Faça isso **uma única vez** — depois ficam salvos no Drive para sempre.

Você precisa subir dois arquivos para o seu Google Drive:

| Arquivo | Onde está localmente | Tamanho |
|---|---|---|
| `yj4Iu2-UFPR-ALPR.zip` | `/workspaces/car_license_plate_detector/` | ~9.5 GB |
| `number-plate-yolo26s.pt` | `/workspaces/car_license_plate_detector/` | ~20 MB |

### Opção A — Upload via rclone (recomendado, mais rápido)

O rclone já está instalado e configurado neste devcontainer. Basta rodar:

```bash
# Sobe o zip (demora — não desligue o computador enquanto roda)
rclone copy /workspaces/car_license_plate_detector/yj4Iu2-UFPR-ALPR.zip gdrive: --progress

# Sobe o modelo (rápido)
rclone copy /workspaces/car_license_plate_detector/number-plate-yolo26s.pt gdrive: --progress
```

Se a conexão cair no meio, rode o mesmo comando de novo — o rclone retoma de onde parou.

### Opção B — Upload via navegador (alternativa)

Abra o Google Drive no navegador e arraste os arquivos diretamente do Explorer do Windows.
**Atenção:** não feche o navegador nem desligue o computador durante o upload.

---

## PASSO 1: Abrir o Google Colab

Acesse: https://colab.research.google.com

---

## PASSO 2: Configurar GPU T4

No menu do Colab:
```
Ambiente de execução → Alterar o tipo de ambiente de execução → T4 GPU → Salvar
```

---

## PASSO 3: Subir o script para o Colab

No painel esquerdo do Colab, clique no ícone de pasta → botão de upload → selecione:
```
/workspaces/car_license_plate_detector/src/fine_tuning_colab.py
```

---

## PASSO 4: Instalar dependências

Em uma célula do Colab, rode:

```python
!pip install ultralytics -q
```

---

## PASSO 5: Rodar o script

```python
!python fine_tuning_colab.py
```

O script vai automaticamente:
1. Montar o Google Drive (abre janela de autorização — clique em Permitir)
2. Descompactar o zip do Drive para `/content/` (demora alguns minutos — ~9.5GB)
3. Converter anotações UFPR → formato YOLO
4. Gerar o `dataset.yaml` com caminhos do Colab
5. Iniciar o fine-tuning (estimativa: 3–5 horas na T4)

---

## PASSO 6: Salvar o modelo treinado no Drive

**IMPORTANTE:** ao fechar a sessão do Colab, tudo em `/content/` é apagado.
Rode este comando assim que o treino terminar:

```python
import shutil
shutil.copy(
    '/content/runs/fine_tuning/placas_brasileiras_colab_v1/weights/best.pt',
    '/content/drive/MyDrive/best_placas_v1.pt'
)
print("Modelo salvo no Drive!")
```

---

## PASSO 7: Avaliar o modelo treinado

De volta ao devcontainer local, baixe o `best_placas_v1.pt` do Drive e rode:

```bash
# Baixa o modelo do Drive para o repositório local
rclone copy gdrive:best_placas_v1.pt /workspaces/car_license_plate_detector/

# Atualiza o caminho do modelo no script de avaliação e roda
python src/eval_por_tipo.py
```

Compare os resultados com o baseline:

| Métrica | Baseline (modelo original) | Meta após fine-tuning |
|---|---|---|
| IoU geral | 68,60% | > 75% |
| IoU motos | 61,78% | > 70% |
| mAP@50 geral | 83,11% | > 88% |

---

## Referências rápidas

| O que | Onde |
|---|---|
| Script de fine-tuning (Colab) | `src/fine_tuning_colab.py` |
| Script de fine-tuning (OCI/local) | `src/fine_tuning.py` |
| Script de avaliação | `src/eval_por_tipo.py` |
| Plano detalhado do fine-tuning | `Docs/plano_fine_tuning.md` |
| Modelo no Drive | `gdrive:best_placas_v1.pt` |
