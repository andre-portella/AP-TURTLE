# Projeto da Disciplina de Aprendizado Profundo para Reconhecimento Visual

Este repositório reúne o código, experimentos e análises desenvolvidos para a disciplina de **Aprendizado Profundo**, com foco na aplicação de transferência não supervisionada em associação à consulta por rótulo de baixo custo.

Referências desse projeto:

- **Low-budget Label Query through Domain Alignment Enforcement** - (https://doi.org/10.1016/j.cviu.2022.103485) publicado no Computer Vision and Image Understanding (**CVIU**) journal, 2022.
- **Let Go of Your Labels with Unsupervised Transfer** - (https://arxiv.org/abs/2406.07236) publicada na ICML, 2024.

---
# Requisitos

- tqdm
- numpy
- torch
- clip
- torchvision
- scipy
- scikit-learn
- pandas



# 1. Visão Geral do Pipeline

O pipeline completo envolve as seguintes etapas:

1. Extração das representações (ResNet-18, VGG-11 e DIALNet)  
2. Extração dos rótulos  
3. Execução do TURTLE  
4. Seleção de amostras incertas  
5. Ajuste fino (Fine-Tuning)  
6. Análises e visualizações

---

# 2. Extração das Representações

As representações devem ser pré-computadas antes de executar o TURTLE.

## CIFAR-9
```bash
python precompute_representations.py --dataset cifar9 --phis resnet18
python precompute_representations.py --dataset cifar9 --phis vgg11
```

## STL-9
```bash
python precompute_representations.py --dataset stl9 --phis resnet18
python precompute_representations.py --dataset stl9 --phis vgg11
```

---

# 3. Extração dos Rótulos

## CIFAR-9
```bash
python precompute_labels.py --dataset cifar9
```

## STL-9
```bash
python precompute_labels.py --dataset stl9
```

---

# 4. Representações da DIALNet

Para comparação direta com *Saltori et al.*, é necessário extrair as representações da **DIALNet**.

### Passo 1 — Acessar o diretório
```
low_budget_label_query
```

### Passo 2 — Executar `digits.py` em modo *single*

- Target = STL-9  → Source = CIFAR-9  
- Target = CIFAR-9 → Source = STL-9  

Exemplo:

```bash
python digits.py --root data/digits --source STL9 --target CIFAR9 --arch dialnet --mode single --scorer entropy --sampler random
```

> *Observação:* `scorer` e `sampler` não são usados no modo single, mas devem ser fornecidos para evitar erros.

### Passo 3 — Extrair as features com o modelo treinado

```bash
python extract_features.py --source STL9 --target CIFAR9 --data-root data/digits --num-classes 9 --model-arch-type cifar9-stl9
```

As representações serão salvas em `data/`.

---

# 5. Execução do TURTLE

Após extrair as representações, o TURTLE pode ser executado:

```bash
python run_turtle.py --dataset <DATASET> --phis <MODELO>
```

### Exemplo
```bash
python run_turtle.py --dataset cifar9 --phis vgg11
```

---

# 6. Seleção de Amostras Incertas

O script `uncertain_samples.py` identifica as amostras mais incertas geradas pelo TURTLE.

Ele realiza seleções em **valores absolutos** (1, 10, 100, 1000) e **percentuais** (0.1%, 1%, 10%, 20%), considerando três estratégias:

- **Aleatória**  
- **Mais incertas**  
- **Uniforme**  

### Uso
```bash
python uncertain_samples.py --dataset <DATASET> --phis <ESPAÇO1> <ESPAÇO2>
```

### Exemplo
```bash
python uncertain_samples.py --dataset stl9 --phis resnet18
```

> O TURTLE deve ter sido executado antes para gerar embeddings e checkpoints.

---

# 7. Ajuste Fino (Fine-Tuning)

O script `fine_tuning.py` realiza o ajuste fino com base nas amostras selecionadas.  
Os resultados são salvos automaticamente, incluindo métricas e hiperparâmetros:

- Estratégia de seleção  
- Quantidade de amostras  
- Taxa de aprendizagem  
- Número de épocas  
- Coeficiente de regularização  

### Uso
```bash
python fine_tuning.py --dataset <DATASET> --phis <ESPAÇO> --newTask <VALOR>
```

### Exemplo
```bash
python fine_tuning.py --dataset stl9 --phis resnet18 --newTask 1
```

---

# 8. Análises e Visualizações

Este repositório contém notebooks auxiliares:

- **entropia.ipynb**  
  Avalia o comportamento das diferentes estratégias de seleção.

- **graficos.ipynb**  
  Gera gráficos comparando:
  - TURTLE vs Fine-Tuning  
  - Diferentes espaços de representação (ResNet-18, VGG-11, DIALNet)  
  - Hiperparâmetros selecionados  
  - Resultados no CIFAR-9 e STL-9  

---