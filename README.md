
# Uncertain Samples

Este repositório contém o script `uncertain_samples.py`, que permite identificar as amostras mais incertas rotuladas pelo **TURTLE**, um método de aprendizado de representação e clustering.

## Estrutura do script

O script recebe três parâmetros principais:

- `--dataset` : Nome do dataset a ser usado (ex.: \`stl10\`, \`cifar10\`)  
- `--phis` : Lista de espaços de features que serão utilizados (ex.: \`clipvitL14 dinov2\`)  
- `--k` : Porcentagem de amostras rotuladas manualmente 

---

## Uso

Para executar o script, use o seguinte comando:

```bash
python uncertain_samples.py --dataset <DATASET> --phis <ESPAÇO1> <ESPAÇO2> --num_samples <NÚMERO_DE_AMOSTRAS>
```

### Exemplo de uso

Selecionar 10% amostras incertas do dataset `stl10` usando o espaço de representação `ResNet50`:

```bash
python uncertain_samples.py --dataset stl10 --phis clipRN50 --k 10
```

> **Observação:** É necessário **executar previamente o TURTLE** com o mesmo dataset e espaços de features antes de rodar este script, para que os embeddings e checkpoints estejam disponíveis.

---

## Estrutura de saída

O script retorna uma lista de amostras incertas.



# Fine-Tuning (versão 2)

Este repositório contém o script `fine_tuning_v2.py`, que realiza o ajuste fino do **TURTLE**.


## Estrutura do script

O script recebe três parâmetros principais:

- `--dataset` : Nome do dataset a ser usado (ex.: \`stl10\`, \`cifar10\`)  
- `--phis` : Lista de espaços de features que serão utilizados (ex.: \`clipvitL14 dinov2\`)  
- `--k` : Porcentagem de amostras rotuladas manualmente 


Para executar o script, use o seguinte comando:

```bash
python fine_tuning_new.py --dataset <DATASET> --phis <ESPAÇO1> <ESPAÇO2> --k <%>
```


### Exemplo de uso

Fine-tuning do `stl10`, considerando a rotulação de 10% das amostras incertas e o uso do espaço de representação `ResNet50`:

```bash
python fine_tuning_new.py --dataset stl10 --phis clipRN50  --k 10
```