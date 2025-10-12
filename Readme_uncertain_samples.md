# Uncertain Samples

Este repositório contém o script `uncertain_samples.py`, que permite identificar amostras mais incertas rotuladas pelo TURTLE.

## Uso

Para executar o script, definir um dataset, os espaços de features e o número de amostras incertas a serem identificadas:

```bash
python uncertain_samples.py --dataset <DATASE> --phis <ESPAÇO1> <ESPAÇO2> --num_samples <NUM. AMOSTRAS>


Exemplo de uso:

Número de amostras: 10.
Dataset: stl10.
Espaços de representação: clipvitL14, dinov2.
Obs.: Previamente executar o TURTLE com esse dataset e esses espaços de representação. 

```bash
python uncertain_samples.py --dataset stl10 --phis clipvitL14 dinov2 --num_samples 10

