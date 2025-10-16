import argparse
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
from scipy.optimize import linear_sum_assignment
from utils import datasets_to_c

def load_task_encoder(checkpoint_path, feature_dims, C, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    task_encoder = [nn.Linear(d, C).to(device) for d in feature_dims]
    for i, task_phi in enumerate(task_encoder):
        task_phi.load_state_dict(checkpoint[f'phi{i+1}'])
    return task_encoder


def task_encoding(Zs, task_encoder):
    assert len(Zs) == len(task_encoder)
    label_per_space = [F.softmax(task_phi(z), dim=1) for task_phi, z in zip(task_encoder, Zs)]
    labels = torch.mean(torch.stack(label_per_space), dim=0)
    return labels, label_per_space


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--phis", nargs="+", required=True)
    parser.add_argument("--num_samples", type=int, default=10, help="Número de amostras mais incertas")
    parser.add_argument("--root_dir", type=str, default="./data")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dataset = args.dataset
    phis = args.phis
    root_dir = args.root_dir
    device = args.device
    num_samples = args.num_samples

    C = datasets_to_c[dataset]

    # ===========================================================
    # 1. Carregar representações e rótulos
    # ===========================================================
    Zs_train = [np.load(f"{root_dir}/representations/{phi}/{dataset}_train.npy").astype(np.float32) for phi in phis]
    # Zs_val = [np.load(f"{root_dir}/representations/{phi}/{dataset}_val.npy").astype(np.float32) for phi in phis]
    y_gt_train = np.load(f"{root_dir}/labels/{dataset}_train.npy")
    # y_gt_val = np.load(f"{root_dir}/labels/{dataset}_val.npy")

    feature_dims = [Z_train.shape[1] for Z_train in Zs_train]
    checkpoint_path = f"{root_dir}/task_checkpoints/{len(phis)}space/{'_'.join(phis)}/{dataset}/turtle_{'_'.join(phis)}_innerlr0.001_outerlr0.001_T6000_M10_coldstart_gamma10.0_bs10000_seed42.pt"

    # ===========================================================
    # 2. Carregar task encoder
    # ===========================================================
    task_encoder = load_task_encoder(checkpoint_path, feature_dims, C, device)
    print(f"Task encoder carregado de {checkpoint_path}")

    with torch.no_grad():
        # Zs_train é uma lista com as representações de cada espaço (dinov2, clip)
        # task encoder recebe essa lista e retorna:
        # vetor de predições combinadas
        # lista de label, onde cada item é um tensor de probabilidades (N,C)(N amostras, C classes) produzida por cada espaço
        Zs_train_torch = [torch.from_numpy(Z_train).to(args.device) for Z_train in Zs_train]
        _, label_per_space_train = task_encoding(Zs_train_torch, task_encoder)

        print(label_per_space_train)

        # ===========================================================
        # 3. Combinação das probabilidades
        # ===========================================================
        # Suponha 2 espaços (K=2), 2 classes (C=2):

        # label_per_space_train[0] = tensor([[0.8, 0.2],
        #                                    [0.3, 0.7]])

        # label_per_space_train[1] = tensor([[0.9, 0.1],
        #                                    [0.4, 0.6]])

        # como garantir que as colunas são correspondentes entre os diferentes classificadores lineares?
        #é garantida pela função objetivo!

        # como passar os embeddings para o classificador linear final?

        # Produto entre classificadores (cada phi contribui com suas probabilidades)
        # vetor de 1s por ser elemento neutro da multiplicação
        # multiplicação elemento a elemento
        combined = torch.ones_like(label_per_space_train[0])
        for probs in label_per_space_train:
            combined *= probs
        combined = combined / combined.sum(dim=1, keepdim=True)  # normalização

        # ===========================================================
        # 4. Incerteza
        # ===========================================================
        # combined = tensor([
        #   [0.80, 0.10, 0.10],  # amostra 1 - confiante na classe 0
        #   [0.34, 0.33, 0.33],  # amostra 2 - indecisa
        #   [0.05, 0.05, 0.90],  # amostra 3 - confiante na classe 2
        # ])

        # values = tensor([0.80, 0.34, 0.90])   # maior probabilidade por amostra

        # incerteza = tensor([0.2, 0.66, 0.10])
        uncertainty = 1 - combined.max(dim=1).values

        # Seleciona as N amostras mais incertas
        topk = torch.topk(uncertainty, k=num_samples)

        #trazer para CPU por conta do numpy
        indices_mais_incertos = topk.indices.cpu().numpy()

        # torch.topk(x, 3)
        # torch.return_types.topk(values=tensor([5., 4., 3.]), indices=tensor([4, 3, 2]))

        valores_incerteza = topk.values.cpu().numpy()

        print("Índices das amostras mais incertas (para rotulação manual - conjunto de treino):")
        for idx, u in zip(indices_mais_incertos, valores_incerteza):
            print(f"  Amostra {idx}: Incerteza = {u:.6f}")

        # ===========================================================
        # 5. Pseudo-labels e alinhamento
        # ===========================================================
        pred_labels_train = combined.argmax(dim=1).cpu().numpy()

        # A matriz w[i, j] vai contar quantas vezes o cluster previsto i correspondeu à classe real j
        # Isso ajuda a identificar o melhor mapeamento entre clusters e classes verdadeiras.
        D = max(pred_labels_train.max(), y_gt_train.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(pred_labels_train.size):
            w[pred_labels_train[i], y_gt_train[i]] += 1

        # Exemplo:
        # pred_labels_train = [1, 1, 0]
        # y_gt_train        = [0, 1, 2]
        #
        # w antes:
        # [[0, 0, 0],
        #  [0, 0, 0],
        #  [0, 0, 0]]
        #
        # w depois:
        # [[0, 0, 1], atribui 0 e era classe 2
        #  [1, 1, 0],
        #  [0, 0, 0]]

        # 3. Resolver assignment problem
        #Como atribuir n trabalhadores a n tarefas de forma que o custo total seja o menor possível
        # hungarian trabalha com minimização, por isso W.max() - w


        row_ind, col_ind = linear_sum_assignment(w.max() - w)
        # Exemplo:
        # row_ind = [0, 1, 2]
        # col_ind = [2, 1, 0]

        # Criar mapeamento de clusters previstos para labels reais
        match = np.zeros(D, dtype=np.int64)
        for r, c in zip(row_ind, col_ind):
            match[r] = c
            
        # match = [2, 1, 0]
        # significa:
        # se cluster = 0, ele corresponde à classe real 2
        # se cluster = 1, ele corresponde à classe real 1
        # se cluster = 2, ele corresponde à classe real 0

        # Criar o mapeamento inverso: da classe real -> cluster predito
        inverse_match = np.zeros_like(match)
        for pred, real in enumerate(match):
            inverse_match[real] = pred

        # ===========================================================
        # 5a. Alinhar previsões
        # ===========================================================
        # Alinhar previsões
        labels_train_aligned = match[pred_labels_train]
        # Exemplo:
        # pred_labels_train = [1, 1, 0]
        # match = [2, 1, 0] ==> 0 vira 2; 1 vira 1; 2 vira 0
        # labels_train_indices = [1, 1, 2]

        # Substituir amostras mais incertas pelos labels reais
        labels_train_aligned[indices_mais_incertos] = y_gt_train[indices_mais_incertos]


        # ===========================================================
        # 5b. Criar versões alinhada e não alinhada
        # ===========================================================

        # Versão: não alinhada (sem aplicar o match)
        labels_train_original = pred_labels_train.copy()

        # Converter rótulos reais (ground truth) para o espaço dos clusters originais
        y_gt_train_aligned_to_original = inverse_match[y_gt_train]

        # Substituir amostras mais incertas pelos rótulos reais convertidos
        labels_train_original[indices_mais_incertos] = y_gt_train_aligned_to_original[indices_mais_incertos]

                
        # Converter para tensor LongTensor para cross_entropy
        labels_train_aligned_tensor = torch.from_numpy(labels_train_aligned).long().to(device)
        labels_train_original_tensor = torch.from_numpy(labels_train_original).long().to(device)


        # ===========================================================
        # 6. Salvar representações combinadas de TREINO
        # ===========================================================
        save_dir = f"{root_dir}/results/{len(phis)}space/{'_'.join(phis)}/"
        os.makedirs(save_dir, exist_ok=True)

        torch.save(labels_train_aligned_tensor, os.path.join(save_dir, f"labels_train_aligned_{dataset}.pt"))
        torch.save(labels_train_original_tensor, os.path.join(save_dir, f"labels_train_original_{dataset}.pt"))

        print(f"\nArquivos salvos:")
        print(f" - Alinhado com ground truth: {os.path.join(save_dir, f'labels_train_aligned_{dataset}.pt')}")
        print(f" - Original (não alinhado):  {os.path.join(save_dir, f'labels_train_original_{dataset}.pt')}")
        # # ===========================================================
        # # 7. Salvar representações combinadas de TESTE
        # # ===========================================================
        # Zs_val_torch = [torch.from_numpy(Z_val).to(device) for Z_val in Zs_val]
        # y_val_tensor = torch.from_numpy(y_gt_val).long().squeeze().to(device)
        # representations_combined_test = [y_val_tensor] + Zs_val_torch
        # save_test = f"{root_dir}/results/{len(phis)}space/{'_'.join(phis)}/representations_combined_{dataset}_test.pt"
        # torch.save(representations_combined_test, save_test)
        # print(f"Salvo: {save_test}")


if __name__ == "__main__":
    main()
