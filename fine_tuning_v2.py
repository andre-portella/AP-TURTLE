#!/usr/bin/env python3
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from utils import seed_everything, get_cluster_acc, datasets_to_c


def load_task_encoder(checkpoint_path, feature_dims, C, device):
    """Carrega o task encoder treinado pelo TURTLE (classificadores lineares)."""
    task_encoder = [nn.Linear(d, C).to(device) for d in feature_dims]
    checkpoint = torch.load(checkpoint_path, map_location=device)
    for task_phi, ckpt_phi in zip(task_encoder, checkpoint.values()):
        task_phi.load_state_dict(ckpt_phi)
    return task_encoder


def task_encoding(Zs, task_encoder):
    """
    Gera as predições médias dos espaços de representação.
    Cada phi (representação) passa por um classificador linear (task_phi),
    e as probabilidades são combinadas por média.
    """
    assert len(Zs) == len(task_encoder)
    label_per_space = [F.softmax(task_phi(z), dim=1) for task_phi, z in zip(task_encoder, Zs)]
    labels = torch.mean(torch.stack(label_per_space), dim=0)
    return labels, label_per_space


def init_inner(feature_dims, C, inner_lr, device):
    """Inicializa os classificadores lineares do inner loop."""
    W_in = [nn.Linear(d, C).to(device) for d in feature_dims]
    inner_opt = torch.optim.Adam(sum([list(W.parameters()) for W in W_in], []),
                                 lr=inner_lr, betas=(0.9, 0.999))
    return W_in, inner_opt


# ============================================================
# Função principal de Fine-Tuning
# ============================================================

def finetune(
    root_dir, dataset, phis, checkpoint_path,
    C, device, inner_lr=1e-3, outer_lr=1e-3,
    batch_size=10000, epochs=200, M=10, gamma=10.0,
    warm_start=False, seed=42,
    K=10
):
    seed_everything(seed)

    # ------------------------------------------------------------
    # Carrega representações e rótulos verdadeiros de validação
    # ------------------------------------------------------------
    Zs_train = [np.load(f"{root_dir}/representations/{phi}/{dataset}_train.npy").astype(np.float32) for phi in phis]
    Zs_val = [np.load(f"{root_dir}/representations/{phi}/{dataset}_val.npy").astype(np.float32) for phi in phis]
    y_gt_val = np.load(f"{root_dir}/labels/{dataset}_val.npy")

    n_tr = Zs_train[0].shape[0]
    batch_size = min(batch_size, n_tr)
    feature_dims = [Z_train.shape[1] for Z_train in Zs_train]

    # ------------------------------------------------------------
    # Carrega o conjunto híbrido de rótulos
    # ------------------------------------------------------------
    labels_path = f"{root_dir}/results/{len(phis)}space/{'_'.join(phis)}/labels_train_original_{dataset}_{K}.pt"
    labels_train_tensor = torch.load(labels_path, map_location=device).long()
    
    # ------------------------------------------------------------
    # Carregar índices das amostras mais incertas
    # ------------------------------------------------------------
    indices_incertos_path = f"{root_dir}/results/{len(phis)}space/{'_'.join(phis)}/indices_incertos_{dataset}_{K}.pt"
    indices_incertos = torch.load(indices_incertos_path, map_location=device, weights_only=False)

    # Máscara indicando quais amostras possuem rótulos confiáveis
    mask_fixed_full = torch.zeros_like(labels_train_tensor, dtype=torch.bool)
    mask_fixed_full[indices_incertos] = True

    # ------------------------------------------------------------
    # Inicializa task encoder e otimizadores
    # ------------------------------------------------------------
    task_encoder = load_task_encoder(checkpoint_path, feature_dims, C, device)
    optimizer = torch.optim.Adam(sum([list(phi.parameters()) for phi in task_encoder], []),
                                 lr=outer_lr, betas=(0.9, 0.999))

    # Classificadores do inner loop
    W_in, inner_opt = init_inner(feature_dims, C, inner_lr, device)

    # ------------------------------------------------------------
    # Loop de treinamento (outer loop)
    # ------------------------------------------------------------
    iters = epochs
    for it in range(iters):
        # Seleciona um batch aleatório
        indices = np.random.choice(n_tr, size=batch_size, replace=False)
        Zs_tr = [torch.from_numpy(Z_train[indices]).to(device) for Z_train in Zs_train]

        # Predições do task encoder no batch
        labels_batch_pred, label_per_space_batch = task_encoding(Zs_tr, task_encoder)

        # Seleciona os rótulos fixos e máscara correspondente
        labels_fixed_batch = labels_train_tensor[indices].to(device)
        mask_fixed_batch = mask_fixed_full[indices].to(device)

        # One-hot dos rótulos fixos
        labels_onehot_batch = F.one_hot(labels_fixed_batch.clamp(min=0), num_classes=C).float()

        # Combina rótulos fixos (onde existem) e predições (onde não há correção)
        combined_labels_inner = torch.where(
            mask_fixed_batch.unsqueeze(1),
            labels_onehot_batch,
            labels_batch_pred.detach()
        )

        # ------------------------------------------------------------
        # Inner loop: ajusta W_in para aproximar combined_labels_inner
        # ------------------------------------------------------------
        if not warm_start:
            W_in, inner_opt = init_inner(feature_dims, C, inner_lr, device)

        for _ in range(M):
            inner_opt.zero_grad()
            # cross-entropy
            # loss_inner = sum([
            #     -(combined_labels_inner * F.log_softmax(w_in(z), dim=1)).sum(dim=1).mean()
            #     for w_in, z in zip(W_in, Zs_tr)
            # ])

            loss_inner = sum([F.cross_entropy(w_in(z_tr), combined_labels_inner.detach()) for w_in, z_tr in zip(W_in, Zs_tr)])
            loss_inner.backward()
            inner_opt.step()

        # ------------------------------------------------------------
        # Outer loop: atualiza o task encoder
        # ------------------------------------------------------------
        optimizer.zero_grad()

        # Perda de outer: comparação entre predições do encoder e saída dos classificadores
        # loss_outer = sum([
        #     -(labels_batch_pred * F.log_softmax(w_in(z).detach(), dim=1)).sum(dim=1).mean()
        #     for w_in, z in zip(W_in, Zs_tr)
        # ])

        loss_outer = sum([F.cross_entropy(w_in(z_tr).detach(), labels_batch_pred) for w_in, z_tr in zip(W_in, Zs_tr)])

        # loss_outer = sum([
        #     -(labels_batch_pred * F.log_softmax(w_in(z).detach(), dim=1)).sum(dim=1).mean()
        #     for w_in, z in zip(W_in, Zs_tr)
        # ])

        # Regularização por entropia (promove diversidade de classes)
        entr_reg = sum([torch.special.entr(l.mean(0)).sum() for l in label_per_space_batch])

        # Perda total
        loss_final = loss_outer - (gamma * entr_reg if gamma is not None else 0.0)

        loss_final.backward()
        optimizer.step()

        # ------------------------------------------------------------
        # Avaliação intermediária
        # ------------------------------------------------------------
        if (it + 1) % max(1, iters // 20) == 0 or (it + 1) == iters:
            with torch.no_grad():
                labels_val, _ = task_encoding([torch.from_numpy(Z_val_i).to(device) for Z_val_i in Zs_val], task_encoder)
                preds_val = labels_val.argmax(dim=1).cpu().numpy()
                cluster_acc, _ = get_cluster_acc(preds_val, y_gt_val)

            # tqdm.write(
            #     f'Iter {it+1}/{iters}: '
            #     f'inner_loss {loss_inner.detach().item():.4f}, '
            #     f'outer_loss {loss_outer.detach().item():.4f}, '
            #     f'entropy_reg {entr_reg.detach().item():.4f}, '
            #     f'cluster_acc {cluster_acc:.4f}'
            # )

    # ------------------------------------------------------------
    # Avaliação final
    # ------------------------------------------------------------
    with torch.no_grad():
        labels_val, _ = task_encoding(
            [torch.from_numpy(Z_val_i).to(device) for Z_val_i in Zs_val],
            task_encoder
        )
        preds_val = labels_val.argmax(dim=1).cpu().numpy()
        acc_val, _ = get_cluster_acc(preds_val, y_gt_val)

    return acc_val



def _parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--phis', type=str, nargs='+', required=True)
    parser.add_argument('--root_dir', type=str, default='./data')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--correcoes_path', type=str, default=None,
                        help='(opcional) caminho para dicionário de correções (indices_mais_incertos e labels_corretos_inverse)')
    return parser.parse_args(args)


def main(args=None):
    args = _parse_args(args)
    phis = args.phis
    C = datasets_to_c[args.dataset]

    checkpoint_path = f"{args.root_dir}/task_checkpoints/{len(phis)}space/{'_'.join(phis)}/{args.dataset}/" \
                      f"turtle_{'_'.join(phis)}_innerlr0.001_outerlr0.001_T6000_M10_coldstart_gamma10.0_bs10000_seed42.pt"


    learning_rates = [1e-4, 5e-4]
    gammas = [10]
    batch_sizes = [2000, 5000, 10000]
    epochs = [200, 400, 800]
    results = []

    for lr in learning_rates:
        for gamma in gammas:
            for bs in batch_sizes:
                for epoch in epochs:
                    acc = finetune(
                        root_dir=args.root_dir,
                        dataset=args.dataset,
                        phis=phis,
                        checkpoint_path=checkpoint_path,
                        C=C,
                        device=args.device,
                        inner_lr=lr,
                        outer_lr=lr,
                        batch_size=bs,
                        epochs=epoch,
                        M=10,
                        gamma=gamma,
                        warm_start=False,
                        seed=args.seed,
                        K=args.k
                    )

                    results.append((lr, gamma, bs, acc))
                    print(f"[lr={lr}, gamma={gamma}, bs={bs}, epoch={epoch}] -> ValAcc={acc:.4f}")


if __name__ == '__main__':
    main()
