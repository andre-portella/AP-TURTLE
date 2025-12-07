import argparse
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from scipy.optimize import linear_sum_assignment
from utils import datasets_to_c
from samplers import RandomSampler, UncertaintySampler, UniformSampler


# ===========================================================
# Carrega task encoder (mantido, apenas organizado)
# ===========================================================
def load_task_encoder(checkpoint_path, feature_dims, C, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    task_encoder = [nn.Linear(d, C).to(device) for d in feature_dims]
    for i, task_phi in enumerate(task_encoder):
        task_phi.load_state_dict(checkpoint[f'phi{i+1}'])
    return task_encoder


# ===========================================================
# Combina classificadores (mantido, apenas organizado)
# ===========================================================
def task_encoding(Zs, task_encoder):
    assert len(Zs) == len(task_encoder)
    label_per_space = [F.softmax(task_phi(z), dim=1) for task_phi, z in zip(task_encoder, Zs)]
    labels = torch.mean(torch.stack(label_per_space), dim=0)
    return labels, label_per_space


# ===========================================================
# Função auxiliar: salva arquivos de rótulos
# ===========================================================
def save_labels(save_dir, prefix, dataset, Kname, aligned_tensor, original_tensor, idxs):
    torch.save(aligned_tensor, os.path.join(save_dir, f"{prefix}_aligned_{dataset}_{Kname}.pt"))
    torch.save(original_tensor, os.path.join(save_dir, f"{prefix}_original_{dataset}_{Kname}.pt"))
    torch.save(idxs, os.path.join(save_dir, f"{prefix}_indexes_{dataset}_{Kname}.pt"))

    print(f" - {prefix}: {os.path.join(save_dir, f'{prefix}_aligned_{dataset}_{Kname}.pt')}")
    print(f" - {prefix}: {os.path.join(save_dir, f'{prefix}_original_{dataset}_{Kname}.pt')}")
    print(f" - {prefix}: {os.path.join(save_dir, f'{prefix}_indexes_{dataset}_{Kname}.pt')}")


# ===========================================================
# Processa UM valor de K (pode ser abs ou pct)
# ===========================================================
def process_K(Kvalue, mode, N, Zs_train, phis, dataset, root_dir, device,
              task_encoder, feature_dims, combined, entropy, uncertainty,
              y_gt_train, match, inverse_match, pred_labels_train):

    if mode == "pct":
        num_samples = int((Kvalue / 100.0) * N)
        ratio = Kvalue / 100.0
        Kname = f"Kpct_{Kvalue}"
    else:
        num_samples = Kvalue
        ratio = num_samples / N
        Kname = f"Kabs_{Kvalue}"

    print(f"\n==========================")
    print(f"Processando {Kname}: num_samples = {num_samples}")
    print("==========================\n")

    save_dir = f"{root_dir}/results/{len(phis)}space/{'_'.join(phis)}/{Kname}/"
    os.makedirs(save_dir, exist_ok=True)

    # ===========================================================
    # Samplers
    # ===========================================================
    sampler_uncertainty = UncertaintySampler(ratio=ratio)
    selected_unc = sampler_uncertainty.select(uncertainty.detach().cpu().numpy())

    sampler_uniform = UniformSampler(ratio=ratio)
    selected_unif = sampler_uniform.select(entropy.detach().cpu().numpy())

    sampler_random = RandomSampler(ratio=ratio)

    # ===========================================================
    # Alinhamento
    # ===========================================================
    labels_train_aligned = match[pred_labels_train]
    labels_train_original = pred_labels_train.copy()
    y_gt_train_aligned_to_original = inverse_match[y_gt_train]

    # ===========================================================
    # UNCERTAINTY
    # ===========================================================
    labels_aligned_unc = labels_train_aligned.copy()
    labels_orig_unc = labels_train_original.copy()

    labels_aligned_unc[selected_unc] = y_gt_train[selected_unc]
    labels_orig_unc[selected_unc] = y_gt_train_aligned_to_original[selected_unc]

    save_labels(
        save_dir, "uncertain", dataset, Kname,
        torch.from_numpy(labels_aligned_unc).long().to(device),
        torch.from_numpy(labels_orig_unc).long().to(device),
        selected_unc
    )

    # ===========================================================
    # UNIFORM
    # ===========================================================
    labels_aligned_unif = labels_train_aligned.copy()
    labels_orig_unif = labels_train_original.copy()

    labels_aligned_unif[selected_unif] = y_gt_train[selected_unif]
    labels_orig_unif[selected_unif] = y_gt_train_aligned_to_original[selected_unif]

    save_labels(
        save_dir, "uniform", dataset, Kname,
        torch.from_numpy(labels_aligned_unif).long().to(device),
        torch.from_numpy(labels_orig_unif).long().to(device),
        selected_unif
    )

    # ===========================================================
    # RANDOM — 10 REPETIÇÕES
    # ===========================================================
    for i in range(1, 11):
        print(f"\n===> Gerando amostragem aleatória {i}/10 ({Kname})")

        selected_rand = sampler_random.select(len(Zs_train[0]))

        labels_orig_rand = labels_train_original.copy()
        labels_aligned_rand = labels_train_aligned.copy()

        labels_orig_rand[selected_rand] = y_gt_train_aligned_to_original[selected_rand]
        labels_aligned_rand[selected_rand] = y_gt_train[selected_rand]

        save_labels(
            save_dir, f"random_{i}", dataset, Kname,
            torch.from_numpy(labels_aligned_rand).long().to(device),
            torch.from_numpy(labels_orig_rand).long().to(device),
            selected_rand
        )


# ===========================================================
# MAIN
# ===========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--phis", nargs="+", required=True)
    parser.add_argument("--root_dir", type=str, default="./data")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")


    parser.add_argument("--k_pct", nargs="+", type=float, default=[0.1, 1, 10, 20, 30])
    parser.add_argument("--k_abs", nargs="+", type=int, default=[1, 10, 100, 1000])

    args = parser.parse_args()

    dataset = args.dataset
    phis = args.phis
    root_dir = args.root_dir
    device = args.device

    C = datasets_to_c[dataset]

    # ===========================================================
    # 1. Carregar representações
    # ===========================================================
    Zs_train = [np.load(f"{root_dir}/representations/{phi}/{dataset}_train.npy").astype(np.float32)
                for phi in phis]
    y_gt_train = np.load(f"{root_dir}/labels/{dataset}_train.npy")

    feature_dims = [Z.shape[1] for Z in Zs_train]

    checkpoint_path = (
        f"{root_dir}/task_checkpoints/{len(phis)}space/{'_'.join(phis)}/{dataset}/"
        f"turtle_{'_'.join(phis)}_innerlr0.001_outerlr0.001_T6000_M10_coldstart_gamma10.0_bs10000_seed42.pt"
    )

    # ===========================================================
    # 2. Carregar task encoder
    # ===========================================================
    task_encoder = load_task_encoder(checkpoint_path, feature_dims, C, device)
    print(f"Task encoder carregado de {checkpoint_path}")

    # ===========================================================
    # 3. Executar predições, entropia, incerteza
    # ===========================================================
    Zs_train_torch = [torch.from_numpy(Z).to(device) for Z in Zs_train]

    _, label_per_space = task_encoding(Zs_train_torch, task_encoder)

    if len(label_per_space) == 1:
        combined = label_per_space[0]
    else:
        combined = torch.stack(label_per_space).mean(dim=0)
        combined = combined / combined.sum(dim=1, keepdim=True)

    # ENTROPIA
    entropies = [-(p * torch.log(torch.clamp(p, min=sys.float_info.epsilon))).sum(dim=1)
                 for p in label_per_space]

    entropy = torch.stack(entropies).mean(dim=0) if len(entropies) > 1 else entropies[0]

    # ===========================================================
    # Salvar entropia (por espaço e média final)
    # ===========================================================
    entropy_dir = f"{root_dir}/results/{len(phis)}space/{'_'.join(phis)}/"
    os.makedirs(entropy_dir, exist_ok=True)

    # Salvar entropia média (final)
    torch.save(entropy.cpu(), os.path.join(entropy_dir, f"entropy_mean_{dataset}.pt"))

    # # Se houver múltiplos espaços de representação, salvar a entropia individual
    # if len(entropies) > 1:
    #     for i, ent in enumerate(entropies):
    #         torch.save(ent.cpu(), os.path.join(entropy_dir, f"entropy_space{i+1}_{dataset}.pt"))

    print(f"Entropias salvas em {entropy_dir}")

    # KL + ENTROPIA
    if len(label_per_space) > 1:
        kl = torch.stack([
            (p * (torch.log(p + 1e-12) - torch.log(combined + 1e-12))).sum(dim=1)
            for p in label_per_space
        ]).mean(dim=0)
        uncertainty = kl * entropy
    else:
        uncertainty = entropy

    # ===========================================================
    # alinhamento
    # ===========================================================
    pred_labels_train = combined.argmax(dim=1).cpu().numpy()

    D = max(pred_labels_train.max(), y_gt_train.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    for i in range(pred_labels_train.size):
        w[pred_labels_train[i], y_gt_train[i]] += 1

    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    match = np.zeros(D, dtype=np.int64)
    for r, c in zip(row_ind, col_ind):
        match[r] = c

    inverse_match = np.zeros_like(match)
    for pred, real in enumerate(match):
        inverse_match[real] = pred

    # ===========================================================
    # 4. Processar TODOS OS Ks (pct + abs)
    # ===========================================================
    N = len(Zs_train[0])

    for k in args.k_pct:
        process_K(k, "pct", N, Zs_train, phis, dataset, root_dir, device,
                  task_encoder, feature_dims, combined, entropy, uncertainty,
                  y_gt_train, match, inverse_match, pred_labels_train)

    for k in args.k_abs:
        process_K(k, "abs", N, Zs_train, phis, dataset, root_dir, device,
                  task_encoder, feature_dims, combined, entropy, uncertainty,
                  y_gt_train, match, inverse_match, pred_labels_train)


# ===========================================================
# RUN
# ===========================================================
if __name__ == "__main__":
    main()
