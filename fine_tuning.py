import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import pandas as pd

from utils import seed_everything, get_cluster_acc, datasets_to_c


def encoder(checkpoint_path, feature_dims, C, newTask, device):
    if newTask:
        return [nn.Linear(d, C).to(device) for d in feature_dims]
    else:
        return load_task_encoder(checkpoint_path, feature_dims, C, device)


# ============================================================
# Carrega Task Encoder
# ============================================================



def load_task_encoder(checkpoint_path, feature_dims, C, device):
    task_encoder = [nn.Linear(d, C).to(device) for d in feature_dims]
    
    if not os.path.exists(checkpoint_path):
        print(f"[AVISO] Checkpoint não encontrado: {checkpoint_path}")
        return task_encoder

    try:
        ckpt = torch.load(checkpoint_path, map_location=device)
        for phi, saved in zip(task_encoder, ckpt.values()):
            phi.load_state_dict(saved)
    except Exception as e:
        print(f"[ERRO] Falha ao carregar checkpoint: {e}")
    
    return task_encoder


# ============================================================
# Gera predições médias (TURTLE)
# ============================================================

def task_encoding(Zs, task_encoder):
    label_per_space = [F.softmax(phi(z), dim=1) for phi, z in zip(task_encoder, Zs)]
    labels = torch.mean(torch.stack(label_per_space), dim=0)
    return labels, label_per_space



# ============================================================
# Fine-tuning TURTLE
# ============================================================


def carregar_datasets(root_dir, dataset, phis):
    Zs_train = [np.load(f"{root_dir}/representations/{phi}/{dataset}_train.npy").astype(np.float32) for phi in phis]
    Zs_val   = [np.load(f"{root_dir}/representations/{phi}/{dataset}_val.npy").astype(np.float32) for phi in phis]
    y_gt_val = np.load(f"{root_dir}/labels/{dataset}_val.npy")

    return Zs_train, Zs_val, y_gt_val


def finetune(
    Zs_train, Zs_val, y_gt_val, root_dir, dataset, phis,
    C, device, outer_lr=1e-3, epochs=200, gamma=10.0,
    seed=42, K_dir="", strategy="random", run_id=1,
    wd=0.0, newTask=0, verbose=True, task_encoder=None
):
    seed_everything(seed)

    # ----------------------------------------
    # Carrega representações
    # ----------------------------------------

    # Caminho dos arquivos
    base = f"{root_dir}/results/{len(phis)}space/{'_'.join(phis)}/{K_dir}"

    # ----------------------------------------
    # Estratégias: random / uncertain / uniform
    # ----------------------------------------

    if strategy == "random":
        labels_path = f"{base}/random_{run_id}_original_{dataset}_{K_dir}.pt"
        idx_path    = f"{base}/random_{run_id}_indexes_{dataset}_{K_dir}.pt"

    elif strategy == "uncertain":
        labels_path = f"{base}/uncertain_original_{dataset}_{K_dir}.pt"
        idx_path    = f"{base}/uncertain_indexes_{dataset}_{K_dir}.pt"

    elif strategy == "uniform":
        labels_path = f"{base}/uniform_original_{dataset}_{K_dir}.pt"
        idx_path    = f"{base}/uniform_indexes_{dataset}_{K_dir}.pt"

    else:
        raise ValueError("Estratégia inválida")

    if not os.path.exists(labels_path):
        print(f"[ERRO] labels_path não existe: {labels_path}")
        return 0, 0
    if not os.path.exists(idx_path):
        print(f"[ERRO] idx_path não existe: {idx_path}")
        return 0, 0

    labels_train_tensor = torch.load(labels_path, map_location=device).long()
    indices = torch.load(idx_path, map_location=device, weights_only=False)


    # Filtra apenas índices selecionados
    Zs_train = [Z[indices] for Z in Zs_train]
    labels_train_tensor = labels_train_tensor[indices]
    labels_train_tensor = F.one_hot(labels_train_tensor, num_classes=C).float()

    Zs_train_tensors = [torch.from_numpy(Z).to(device) for Z in Zs_train]

    labels = labels_train_tensor.to(device)

    # ----------------------------------------
    # Inicializa task encoder
    # ----------------------------------------
    
    optimizer = torch.optim.Adam(
        sum([list(phi.parameters()) for phi in task_encoder], []),
        lr=outer_lr
    )

    # ----------------------------------------
    # Treino
    # ----------------------------------------
    for epoch in range(epochs):
        optimizer.zero_grad()

        loss_outer = sum([
            F.cross_entropy(task_phi(z), labels)
            for task_phi, z in zip(task_encoder, Zs_train_tensors)
        ])

        _, per_space = task_encoding(Zs_train_tensors, task_encoder)
        entr_reg = sum([torch.special.entr(l.mean(0)).sum() for l in per_space])

        loss = loss_outer - gamma * entr_reg
        loss.backward()
        optimizer.step()

    # ----------------------------------------
    # Avaliação
    # ----------------------------------------
    with torch.no_grad():
        labels_val, _ = task_encoding(
            [torch.from_numpy(Z).to(device) for Z in Zs_val],
            task_encoder
        )
        preds = labels_val.argmax(dim=1).cpu().numpy()
        acc, _ = get_cluster_acc(preds, y_gt_val)

    return acc, loss.item()


def acuracia_inicial(Zs_val, y_gt_val, task_encoder, device):
    with torch.no_grad():
        labels_val_init, _ = task_encoding(
            [torch.from_numpy(Z).to(device) for Z in Zs_val],
            task_encoder
        )
        preds_init = labels_val_init.argmax(dim=1).cpu().numpy()
        acc_init, _ = get_cluster_acc(preds_init, y_gt_val)

        return acc_init



# ============================================================
# GRID SEARCH COMPLETO SOBRE TODOS OS Ks
# ============================================================

def main(args=None):
    args = _parse_args(args)
    dataset = args.dataset
    phis = args.phis
    root = args.root_dir
    device = args.device
    newTask = args.newTask

    C = datasets_to_c[dataset]

    checkpoint_path = (
        f"{root}/task_checkpoints/{len(phis)}space/{'_'.join(phis)}/{dataset}/"
        f"turtle_{'_'.join(phis)}_innerlr0.001_outerlr0.001_T6000_M10_coldstart_gamma10.0_bs10000_seed42.pt"
    )

    porcentagem = [0.1, 1, 10, 20]
    absolutos   = [1, 10, 100, 1000]

    Ks = [f"Kpct_{p}" for p in porcentagem] + [f"Kabs_{a}" for a in absolutos]

    strategies = ["random", "uncertain", "uniform"]

    learning_rates = [1e-3, 1e-4, 1e-5]
    gammas = [0, 5, 10]
    epochs_list = [200, 400, 800]

    results = []

    Zs_train, Zs_val, y_gt_val = carregar_datasets(root_dir=root, dataset=dataset, phis=phis)
    feature_dims = [Z.shape[1] for Z in Zs_train]


    task_encoder = encoder(checkpoint_path, feature_dims, C, newTask, device)

    print("\n====================================================")
    print("INICIANDO GRID SEARCH PARA TODOS OS Ks")
    print("====================================================\n")
    acc_init = acuracia_inicial(Zs_val, y_gt_val, task_encoder, device)

    for K_dir in Ks:
        print(f"\n================ K = {K_dir} ================\n")
        print(f"[INIT] Acc inicial (sem fine-tuning) para {K_dir}: {acc_init:.4f}")

        best_K_acc = -1
        best_K_params = None

        for lr in learning_rates:
            for gamma in gammas:
                for epochs in epochs_list:

                    for strategy in strategies:

                        if strategy == "random":
                            accs = []
                            for run in range(1, 11):
                                task_encoder = encoder(checkpoint_path, feature_dims, C, newTask, device)
                                acc, _ = finetune(
                                    Zs_train, Zs_val, y_gt_val,
                                    root, dataset, phis,
                                    C, device,
                                    outer_lr=lr, epochs=epochs, gamma=gamma,
                                    seed=args.seed,
                                    K_dir=K_dir,
                                    strategy=strategy,
                                    run_id=run,
                                    newTask=newTask,
                                    verbose=False,
                                    task_encoder=task_encoder
                                )

                                accs.append(acc)

                            acc_mean = np.mean(accs)
                            acc_std = np.std(accs)
                            acc_value = acc_mean

                        else:
                            task_encoder = encoder(checkpoint_path, feature_dims, C, newTask, device)
                            acc_value, _ = finetune(
                                Zs_train, Zs_val, y_gt_val,
                                root, dataset, phis,
                                C, device,
                                outer_lr=lr, epochs=epochs, gamma=gamma,
                                seed=args.seed,
                                K_dir=K_dir,
                                strategy=strategy,
                                verbose=False,
                                task_encoder=task_encoder
                            )
                            acc_std = 0

                        results.append([K_dir, strategy, lr, gamma, epochs, acc_value, acc_std])

                        print(f"K={K_dir:10} | {strategy:8} | LR={lr:.0e} | G={gamma} | E={epochs} "
                              f"-> Acc={acc_value:.4f} (±{acc_std:.4f})")

                        if acc_value > best_K_acc:
                            best_K_acc = acc_value
                            best_K_params = (strategy, lr, gamma, epochs)


        print(f"\n>>> MELHOR RESULTADO PARA {K_dir}: {best_K_acc:.4f} usando {best_K_params}\n")

    df = pd.DataFrame(results, columns=[
        "K", "strategy", "lr", "gamma", "epochs", "acc", "acc_std"
    ])

    grid_result = f"results_grid_search/grid_results_{dataset}_{'_'.join(phis)}_{newTask}.csv"

    df.to_csv(grid_result, index=False)

    print("\n====================================================")
    print(f"GRID COMPLETO SALVO EM {grid_result}")
    print("====================================================\n")



# ============================================================
# Parser
# ============================================================

def _parse_args(args):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--phis", type=str, nargs="+", required=True)
    p.add_argument("--root_dir", type=str, default="./data")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--newTask", type=int, default=0)
    return p.parse_args(args)


if __name__ == "__main__":
    main()
