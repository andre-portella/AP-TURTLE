import os
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np

# Imports do seu projeto
from model import Model
from datasets import CIFAR9, STL9
from torchvision import transforms
from utils import PathType


# ---------------------------------------------------------
# Carregador de Dataset
# ---------------------------------------------------------
def load_dataset(root_dir, dataset_name, transform, train=False):
    split = "Treino" if train else "Teste/Validação"
    print(f"Carregando dataset: {dataset_name} ({split} split)")

    if dataset_name == 'cifar9':
        return CIFAR9(root=root_dir, train=train, transform=transform, download=True)

    elif dataset_name == 'stl9':
        split_arg = 'train' if train else 'test'
        return STL9(root=root_dir, split=split_arg, transform=transform, download=True)

    raise NotImplementedError(f"Dataset {dataset_name} não implementado.")


# ---------------------------------------------------------
# Extração de Features
# ---------------------------------------------------------
def extract_features_and_save(data, model, split_name, batch_size=64, num_workers=4):

    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Coloca modelo na GPU
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    all_embeds, all_targets = [], []

    print(f"Iniciando extração de features para {split_name}...")

    with torch.no_grad():
        for x, y in loader:
            x = x.cuda()

            # Método preferencial
            if hasattr(model.module, "forward_features"):
                emb = model.module.forward_features(x)
            else:
                # Fallback para ResNet-like
                feats = model.module.base_model.features(x)
                emb = model.module.base_model.avgpool(feats)
                emb = torch.flatten(emb, 1)

            all_embeds.append(emb.cpu())
            all_targets.append(y.cpu())

    all_embeds = torch.cat(all_embeds)
    all_targets = torch.cat(all_targets)

    embeddings_np = all_embeds.numpy()
    targets_np = all_targets.numpy()

    # Diretórios de saída
    dir_representations = "../turtle/data/representations/dialnet"
    dir_labels = "../turtle/labels"

    os.makedirs(dir_representations, exist_ok=True)
    os.makedirs(dir_labels, exist_ok=True)

    embed_path = os.path.join(dir_representations, f"{split_name}.npy")
    target_path = os.path.join(dir_labels, f"{split_name}.npy")

    np.save(embed_path, embeddings_np)
    np.save(target_path, targets_np)

    print(f"[OK] {len(embeddings_np)} embeddings salvos em {embed_path}")
    print(f"[OK] {len(targets_np)} targets salvos em {target_path}")

    return embeddings_np, targets_np


# ---------------------------------------------------------
# Função Principal
# ---------------------------------------------------------
def feature_extraction_main():
    parser = argparse.ArgumentParser(description='Feature Extraction using Pre-trained DIALNet')

    parser.add_argument('--source', required=True)
    parser.add_argument('--target', required=True,
                        choices=['CIFAR9', 'STL9', 'MNIST', 'SVHN', 'USPS'])
    parser.add_argument('--data-root', type=PathType(exists=True, type='dir'), required=True)
    
    parser.add_argument('--arch', default='dialnet')
    parser.add_argument('--num-classes', type=int, default=9)
    parser.add_argument('--model-arch-type', default='cifar9-stl9')
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)

    args = parser.parse_args()

    args.source = args.source.lower()
    args.target = args.target.lower()

    # Caminho dos pesos do modelo
    model_path = os.path.join(
        "clusters/",
        f"digits_{args.source}_{args.target}_dialnet_single/",
        f"digits_{args.source}_{args.target}_dialnet_single_model_best.pth.tar"
    )

    # Criar modelo
    model_params = {'training_mode': 'single', 'arch': args.model_arch_type}
    model = Model(args.num_classes, base_model=args.arch, **model_params)

    # Carregar pesos
    print(f"\n=> Carregando pesos de: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')

    # Extrair state_dict
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Remover prefixo 'module.'
    from collections import OrderedDict
    cleaned_sd = OrderedDict(
        (key.replace("module.", ""), value)
        for key, value in state_dict.items()
    )

    model.load_state_dict(cleaned_sd)

    # -----------------------------------------------------
    # Normalização por dataset
    # -----------------------------------------------------
    resize = transforms.Resize(32)
    to_tensor = transforms.ToTensor()

    if args.target == 'cifar9':
        normalize = transforms.Normalize(
            mean=[0.424, 0.415, 0.384],
            std=[0.283, 0.278, 0.284]
        )

    elif args.target == 'stl9':
        normalize = transforms.Normalize(
            mean=[0.447, 0.440, 0.407],
            std=[0.260, 0.257, 0.271]
        )

    else:
        raise NotImplementedError(f"Normalização não definida para {args.target}")

    data_transform = transforms.Compose([
        resize,
        to_tensor,
        normalize
    ])

    # -----------------------------------------------------
    # Extração de TREINO
    # -----------------------------------------------------
    train_data = load_dataset(args.data_root, args.target, data_transform, train=True)

    extract_features_and_save(
        train_data,
        model,
        split_name=f'{args.target.lower()}_train',
        batch_size=args.batch_size,
        num_workers=args.workers
    )

    # -----------------------------------------------------
    # Extração de TESTE/VAL
    # -----------------------------------------------------
    val_data = load_dataset(args.data_root, args.target, data_transform, train=False)

    extract_features_and_save(
        val_data,
        model,
        split_name=f'{args.target.lower()}_val',
        batch_size=args.batch_size,
        num_workers=args.workers
    )

    print("\nExtração de Treino e Validação concluída.")


if __name__ == '__main__':
    feature_extraction_main()
