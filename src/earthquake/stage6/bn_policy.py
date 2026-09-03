"""BatchNorm train-mode helper: keep BN in eval so running stats stay pretrained."""

from __future__ import annotations

import torch


def set_train_bn_eval(model: torch.nn.Module) -> None:
    model.train()
    for mod in model.modules():
        if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
            mod.eval()
            for p in mod.parameters():
                p.requires_grad_(False)
