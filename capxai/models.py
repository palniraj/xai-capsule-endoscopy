"""Backbones, Grad-CAM target layers and layer orders for the randomisation test."""
from __future__ import annotations

import torch
from torch import nn
from torchvision import models as tvm

# arch -> (constructor, classifier attribute, Grad-CAM layer, randomisation cascade top->bottom)
ARCHS = {
    "resnet18": (tvm.resnet18, "fc", "layer4",
                 ["fc", "layer4", "layer3", "layer2", "layer1", "conv1"]),
    "resnet50": (tvm.resnet50, "fc", "layer4",
                 ["fc", "layer4", "layer3", "layer2", "layer1", "conv1"]),
    "densenet121": (tvm.densenet121, "classifier", "features.norm5",
                    ["classifier", "features.denseblock4", "features.denseblock3",
                     "features.denseblock2", "features.denseblock1", "features.conv0"]),
}


def build_model(arch: str, n_classes: int, pretrained: bool = True) -> nn.Module:
    if arch not in ARCHS:
        raise ValueError(f"Unknown arch {arch!r}; choose from {sorted(ARCHS)}")
    ctor, head, _, _ = ARCHS[arch]
    model = ctor(weights="DEFAULT" if pretrained else None)
    old = getattr(model, head)
    setattr(model, head, nn.Linear(old.in_features, n_classes))
    return model


def gradcam_layer(model: nn.Module, arch: str) -> nn.Module:
    return model.get_submodule(ARCHS[arch][2])


def randomisation_cascade(arch: str) -> list[str]:
    return list(ARCHS[arch][3])


@torch.no_grad()
def reinit_module(module: nn.Module, generator_seed: int = 0):
    """Re-initialise every parameterised sub-module (Adebayo et al., 2018)."""
    torch.manual_seed(generator_seed)
    for m in module.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()


def save_checkpoint(path, model, arch, classes, extra=None):
    torch.save({"arch": arch, "classes": list(classes), "state_dict": model.state_dict(),
                **(extra or {})}, path)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["arch"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
