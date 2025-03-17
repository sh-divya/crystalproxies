import yaml
from pathlib import Path

import torch
from torch import nn

from dave.proxies.models import make_model, ProxyEmbeddingModel


class DavePredictor(nn.Module):
    def __init__(self, arch="physmlp", path_to_weights=None, device="cpu"):
        super().__init__()
        if isinstance(device, torch.device):
            pass
        elif isinstance(device, str):
            device = torch.device(device)
        else:
            raise TypeError(f"Please specify {device} as either string or torch object")
        if arch in ["physmlp"]:
            self.arch = arch
        else:
            raise ValueError(f"{arch} not available as saved weights")
        try:
            ckpt_path = Path(path_to_weights)
            self.ckpt_fptr = ckpt_path / f"{arch}.ckpt"
            ckpt = torch.load(self.ckpt_fptr, map_location=device)
            model_hp = ckpt["hyper_parameters"]

            self.ckpt_config = {}
            self.ckpt_config["model"] = model_hp["model"]

            self.target = model_hp["target"]
            if self.target == "Eform":
                self.ckpt_config["config"] = f"{arch}-mbform"
            elif self.target == "Band Gap":
                self.ckpt_config["config"] = f"{arch}-mbgap"
            self.ckpt_config["comp_size"] = model_hp["comp_size"]
            self.ckpt_config["lat_size"] = model_hp["lat_size"]
            self.ckpt_config["alphabet"] = model_hp["alphabet"]
            self.proxy = make_model(self.ckpt_config)

            self.proxy.load_state_dict(
                {k[6:]: v for k, v in ckpt["state_dict"].items()}
            )
            self.proxy.eval()

            self.xscale = model_hp["scales"]["x"]
            self.yscale = model_hp["scales"]["y"]
        except TypeError:
            print(f"Please enter path to where weights are store for {arch} model")
            raise Exception

    def __call__(self, x, scale_input=False):
        if scale_input:
            comp, sg, lat = x
            lat = (lat - self.xscale["mean"]) / self.xscale["std"]
            x = (comp, sg, lat)
        if self.yscale:
            return (self.proxy(x) * self.yscale["std"]) + self.yscale["mean"]
        else:
            return self.proxy(x)
