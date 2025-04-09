import sys
from pathlib import Path

DAVE_PATH = Path(__file__).parent.parent
sys.path.append(str(DAVE_PATH))

from dave.proxies.data import CrystalFeat
from dave import DavePredictor
from torch.utils.data import DataLoader
import torch

model = DavePredictor(
            arch="physmlp",
			path_to_weights="/network/scratch/d/divya.sharma/crystals/models/dave-proxy/mbform"
		)

root = Path("/network/scratch/d/divya.sharma/crystals/data") / "materials_dataset_v4"
wyck_max = 228
name = "matbench_mp_e_form"
tmp_root = root / "data" / name

sub = "train"
trans = {
    "x": {
        "mean": torch.load(str(tmp_root / "x.mean")),
        "std": torch.load(str(tmp_root / "x.std")),
    },
    
    "y": {
        "mean": torch.load(str(tmp_root / "y.mean")),
        "std": torch.load(str(tmp_root / "y.std")),
        },
}

temp = CrystalFeat(str(tmp_root), target="Eform", subset=sub, scalex=trans["x"], scaley=trans["y"], wyck_max=wyck_max)
lder = DataLoader(temp, batch_size=2, shuffle=True, pin_memory=True)
for item in lder:
    x, _ = item
    # print(x[0])
    # print(x[1])
    # print(x[2])
    print(x[3])
    print(x[3].shape)
    print(model(x))
    raise Exception
