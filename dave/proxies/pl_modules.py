import time

import pytorch_lightning as pl
import torch.optim as optim
from torchmetrics import MeanAbsoluteError, MeanSquaredError

from dave.utils.gnn import preprocess_data


class ProxyModule(pl.LightningModule):
    def __init__(self, proxy, loss, config):
        super().__init__()
        self.model_name = config["config"].split("-")[0]
        if self.model_name in ["fae", "faecry", "sch", "pyxtal_faenet"]:
            self.preproc_method = "graph"
        self.model = proxy
        self.criterion = loss
        self.lr = config["optim"]["lr"]
        self.loss = 0
        self.config = {k: v for k, v in config.items() if k not in ["root", "src"]}
        self.mae = MeanAbsoluteError()
        self.mse = MeanSquaredError()
        self.best_mae = 10e6
        self.best_mse = 10e6
        self.save_hyperparameters(self.config)
        self.active_logger = config.get("debug") is None
        self.preproc_method = False

    def training_step(self, batch, batch_idx):
        x, y = preprocess_data(batch, self.preproc_method)
        out = self.model(x, batch_idx).squeeze(-1)
        if self.model_name == "pyxtal_faenet":
            loss = self.criterion(out, y, batch)
        else:
            loss = self.criterion(out, y)
        mae = self.mae(out, y)
        mse = self.mse(out, y)
        lr = self.optimizers().param_groups[0]["lr"]

        self.log("train_loss", loss)
        self.log("train_mae", mae)
        self.log("train_mse", mse)
        self.log("learning_rate", lr)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = preprocess_data(batch, self.preproc_method)
        out = self.model(x, batch_idx).squeeze(-1)
        if self.model_name == "pyxtal_faenet":
            loss = self.criterion(out, y, batch)
        else:
            loss = self.criterion(out, y)
        mae = self.mae(out, y)
        mse = self.mse(out, y)

        self.log("val_loss", loss)
        self.log("val_mae", mae)
        self.log("val_mse", mse)
        return loss

    def on_validation_epoch_end(self):
        total_val_mae = self.mae.compute()
        total_val_mse = self.mse.compute()

        self.log("total_val_mae", total_val_mae)
        self.log("total_val_mse", total_val_mse)

        if total_val_mae < self.best_mae:
            self.best_mae = total_val_mae
        if total_val_mse < self.best_mse:
            self.best_mse = total_val_mse
        self.mae.reset()
        self.mse.reset()

    def on_validation_end(self) -> None:
        if self.active_logger:
            self.logger.experiment.summary["Best MAE"] = self.best_mae
            self.logger.experiment.summary["Best MSE"] = self.best_mse
        else:
            print(f"\nBest MAE: {self.best_mae}\n")

    def test_step(self, batch, batch_idx):
        if self.preproc_method == "graph":
            x = batch
        else:
            x, _ = batch
        s = time.time()
        _ = self.model(x, batch_idx).squeeze(-1)
        if self.preproc_method == "graph":
            batch_size = batch.num_graphs
        else:
            batch_size = batch[0][0].shape[0]
        sample_inf_time = (time.time() - s) / batch_size

        self.log("sample_inf_time", sample_inf_time, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), self.lr)
        if self.config["optim"]["scheduler"]["name"] == "ReduceLROnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=self.config["optim"]["scheduler"]["decay_factor"],
                patience=self.config["optim"]["scheduler"].get("patience")
                or self.config["optim"]["es_patience"],
            )
        elif self.config["optim"]["scheduler"]["name"] == "StepLR":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config["optim"]["scheduler"]["step_size"],
                gamma=self.config["optim"]["scheduler"]["decay_factor"],
            )
        else:
            scheduler = None
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_mae",
            },
        }
