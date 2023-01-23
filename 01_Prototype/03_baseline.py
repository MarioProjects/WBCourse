##################################################
################## Imports #######################
##################################################
import os
import wandb
import pandas as pd

import torch

from torch.optim.lr_scheduler import MultiStepLR
import torchvision

from pytorch_lightning import Trainer
from pytorch_lightning import LightningModule

from torchmetrics import MeanMetric
from torchmetrics.functional import accuracy

import pretty_errors

import params

from utils import OrangesDataModule

##################################################
################## Main ##########################
##################################################

CONFIG = {
    "lr": 0.01,
    "batch_size": 8,
    "num_classes": 1,
    "num_epochs": 15,
    "optimizer": "SGD",
    "scheduler": "MultiStepLR",
    "milestones": [5, 10],
    "gamma": 0.1,
    "model": "resnet18",
    "pretrained": False,
    "loss": "BCEWithLogitsLoss",
}

# Initialize wandb run and download artifacts
run = wandb.init(
    project=params.WANDB_PROJECT,
    entity=params.ENTITY,
    config=CONFIG,
    job_type="training"
)
# Download the artifact with the raw data (images)
raw_data_at = run.use_artifact(
    'marioparreno/mlops-wandb-course/oranges:latest',
    type='raw_data'
)
data_dir = raw_data_at.download()

# Download the artifact with the split information (csv)
split_data_at = run.use_artifact(
    'marioparreno/mlops-wandb-course/oranges_split:latest',
    type='split_data'
)
artifact_dir = split_data_at.download()

best_model_artifact = wandb.Artifact('best_model', type='model')
last_model_artifact = wandb.Artifact('last_model', type='model')

#################################################
## Create the trainer -> train the model & log ##
#################################################

class LitResnet(LightningModule):
    def __init__(self, lr, num_classes, model, pretrained, loss, optimizer, scheduler, milestones, gamma):
        super().__init__()

        self.lr = lr
        if loss == "BCEWithLogitsLoss":
            self.criterion = torch.nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Loss not supported")
        if model == "resnet18":
            self.model = torchvision.models.resnet18(
                pretrained=pretrained, num_classes=num_classes
            )
        elif model == "resnet34":
            self.model = torchvision.models.resnet34(
                pretrained=pretrained, num_classes=num_classes
            )
        else:
            raise ValueError("Model not supported")

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.milestones = milestones
        self.gamma = gamma

        self.train_accuracy = MeanMetric()
        self.val_accuracy = MeanMetric()
        self.test_accuracy = MeanMetric()
        self.test_summary_table = wandb.Table(
            columns=["Images", "Is Rotten", "Predicted"]
        )
        self.best_val_acc = 0
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.unsqueeze(1).float())

        preds = torch.sigmoid(logits)
        preds = (preds > 0.5).float()

        step_acc = accuracy(preds.squeeze(), y.squeeze(), task="binary")
        self.train_accuracy.update(step_acc)

        return loss

    def training_epoch_end(self, training_step_outputs):
        train_epoch_accuracy = self.train_accuracy.compute()
        wandb.log(
            {"train_accuracy": train_epoch_accuracy},
            step=self.current_epoch
        )
        self.train_accuracy.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        preds = torch.sigmoid(logits)
        preds = (preds > 0.5).float()

        step_acc = accuracy(preds.squeeze(), y.squeeze(), task="binary")
        self.val_accuracy.update(step_acc)
    
    def validation_epoch_end(self, validation_step_outputs):
        val_epoch_accuracy = self.val_accuracy.compute()
        wandb.log(
            {"val_accuracy": val_epoch_accuracy},
            step=self.current_epoch
        )

        if val_epoch_accuracy > self.best_val_acc:
            self.best_val_acc = val_epoch_accuracy
            torch.save(self.model.state_dict(), "best_model.pt")

        self.val_accuracy.reset()

    def test_step(self, batch, batch_idx):

        x, y = batch
        logits = self(x)
        
        preds = torch.sigmoid(logits)

        for i in range(len(x)):
            image = x[i].cpu().permute(1, 2, 0).numpy()
            self.test_summary_table.add_data(
                wandb.Image(image),
                y[i].item(),
                preds[i].item()
            )

        preds = (preds > 0.5).float()
        step_acc = accuracy(preds.squeeze(), y.squeeze(), task="binary")
        self.test_accuracy.update(step_acc)
    
    def test_epoch_end(self, validation_step_outputs):
        test_epoch_accuracy = self.test_accuracy.compute()
        wandb.summary["test_accuracy"] = test_epoch_accuracy
        self.test_accuracy.reset()

        wandb.log({"test_summary_table": self.test_summary_table})

    def configure_optimizers(self):
        if self.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=5e-4,
            )
        else:
            raise ValueError("Optimizer not supported")
        
        if self.scheduler == "MultiStepLR":
            scheduler = MultiStepLR(
                optimizer,
                milestones=CONFIG["milestones"],
                gamma=CONFIG["gamma"]
            )
        else:
            raise ValueError("Scheduler not supported")
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


##################################################
################## Training ######################
##################################################

model = LitResnet(
    lr=CONFIG["lr"], 
    num_classes=CONFIG["num_classes"],
    model=CONFIG["model"],
    pretrained=CONFIG["pretrained"],
    loss=CONFIG["loss"],
    optimizer=CONFIG["optimizer"],
    scheduler=CONFIG["scheduler"],
    milestones=CONFIG["milestones"],
    gamma=CONFIG["gamma"]
)

trainer = Trainer(
    max_epochs=CONFIG["num_epochs"],
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    log_every_n_steps=5,
    num_sanity_val_steps=0,
)

df = pd.read_csv(os.path.join(artifact_dir, 'data_split.csv'))
dm = OrangesDataModule(os.path.join(data_dir, 'images'), df, CONFIG["batch_size"])

trainer.fit(model, dm)

# Save the models
torch.save(model.state_dict(), "last_model.pt")
last_model_artifact.add_file("last_model.pt")
run.log_artifact(last_model_artifact, aliases=["last"])

best_model_artifact.add_file("best_model.pt")
run.log_artifact(best_model_artifact, aliases=["best_model"])

# Run the test
trainer.test(model, datamodule=dm)

run.finish()