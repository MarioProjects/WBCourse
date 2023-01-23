##################################################
################## Imports #######################
##################################################
import wandb
import pretty_errors

import torch
from pytorch_lightning import Trainer

import params
from logic import *
from arguments import config

##################################################
################## Main ##########################
##################################################

# Initialize wandb run and download artifacts
run = wandb.init(
    project=params.WANDB_PROJECT,
    entity=params.ENTITY,
    config=config,
    job_type="training"
)

dm = get_data(config)

model = LitResnet(config)

trainer = Trainer(
    max_epochs=config.num_epochs,
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,
    log_every_n_steps=5,
    num_sanity_val_steps=0,
)

trainer.fit(model, dm)
trainer.test(model, datamodule=dm)

log_model(model)

wandb.finish()