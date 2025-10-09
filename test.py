import json
import numpy as np
from utils.builder import get_dataloader
from models.Trainer import Trainer


################ load the config file ##################
with open('config/config.json', 'r') as f:
    config = json.load(f)

############### load the trainer ###############
trainer = Trainer(config)

############### train set ####################
train_loader = get_dataloader(config, "train")
############### val set ####################
val_loader = get_dataloader(config, "val")
############### test set ####################
test_loader = get_dataloader(config, "test")

############### start testing ##############
trainer.test(val_loader)
