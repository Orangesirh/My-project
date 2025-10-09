import json
from models.Trainer import Trainer
from utils.builder import get_dataloader


################ load the config file ##################
with open('config/config.json', 'r') as f:
    config = json.load(f)
# np.random.seed(config['Trainer']['seed'])

############### load the trainer ###############
trainer = Trainer(config)

############### train set ####################
train_loader = get_dataloader(config, "train")
############### val set ####################
val_loader = get_dataloader(config, "val")

############### start training ##############
trainer.train(train_loader, val_loader)



