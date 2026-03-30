import json
from models.Trainer import Trainer
from utils.visualize import *


image_path = "datasets/image/12418.png"

################ load the config file ##################
with open('config/config.json', 'r') as f:
    config = json.load(f)

############### load the trainer ###############
trainer = Trainer(config)

############### start inference ##############
trainer.inference(image_path)