import json
from models.Trainer import Trainer
from utils.visualize import *


image_path = "datasets/Image_ClearPose/set2_s3_1817-color.png"

################ load the config file ##################
with open('config/config.json', 'r') as f:
    config = json.load(f)

############### load the trainer ###############
trainer = Trainer(config)

############### start inference ##############
trainer.inference(image_path)
