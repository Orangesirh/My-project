# import json
# import numpy as np
# from utils.builder import get_dataloader
# from models.Trainer import Trainer


# ################ load the config file ##################
# with open('config/config.json', 'r') as f:
#     config = json.load(f)

# ############### load the trainer ###############
# trainer = Trainer(config)

# ############### train set ####################
# train_loader = get_dataloader(config, "train")
# ############### val set ####################
# val_loader = get_dataloader(config, "val")
# ############### test set ####################
# test_loader = get_dataloader(config, "test")

# ############### start testing ##############
# trainer.test(val_loader)

import json
from utils.builder import get_dataloader
from models.Trainer import Trainer

with open('config/config.json', 'r') as f:
    config = json.load(f)

trainer = Trainer(config)

test_subsets = [
    "test_heavy_occlusion",
    "test_new_background",
    "test_opaque_distractor",
    "test_translucent_cover",
    "test_non_planar",
    "test_filled_liquid",
]

for subset in test_subsets:
    print(f"\n{'='*60}")
    print(f"Testing on: {subset}")
    print(f"{'='*60}")
    test_loader = get_dataloader(config, subset)
    trainer.test(test_loader)