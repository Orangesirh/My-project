# 临时测试脚本 test_dataloader.py
import time
import torch
from utils.builder import get_dataloader
import json

with open('config/config.json', 'r') as f:
    config = json.load(f)

for num_workers in [2, 4, 6, 8, 12]:
    config['Dataset']['syntodd']['num_workers'] = num_workers
    
    loader = get_dataloader(config, "train")
    
    start = time.time()
    for i, batch in enumerate(loader):
        if i >= 100:  # 只测试100个batch
            break
    elapsed = time.time() - start
    
    print(f"num_workers={num_workers}: {elapsed:.2f}s ({100/elapsed:.2f} it/s)")