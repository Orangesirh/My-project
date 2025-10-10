"""
训练进度条工具
解决tqdm在多进程、wandb、Docker环境下的冲突问题
"""

import sys
from tqdm import tqdm


def create_train_progress_bar(dataloader, epoch=None, desc="Training"):
    """
    创建稳定的训练进度条
    
    Args:
        dataloader: PyTorch DataLoader
        epoch: 当前epoch数（可选）
        desc: 描述文本
    
    Returns:
        tqdm对象
    """
    if epoch is not None:
        desc = f"{desc} Epoch {epoch}"
    
    return tqdm(
        dataloader,
        desc=desc,
        ncols=100,              # 固定宽度
        file=sys.stdout,        # 明确输出流
        dynamic_ncols=False,    # 禁用动态宽度
        ascii=True,             # ASCII字符（兼容性好）
        leave=True,             # 完成后保留
        position=0,             # 固定位置
        mininterval=1.0,        # 最小更新间隔1秒
        maxinterval=10.0,       # 最大更新间隔10秒
        smoothing=0.1           # 平滑速度估计
    )


def create_val_progress_bar(dataloader, desc="Validation"):
    """
    创建验证进度条
    """
    return tqdm(
        dataloader,
        desc=desc,
        ncols=100,
        file=sys.stdout,
        dynamic_ncols=False,
        ascii=True,
        leave=True,
        mininterval=2.0  # 验证可以更新慢一点
    )


def safe_write(pbar, message):
    """
    安全地在进度条下方打印信息
    
    Args:
        pbar: tqdm对象
        message: 要打印的消息
    """
    if pbar is not None:
        pbar.write(message)
    else:
        print(message)