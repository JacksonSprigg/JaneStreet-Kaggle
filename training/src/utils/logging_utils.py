import wandb
from config import config

def init_wandb():
    wandb.require("core") # TODO: This line fixes a bug, but I don't remember what rn
    return wandb.init(
        project=config.PROJECT_NAME,
        dir=config.WANDB_SAVE_DIR,
        name=config.RUN_NAME,
        config=config.LGBM_PARAMS,
        tags=["baseline", "lightgbm", "offline-training"],
    )

def log_error(e: Exception):
    wandb.alert(
        title="Training Failed",
        text=f"Error occurred: {str(e)}"
    )