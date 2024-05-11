def get_config():
    return {
        "image_size": 256,
        "batch_size": 16,
        "num_classes": 10,
        "dim": 768, # Hidden size (default 1024)
        "depth": 12, # layers (default 6)
        "heads": 12, # heads (default 8)
        "mlp_dim": 3072, # MLP size (default 2048)
        "channels": 3,
        "dropout": 0.1,
        "emb_dropout": 0.1,
        "learning_rate": 0.0001,
        "num_epoch": 5,
        "grad_accumulation_steps": 2,
        "model_folder": "weights",
        "model_basename": "vitmodel_",
    }


def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)
