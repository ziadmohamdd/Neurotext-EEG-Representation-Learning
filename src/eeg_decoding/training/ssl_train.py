from ..models.ssl_encoder import MaskedEEG
from .trainer import Trainer

def train_ssl(train_loader, val_loader, channels, config, device, output, metadata, resume=None):
    model = MaskedEEG(channels, config["embedding_dim"], config.get("quality_aware", False))
    trainer = Trainer(model, config, device, output)
    history = trainer.fit(train_loader, val_loader, lambda m, b: m.loss(b["eeg"], b["mask"]), metadata, resume)
    return model, history
