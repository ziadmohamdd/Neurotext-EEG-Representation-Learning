from pathlib import Path
import math
import torch
from .checkpoints import save_checkpoint, load_checkpoint
from ..utils.logging import write_json

class Trainer:
    def __init__(self, model, config, device, output_dir):
        self.model, self.config, self.device = model.to(device), config, device
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                          lr=config.get("learning_rate", 1e-3), weight_decay=config.get("weight_decay", .01))
        self.amp = device.type == "cuda" and config.get("mixed_precision") == "fp16_if_supported"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)

    def epoch(self, loader, objective, training):
        self.model.train(training)
        self.optimizer.zero_grad(set_to_none=True)
        total, count = 0., 0
        accumulation = int(self.config.get("gradient_accumulation_steps", 8))
        if accumulation < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        for step, batch in enumerate(loader):
            batch = {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            with torch.set_grad_enabled(training), torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
                loss = objective(self.model, batch)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite objective")
            if training:
                # Correct scale for the final partial accumulation group.
                group_size = min(accumulation, len(loader) - (step // accumulation) * accumulation)
                self.scaler.scale(loss / group_size).backward()
                if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get("gradient_clip", 1.))
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach())
            count += 1
        if not count:
            raise ValueError("No batches executed")
        return total / count

    def fit(self, train, val, objective, metadata, resume=None):
        start, best, stale, history = 0, math.inf, 0, []
        if resume:
            previous = load_checkpoint(resume, self.model, self.optimizer, self.scaler, restore_rng=True)
            for key in ["split_digest", "config", "normalizer", "stage", "channels", "sfreq", "text_cache_digest", "training_control", "encoder_origin", "feature_scaler"]:
                if key in metadata and previous.get(key) != metadata[key]:
                    raise ValueError(f"Resume metadata mismatch: {key}")
            start, best, stale, history = previous["epoch"] + 1, previous["best"], previous["stale"], previous["history"]
        try:
            for epoch in range(start, int(self.config.get("epochs", 50))):
                if hasattr(getattr(train,"batch_sampler",None), "set_epoch"):
                    train.batch_sampler.set_epoch(epoch)
                # Epoch seeding makes shuffling and validation masks stable across resume.
                torch.manual_seed(int(self.config.get("seed", 42)) + epoch)
                train_loss = self.epoch(train, objective, True)
                with torch.random.fork_rng(devices=[self.device] if self.device.type == "cuda" else []):
                    torch.manual_seed(100_000 + int(self.config.get("seed", 42)))
                    val_loss = self.epoch(val, objective, False)
                improved = val_loss < best
                best, stale = (val_loss, 0) if improved else (best, stale + 1)
                history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
                state = dict(metadata, epoch=epoch, best=best, stale=stale, history=history)
                save_checkpoint(self.output / "last.pt", self.model, self.optimizer, self.scaler, **state)
                if improved:
                    save_checkpoint(self.output / "best.pt", self.model, self.optimizer, self.scaler, **state)
                write_json(self.output / "history.json", history)
                if stale >= self.config.get("patience", 8):
                    break
        except torch.OutOfMemoryError as exc:
            raise RuntimeError("CUDA out of memory: lower max_samples or batch_size (minimum 2 for contrastive alignment), freeze encoders, increase accumulation, or use Colab/CPU.") from exc
        return history
