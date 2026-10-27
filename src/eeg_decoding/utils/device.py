import torch

def device_info(request="auto"):
    name = ("cuda" if torch.cuda.is_available() else "cpu") if request == "auto" else request
    device = torch.device(name)
    info = {"device": str(device), "torch": str(torch.__version__)}
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        info.update(gpu=torch.cuda.get_device_name(device), free_gib=free / 2**30, total_gib=total / 2**30)
    return device, info

def conservative_batch(config, info):
    batch = int(config.get("batch_size", 2))
    if config.get("scale_batch_by_vram") and "free_gib" in info:
        batch = min(batch, max(2, int(info["free_gib"] // 2)))
    return batch
