"""PaperLeet paired-vector toy: no image/text encoder or original CLIP dataset."""
import math
import torch
import torch.nn.functional as F


def run(loss_function):
    generator = torch.Generator().manual_seed(42)
    images = torch.randn(8, 8, generator=generator, dtype=torch.float64).requires_grad_()
    texts = torch.randn(8, 8, generator=generator, dtype=torch.float64).requires_grad_()
    log_scale = torch.tensor(math.log(2), dtype=images.dtype, requires_grad=True)
    optimizer = torch.optim.Adam([images, texts, log_scale], lr=.06)
    losses = []
    for _ in range(80):
        optimizer.zero_grad()
        loss = loss_function(images, texts, log_scale)
        losses.append(loss.detach().item())
        loss.backward()
        optimizer.step()
    similarities = F.normalize(images.detach(), dim=-1) @ F.normalize(texts.detach(), dim=-1).T
    labels = torch.arange(8)
    return {"provenance": "PaperLeet teaching adaptation of CLIP Figure 3",
            "losses": losses, "similarities": similarities.tolist(),
            "image_to_text_accuracy": (similarities.argmax(1) == labels).double().mean().item(),
            "text_to_image_accuracy": (similarities.argmax(0) == labels).double().mean().item(),
            "initial_loss": losses[0], "final_loss": loss_function(images, texts, log_scale).detach().item(),
            "inverse_temperature": log_scale.detach().clamp(max=math.log(100)).exp().item()}
