"""Distribution metrics with explicit, stable definitions."""
import torch


def mmd_rbf(x, y, bandwidths=(0.05, 0.1, 0.2, 0.5, 1.0)) -> float:
    """Unbiased MMD², summed Gaussian kernels exp(-||x-y||²/(2h²)). May be negative."""
    x, y = x.detach().double(), y.detach().double()
    if len(x) < 2 or len(y) < 2:
        raise ValueError("unbiased MMD requires at least two samples per set")
    def kernel_sum(a, b, diagonal=False):
        total = torch.zeros((), dtype=torch.float64)
        for start in range(0, len(a), 256):
            distances = torch.cdist(a[start:start + 256], b).square()
            values = sum(torch.exp(-distances / (2 * h * h)) for h in bandwidths)
            total += values.sum()
        if diagonal:
            total -= len(a) * len(bandwidths)
        return total
    return float(kernel_sum(x, x, True) / (len(x) * (len(x) - 1))
                 + kernel_sum(y, y, True) / (len(y) * (len(y) - 1))
                 - 2 * kernel_sum(x, y) / (len(x) * len(y)))


def mode_coverage(samples, centers, radius):
    """Each sample belongs to its nearest mode only if within radius."""
    distance, nearest = torch.cdist(samples, centers.to(samples)).min(-1)
    inside = distance <= radius
    proportions = [float(((nearest == i) & inside).float().mean()) for i in range(len(centers))]
    return {"proportions": proportions, "outlier_fraction": float((~inside).float().mean()),
            "covered_modes": sum(p > 0 for p in proportions)}
