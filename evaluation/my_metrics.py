import torch
from torch import nn
import torchmetrics
import skimage
import kornia
from torchvision.utils import save_image
from utils import normalize


class SSIMMetric(torchmetrics.Metric):
    # torchmetrics has memory leak
    def __init__(self, window_size=11, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.add_state('running_sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('running_count', default=torch.tensor(0.0), dist_reduce_fx='sum')
        return
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        ans = kornia.losses.ssim(target, preds, self.window_size).mean((1,2,3))
        self.running_sum += ans.sum()
        self.running_count += len(ans)
        return
    def compute(self):
        return self.running_sum.float() / self.running_count
    
class SSIMMetricCPU(torchmetrics.Metric):
    full_state_update=False
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state('running_sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('running_count', default=torch.tensor(0.0), dist_reduce_fx='sum')
        return
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        ans = [
            skimage.metrics.structural_similarity(
                p.squeeze().cpu().numpy(),
                t.squeeze().cpu().numpy(),
                multichannel=False,
                gaussian=True,
                data_range=1.0,
            )
            for p,t in zip(preds, target)
        ]
        self.running_sum += sum(ans)
        self.running_count += len(ans)
        return
    def compute(self):
        return self.running_sum / self.running_count

class PSNRMetric(torchmetrics.Metric):
    # torchmetrics averages samples before taking log
    def __init__(self, data_range=1.0, **kwargs):
        super().__init__(**kwargs)
        self.data_range = torch.tensor(data_range)
        self.add_state('running_sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('running_count', default=torch.tensor(0.0), dist_reduce_fx='sum')
        return
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        ans = -10 * torch.log10( (target-preds).pow(2).mean((1,2,3)) )
        self.running_sum += 20*torch.log10(self.data_range) + ans.sum()
        self.running_count += len(ans)
        return
    def compute(self):
        return self.running_sum.float() / self.running_count
class PSNRMetricCPU(torchmetrics.Metric):
    full_state_update=False
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state('running_sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('running_count', default=torch.tensor(0.0), dist_reduce_fx='sum')
        return
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        ans = [
            skimage.metrics.peak_signal_noise_ratio(
                p.permute(1,2,0).cpu().numpy(),
                t.permute(1,2,0).cpu().numpy(),
            )
            for p,t in zip(preds, target)
        ]
        self.running_sum += sum(ans)
        self.running_count += len(ans)
        return
    def compute(self):
        return self.running_sum / self.running_count
