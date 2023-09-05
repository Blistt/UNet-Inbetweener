import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import _utils.calculator as cal
import kornia
from _utils.utils import get_edt



def get_gen_loss(preds, real, r1=None, r2=None, r3=None, lambr1=None, lambr3=None, lambr2=None, device='cuda:0'):
    
    gen_recon1 = r1(preds, real)
    gen_loss = gen_recon1 * lambr1

    # Adds optional additional losses
    if r2 is not None:
        gen_recon2 = r2(preds, real)
        gen_loss += gen_recon2 * lambr2
    if r3 is not None:
        gen_recon3 = r3(preds, real)
        gen_loss += gen_recon3 * lambr3
    return gen_loss


def gdl_loss(gen_frames, gt_frames, alpha=2, device='cuda:1'):
    filter_x = nn.Conv2d(1, 1, (1, 3), padding=(0, 1)).to(device)
    filter_y = nn.Conv2d(1, 1, (3, 1), padding=(1, 0)).to(device)

    gen_dx = filter_x(gen_frames)
    gen_dy = filter_y(gen_frames)
    gt_dx = filter_x(gt_frames)
    gt_dy = filter_y(gt_frames)

    grad_diff_x = torch.pow(torch.abs(gt_dx - gen_dx), alpha)
    grad_diff_y = torch.pow(torch.abs(gt_dy - gen_dy), alpha)

    grad_total = torch.stack([grad_diff_x, grad_diff_y])

    return torch.mean(grad_total)


class EDT_Loss(nn.Module):
    def __init__(self, device='cuda:1', sub_loss='l1'):
        super(EDT_Loss, self).__init__()
        self.device = device
        self.sub_loss = sub_loss

    def forward(self, preds, real):
        with torch.no_grad():
            # Gets the Euclidean Distance Transform of the real and predicted frames
            real_edt = get_edt(real, device=self.device).to(self.device)
            preds_edt = get_edt(preds, device=self.device).to(self.device)

        # Calculates laplacian pyramid loss of edt transformed images if specified
        if self.sub_loss == 'laplacian':
            preds_edt = kornia.geometry.transform.build_pyramid(preds, 3)
            real_edt = kornia.geometry.transform.build_pyramid(real, 3)
            loss = torch.stack([
                (p-t).abs().mean((1,2,3))
                for p,t in zip(preds_edt, real_edt)
            ]).mean(0)
            return loss.mean()
        
        # Defaults to L1 loss
        else:
            return F.l1_loss(preds_edt, real_edt)
        

class GDL(nn.Module):
    '''
    Gradience difference loss function
    Target: reduce motion blur 
    '''

    def __init__(self, device='cuda:1'):
        super(GDL, self).__init__()
        self.device = device

    def forward(self, gen_frames, gt_frames):
        return gdl_loss(gen_frames, gt_frames, device=self.device)

    
class MS_SSIM(nn.Module):
    '''
    Multi scale SSIM loss. Refer from:
    - https://github.com/jorge-pessoa/pytorch-msssim/blob/master/pytorch_msssim/__init__.py
    - https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py
    '''
    def __init__(self, device='cuda:1'):
        super(MS_SSIM, self).__init__()
        self.device = device

    def forward(self, gen_frames, gt_frames):
        return 1 - self._cal_ms_ssim(gen_frames, gt_frames)
        
    def _cal_ms_ssim(self, gen_tensors, gt_tensors):
        weights = torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        if torch.cuda.is_available():
            weights = weights.to(self.device)
        
        gen = gen_tensors
        gt = gt_tensors
        levels = weights.shape[0]
        mcs = []
        win_size = 3 if gen_tensors.shape[3] < 256 else 11
        win = cal._fspecial_gauss_1d(size=win_size, sigma=1.5)
        win = win.repeat(gen.shape[1], 1, 1, 1)
        
        for i in range(levels):
            ssim_per_channel, cs = cal._ssim_tensor(gen, gt, data_range=1.0, win=win)
            
            if i < levels - 1: 
                mcs.append(torch.relu(cs))
                padding = (gen.shape[2] % 2, gen.shape[3] % 2)
                gen = F.avg_pool2d(gen, kernel_size=2, padding=padding)
                gt = F.avg_pool2d(gt, kernel_size=2, padding=padding)
        
        ssim_per_channel = torch.relu(ssim_per_channel)  # (batch, channel)
        mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)  # (level, batch, channel)
        ms_ssim_val = torch.prod(mcs_and_ssim ** weights.view(-1, 1, 1), dim=0)   
        return ms_ssim_val.mean()
    
class LaplacianPyramidLoss(nn.Module):
    def __init__(self, n_levels=3, colorspace=None, mode='l1'):
        super().__init__()
        self.n_levels = n_levels
        self.colorspace = colorspace
        self.mode = mode
        assert self.mode in ['l1', 'l2']
        return
    def forward(self, preds, target, force_levels=None, force_mode=None):
        if self.colorspace=='lab':
            preds = kornia.color.rgb_to_lab(preds.float())
            target = kornia.color.rgb_to_lab(target.float())
        lvls = self.n_levels if force_levels==None else force_levels
        preds = kornia.geometry.transform.build_pyramid(preds, lvls)
        target = kornia.geometry.transform.build_pyramid(target, lvls)
        mode = self.mode if force_mode==None else force_mode
        if mode=='l1':
            ans = torch.stack([
                (p-t).abs().mean((1,2,3))
                for p,t in zip(preds,target)
            ]).mean(0)
        elif mode=='l2':
            ans = torch.stack([
                (p-t).norm(dim=1, keepdim=True).mean((1,2,3))
                for p,t in zip(preds,target)
            ]).mean(0)
        else:
            assert 0
        return ans.mean()
