import torch
import torchmetrics
import cupy
import numpy as np
import scipy
from torchvision.utils import save_image



# CuPy kernel launcher
try:
    # @cupy.memoize(for_each_device=True)
    def cupy_launch(func, kernel):
        return cupy.cuda.compile_with_cache(kernel).get_function(func)
except:
    cupy_launch = lambda func,kernel: None

'''
DISTANCE TRANSFORM:
Computes the distance transform of a batch of images

- img tensor: (bs,h,w) or (bs,1,h,w)
- returns same shape
- expects white lines, black whitespace
- defaults to diameter if empty image

'''
_batch_edt_kernel = ('kernel_dt', '''
    extern "C" __global__ void kernel_dt(
        const int bs,
        const int h,
        const int w,
        const float diam2,
        float* data,
        float* output
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= bs*h*w) {
            return;
        }
        int pb = idx / (h*w);
        int pi = (idx - h*w*pb) / w;
        int pj = (idx - h*w*pb - w*pi);

        float cost;
        float mincost = diam2;
        for (int j = 0; j < w; j++) {
            cost = data[h*w*pb + w*pi + j] + (pj-j)*(pj-j);
            if (cost < mincost) {
                mincost = cost;
            }
        }
        output[idx] = mincost;
        return;
    }
''')
_batch_edt = None
def batch_edt(img, block=1024):
    # must initialize cuda/cupy after forking
    global _batch_edt
    if _batch_edt is None:
        _batch_edt = cupy_launch(*_batch_edt_kernel)
        print('using cupy edt')

    # bookkeeppingg
    if len(img.shape)==4:
        assert img.shape[1]==1
        img = img.squeeze(1)
        expand = True
    else:
        expand = False
    bs,h,w = img.shape
    diam2 = h**2 + w**2
    odtype = img.dtype
    grid = (img.nelement()+block-1) // block

    # cupy implementation
    if img.is_cuda:
        # first pass, y-axis
        data = ((1-img.type(torch.float32)) * diam2).contiguous()
        intermed = torch.zeros_like(data)
        _batch_edt(
            grid=(grid, 1, 1),
            block=(block, 1, 1),  # < 1024
            args=[
                cupy.int32(bs),
                cupy.int32(h),
                cupy.int32(w),
                cupy.float32(diam2),
                data.data_ptr(),
                intermed.data_ptr(),
            ],
        )
        
        # second pass, x-axis
        intermed = intermed.permute(0,2,1).contiguous()
        out = torch.zeros_like(intermed)
        _batch_edt(
            grid=(grid, 1, 1),
            block=(block, 1, 1),
            args=[
                cupy.int32(bs),
                cupy.int32(w),
                cupy.int32(h),
                cupy.float32(diam2),
                intermed.data_ptr(),
                out.data_ptr(),
            ],
        )
        ans = out.permute(0,2,1).sqrt()
        ans = ans.type(odtype) if odtype!=ans.dtype else ans
    
    # default to scipy cpu implementation
    else:
        sums = img.sum(dim=(1,2))
        ans = torch.tensor(np.stack([
            scipy.ndimage.morphology.distance_transform_edt(i)
            if s!=0 else  # change scipy behavior for empty image
            np.ones_like(i) * np.sqrt(diam2)
            for i,s in zip(1-img, sums)
        ]), dtype=odtype)

    if expand:
        ans = ans.unsqueeze(1)
    return ans


'''
CHAMFER DISTANCE:
Computes the chamfer distance between the prediction and ground truth in a distance transformed 
batch of images

- input: (bs,h,w) or (bs,1,h,w)
- returns: (bs,)
- normalized s.t. metric is same across proportional image scales
'''
# Overall distance
def batch_chamfer_distance(gt, pred, block=1024, return_more=False, bit_reverse=True, binarize_at=0.5):

    # Binarize images
    gt = (gt>binarize_at).float()
    pred = (pred>binarize_at).float()

    if bit_reverse:
        gt = 1-gt
        pred = 1-pred
    t = batch_chamfer_distance_t(gt, pred, block=block)
    p = batch_chamfer_distance_p(gt, pred, block=block)
    cd = (t + p) / 2
    return cd

# Distance from gt to pred
def batch_chamfer_distance_t(gt, pred, block=1024, return_more=False):
    assert gt.device==pred.device and gt.shape==pred.shape
    bs,h,w = gt.shape[0], gt.shape[-2], gt.shape[-1]
    dpred = batch_edt(pred, block=block)
    cd = (gt*dpred).float().mean((-2,-1)) / np.sqrt(h**2+w**2)
    if len(cd.shape)==2:
        assert cd.shape[1]==1
        cd = cd.squeeze(1)
    return cd

# Distance from pred to gt
def batch_chamfer_distance_p(gt, pred, block=1024, return_more=False):
    assert gt.device==pred.device and gt.shape==pred.shape
    bs,h,w = gt.shape[0], gt.shape[-2], gt.shape[-1]
    dgt = batch_edt(gt, block=block)
    cd = (pred*dgt).float().mean((-2,-1)) / np.sqrt(h**2+w**2)
    if len(cd.shape)==2:
        assert cd.shape[1]==1
        cd = cd.squeeze(1)
    return cd



'''
TORCHMETRICS CLASSES
'''
class ChamferDistance2dMetric(torchmetrics.Metric):
    full_state_update=False
    def __init__(
            self, block=1024, binary=0.5,
            **kwargs,
        ):
        super().__init__(**kwargs)
        self.block = block
        self.binary = binary
        self.add_state('running_sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('running_count', default=torch.tensor(0.0), dist_reduce_fx='sum')
        return
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        dist = batch_chamfer_distance(target, preds, block=self.block, binarize_at=self.binary)
        self.running_sum += dist.sum()
        self.running_count += len(dist)
        return
    def compute(self):
        return self.running_sum.float() / self.running_count
class ChamferDistance2dTMetric(ChamferDistance2dMetric):
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        dist = batch_chamfer_distance_t(target, preds, block=self.block, binarize_at=self.binary)
        self.running_sum += dist.sum()
        self.running_count += len(dist)
        return
class ChamferDistance2dPMetric(ChamferDistance2dMetric):
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        dist = batch_chamfer_distance_p(target, preds, block=self.block, binarize_at=self.binary)
        self.running_sum += dist.sum()
        self.running_count += len(dist)
        return