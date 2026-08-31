import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .abstract_class import AbstractEvaluator
import torch.distributed as dist
import os
from sklearn.metrics import f1_score
from model.training_scripts.utils import misc

class ImageF1NoRemain(AbstractEvaluator):
    def __init__(self, threshold = 0.5) -> None:
        super().__init__()
        self.name = "image-level F1"
        self.desc = "image-level F1"
        self.threshold = threshold
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.cnt = 0
    def batch_update(self, predict_label, label, *args, **kwargs):
        self._chekc_image_level_params(predict_label, label)
        predict = (predict_label > self.threshold).float()
        self.TP += torch.sum(predict * label).item()
        self.TN += torch.sum((1-predict) * (1-label)).item()
        self.FP += torch.sum(predict * (1-label)).item()
        self.FN += torch.sum((1-predict) * label).item()
        self.cnt += len(predict_label)
        return None
    
    def epoch_update(self):
        t = torch.tensor([self.TP, self.TN, self.FP, self.FN, self.cnt],  dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        TP = t[0].item()
        TN = t[1].item()
        FP = t[2].item()
        FN = t[3].item()
        cnt = t[4].item()
        precision = TP / ( TP +  FP + 1e-9)
        recall =  TP / ( TP +  FN + 1e-9)
        F1 = 2 * precision * recall / (precision + recall + 1e-9)
        # F1 = torch.mean(F1) # fuse the Batch dimension
        return F1
    def recovery(self):
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.cnt = 0

class ImageF1(AbstractEvaluator):
    def __init__(self, threshold=0.5) -> None:
        super().__init__() 
        self.name = "image-level F1"
        self.desc = "image-level F1"
        self.threshold = threshold
        self.predict = []
        self.label = []
        self.remain_label = []
        self.remain_predict = []
        self.world_size = misc.get_world_size()
        self.local_rank = misc.get_rank()

    def batch_update(self, predict_label, label, *args, **kwargs):
        self._chekc_image_level_params(predict_label, label)
        self.predict.append(predict_label)
        self.label.append(label)
        return None
        
    def remain_update(self, predict_label, label, *args, **kwargs):
        self.remain_predict.append(predict_label)
        self.remain_label.append(label)
        return None

    def epoch_update(self):
        if len(self.predict) != 0:
            predict = torch.cat(self.predict, dim=0)
            label = torch.cat(self.label, dim=0)
            gather_predict_list = [torch.zeros_like(predict) for _ in range(self.world_size)]
            gather_label_list = [torch.zeros_like(label) for _ in range(self.world_size)]
            dist.all_gather(gather_predict_list, predict)
            dist.all_gather(gather_label_list, label)
            gather_predict = torch.cat(gather_predict_list, dim=0)
            gather_label = torch.cat(gather_label_list, dim=0) 
            if len(self.remain_predict) != 0:
                self.remain_predict = torch.cat(self.remain_predict, dim=0)
                self.remain_label = torch.cat(self.remain_label, dim=0)
                gather_predict = torch.cat([gather_predict, self.remain_predict], dim=0)
                gather_label = torch.cat([gather_label, self.remain_label], dim=0)
        else:
            if len(self.remain_predict) == 0:
                raise RuntimeError(f"No data to calculate {self.name}, please check the input data.")
            gather_predict = torch.cat(self.remain_predict, dim=0)
            gather_label = torch.cat(self.remain_label, dim=0)
        # calculate F1
        predict = (gather_predict > self.threshold) * 1.0
        TP = torch.sum(predict * gather_label)
        # TN = torch.sum((1-predict) * (1-gather_label)).item()
        FP = torch.sum(predict * (1-gather_label))
        FN = torch.sum((1-predict) * gather_label)
        precision = TP / (TP + FP + 1e-9)
        recall = TP / (TP + FN + 1e-9)
        F1 = 2 * precision * recall / (precision + recall + 1e-9)
        # F1 = torch.mean(F1) # fuse the Batch dimension
        return F1
    def recovery(self):
        self.predict = []
        self.label = []
        self.remain_label = []
        self.remain_predict = []
        return None
            
class PixelF1(AbstractEvaluator):
    def __init__(self, threshold = 0.5, mode = "origin") -> None:
        super().__init__()
        self.name = "pixel-level F1"
        self.desc = "pixel-level F1"
        self.threshold = threshold
        self.image_num = 0
        #  mode : "origin, reverse, double"
        self.mode = mode

    def Cal_Confusion_Matrix(self, predict, mask, shape_mask):
        """compute local confusion matrix for a batch of predict and target masks
        Args:
            predict (_type_): _description_
            mask (_type_): _description_
            region (_type_): _description_
            
        Returns:
            TP, TN, FP, FN
        """
        threshold = self.threshold
        predict = (predict > threshold).float()
        if(shape_mask != None):
            TP = torch.sum(predict * mask * shape_mask, dim=(1, 2, 3))
            TN = torch.sum((1-predict) * (1-mask) * shape_mask, dim=(1, 2, 3))
            FP = torch.sum(predict * (1-mask) * shape_mask, dim=(1, 2, 3))
            FN = torch.sum((1-predict) * mask * shape_mask, dim=(1, 2, 3))
        else:
            TP = torch.sum(predict * mask, dim=(1, 2, 3))  
            TN = torch.sum((1-predict) * (1-mask), dim=(1, 2, 3)) 
            FP = torch.sum(predict * (1-mask), dim=(1, 2, 3)) 
            FN = torch.sum((1-predict) * mask, dim=(1, 2, 3))         
        return TP, TN, FP, FN

    def Cal_Reverse_Confusion_Matrix(self, predict, mask, shape_mask):
        """compute local confusion matrix for a batch of predict and target masks
        Args:
            predict (_type_): _description_
            mask (_type_): _description_
            region (_type_): _description_
            
        Returns:
            TP, TN, FP, FN
        """
        threshold = self.threshold
        predict = (predict > threshold).float()
        if(shape_mask != None):
            TP = torch.sum((1-predict) * mask * shape_mask, dim=(1, 2, 3))
            TN = torch.sum(predict * (1-mask) * shape_mask, dim=(1, 2, 3))
            FP = torch.sum((1-predict) * (1-mask) * shape_mask, dim=(1, 2, 3))
            FN = torch.sum(predict * mask * shape_mask, dim=(1, 2, 3))
        else:
            TP = torch.sum((1-predict) * mask, dim=(1, 2, 3))
            TN = torch.sum(predict * (1-mask), dim=(1, 2, 3))
            FP = torch.sum((1-predict) * (1-mask), dim=(1, 2, 3))
            FN = torch.sum(predict * mask, dim=(1, 2, 3))
        return TP, TN, FP, FN

    def Cal_F1(self, TP, TN, FP, FN):
        """_summary_

        Args:
            TP (_type_): _description_
            TN (_type_): _description_
            FP (_type_): _description_
            FN (_type_): _description_

        Returns:
            _type_: _description_
        """
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        F1 = 2 * precision * recall / (precision + recall + 1e-8)
        # F1 = torch.mean(F1) # fuse the Batch dimension
        return F1

    def batch_update(self, predict, mask, shape_mask=None, *args, **kwargs): # 注意这里只有pixel-level需要的信息
        self._check_pixel_level_params(predict, mask)
        if self.mode == "origin":
            TP, TN, FP, FN = self.Cal_Confusion_Matrix(predict, mask, shape_mask)
            F1 = self.Cal_F1(TP, TN, FP, FN)
        elif self.mode == "reverse":
            TP, TN, FP, FN = self.Cal_Reverse_Confusion_Matrix(predict, mask, shape_mask)
            F1 = self.Cal_F1(TP, TN, FP, FN)
        elif self.mode == "double":
            # todo
            TP, TN, FP, FN = self.Cal_Confusion_Matrix(predict, mask, shape_mask)
            F1 = torch.max(self.Cal_F1(TP, TN, FP, FN), self.Cal_F1(FN, FP, TN, TP))
        else:
            raise RuntimeError(f"Cal_F1 no mode name {self.mode}")
        
        return F1
    
    def remain_update(self, predict, mask, shape_mask=None, *args, **kwargs):
        return self.batch_update(predict, mask, shape_mask, *args, **kwargs)
    
    def epoch_update(self):

        return None
    def recovery(self):
        self.image_num = 0


def test_origin_image_f1():
    # test imageF1
    # 初始化分布式环境
    dist.init_process_group(backend='nccl', init_method='env://')
    
    num_gpus = torch.cuda.device_count()
    if dist.get_rank() == 0:
        print("number of GPUS", num_gpus)
    
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    
    DATA_LEN = 200
    float_tensor = torch.rand( DATA_LEN * num_gpus).cuda(local_rank)  # 生成一个长度为 5 的浮点数 tensor

    # 生成一个包含 0 或 1 的整数 tensor
    int_tensor = torch.randint(0, 2, (DATA_LEN * num_gpus,)).cuda(local_rank)
    # print(float_tensor)
    # print(int_tensor)
    
    evaluator = ImageF1(threshold=0.5)
    dist.barrier()
    dist.broadcast(float_tensor, src=0)
    dist.broadcast(int_tensor, src=0)
    # 收集所有的预测标签和真实标签，用于之后的 sklearn 验证
    all_predicts = []
    all_labels = []
    
    
    if dist.get_rank() != num_gpus - 1:
        idx = dist.get_rank() * DATA_LEN
        predict_labels = float_tensor[idx: idx + DATA_LEN].cuda(local_rank)
        true_labels = int_tensor[idx: idx + DATA_LEN].cuda(local_rank)
    else:
        idx = dist.get_rank() * DATA_LEN
        predict_labels = float_tensor[idx: idx + DATA_LEN-50].cuda(local_rank)
        true_labels = int_tensor[idx: idx + DATA_LEN-50].cuda(local_rank)


    if dist.get_rank() == 0:  # 只在 rank 0 进程中收集数据
        all_predicts = float_tensor.cpu().numpy()
        all_labels= int_tensor.cpu().numpy()
        # print(all_labels)
            

    # 运行 batch_update 更新统计数据
    evaluator.batch_update(predict_labels, true_labels)
    
    # 模拟一个 epoch 结束，调用 epoch_update 来计算 F1 分数
    gpu_f1_score = evaluator.epoch_update()
    if(dist.get_rank() == 0):
        print(f"F1 Score: {gpu_f1_score}")
        # 使用 sklearn 计算 F1 分数
        # all_predicts = np.concatenate(all_predicts)
        # all_labels = np.concatenate(all_labels)
        sklearn_f1 = f1_score(all_labels[:-50], (all_predicts[:-50] > 0.5).astype(int), average='binary')
        print(f"F1 Score (sklearn): {sklearn_f1}", "cnt = ", len(all_labels[:-50]))
    

    # 清理分布式环境
    dist.destroy_process_group()


class PixelMaxF1(AbstractEvaluator):
    """
    Pixel-level Max-F1 (per-image), by sweeping thresholds in [0,1].
    Interface matches PixelF1: batch_update returns tensor [B] (per-image score).
    """
    def __init__(self, num_thresholds: int = 256, mode: str = "origin") -> None:
        super().__init__()
        self.name = "pixel-level Max-F1"
        self.desc = "pixel-level Max-F1"
        self.num_thresholds = int(num_thresholds)
        self.mode = mode  # "origin" | "reverse" | "double"

    @staticmethod
    def _apply_valid(predict, mask, shape_mask):
        # Ensure float
        predict = predict.float()
        mask = mask.float()
        if shape_mask is not None:
            shape_mask = shape_mask.float()
            predict = predict * shape_mask
            mask = mask * shape_mask
        return predict, mask, shape_mask

    @staticmethod
    def _f1_from_counts(TP, FP, FN, eps=1e-8):
        precision = TP / (TP + FP + eps)
        recall = TP / (TP + FN + eps)
        return 2 * precision * recall / (precision + recall + eps)

    @torch.no_grad()
    def _max_f1_origin(self, predict, mask, shape_mask=None):
        """
        predict, mask: [B,1,H,W] float in [0,1] and {0,1}
        returns: max_f1 per-image [B]
        """
        predict, mask, shape_mask = self._apply_valid(predict, mask, shape_mask)

        B = predict.shape[0]
        # Flatten
        p = predict.view(B, -1)
        y = mask.view(B, -1)

        if shape_mask is not None:
            v = shape_mask.view(B, -1) > 0
            # select valid pixels (ragged). We'll do masked sums without indexing:
            # We already multiplied by shape_mask, but to avoid counting invalid as negatives,
            # build a valid float mask:
            vf = (shape_mask.view(B, -1) > 0).float()
        else:
            vf = torch.ones_like(y)

        # thresholds grid
        T = self.num_thresholds
        thresholds = torch.linspace(0.0, 1.0, T, device=p.device).view(1, T, 1)  # [1,T,1]
        scores = p.unsqueeze(1)  # [B,1,N]
        pred_bin = (scores >= thresholds).float()  # [B,T,N]

        # apply valid mask
        pred_bin = pred_bin * vf.unsqueeze(1)
        yv = y * vf  # [B,N]

        TP = (pred_bin * yv.unsqueeze(1)).sum(dim=-1)                       # [B,T]
        FP = (pred_bin * (1.0 - yv).unsqueeze(1)).sum(dim=-1)               # [B,T]
        FN = ((1.0 - pred_bin) * yv.unsqueeze(1)).sum(dim=-1)               # [B,T]

        f1 = self._f1_from_counts(TP, FP, FN)                               # [B,T]
        max_f1, _ = f1.max(dim=1)
        return max_f1

    def batch_update(self, predict, mask, shape_mask=None, *args, **kwargs):
        self._check_pixel_level_params(predict, mask)

        if self.mode == "origin":
            return self._max_f1_origin(predict, mask, shape_mask)

        elif self.mode == "reverse":
            # reverse means use 1 - predict as score
            return self._max_f1_origin(1.0 - predict, mask, shape_mask)

        elif self.mode == "double":
            # take the better direction per image
            f1_o = self._max_f1_origin(predict, mask, shape_mask)
            f1_r = self._max_f1_origin(1.0 - predict, mask, shape_mask)
            return torch.maximum(f1_o, f1_r)

        else:
            raise RuntimeError(f"PixelMaxF1: unknown mode {self.mode}")

    def remain_update(self, predict, mask, shape_mask=None, *args, **kwargs):
        return self.batch_update(predict, mask, shape_mask, *args, **kwargs)

    def epoch_update(self):
        # keep consistent with your PixelF1 (no epoch aggregation here)
        return None

    def recovery(self):
        return None


class PixelFAR(AbstractEvaluator):
    """
    Pixel-level False Alarm Rate (FAR), i.e. FPR = FP / (FP + TN).
    - If threshold is provided, compute FAR@threshold.
    - If sweep=True, compute min FAR over thresholds (or return FAR curve if needed later).
    Interface matches PixelF1: batch_update returns tensor [B] (per-image score).
    """
    def __init__(self, threshold: float = 0.5, mode: str = "origin",
                 sweep: bool = False, num_thresholds: int = 256) -> None:
        super().__init__()
        self.name = "pixel-level FAR"
        self.desc = "pixel-level FAR (FPR)"
        self.threshold = float(threshold)
        self.mode = mode          # "origin" | "reverse" | "double"
        self.sweep = bool(sweep)  # if True, sweep thresholds and take min FAR (best-case)
        self.num_thresholds = int(num_thresholds)

    @staticmethod
    def _apply_valid(predict, mask, shape_mask):
        predict = predict.float()
        mask = mask.float()
        if shape_mask is not None:
            shape_mask = shape_mask.float()
            predict = predict * shape_mask
            mask = mask * shape_mask
        return predict, mask, shape_mask

    @staticmethod
    def _far_from_counts(FP, TN, eps=1e-8):
        return FP / (FP + TN + eps)

    @torch.no_grad()
    def _far_at_threshold(self, predict, mask, shape_mask=None, threshold=0.5):
        """
        FAR per-image at a fixed threshold. predict/mask: [B,1,H,W]
        """
        predict, mask, shape_mask = self._apply_valid(predict, mask, shape_mask)

        pred_bin = (predict >= threshold).float()

        if shape_mask is not None:
            v = (shape_mask > 0).float()
            pred_bin = pred_bin * v
            mask = mask * v
            # valid negatives are those where v==1 and mask==0
            neg = (1.0 - mask) * v
        else:
            neg = (1.0 - mask)

        # FP: predicted 1 on negatives
        FP = (pred_bin * neg).sum(dim=(1, 2, 3))
        # TN: predicted 0 on negatives
        TN = ((1.0 - pred_bin) * neg).sum(dim=(1, 2, 3))

        return self._far_from_counts(FP, TN)

    @torch.no_grad()
    def _far_sweep_min(self, predict, mask, shape_mask=None):
        """
        Sweep thresholds and return min FAR per-image (best-case FAR).
        This is sometimes used as an oracle-style measure (like Max-F1).
        """
        predict, mask, shape_mask = self._apply_valid(predict, mask, shape_mask)

        B = predict.shape[0]
        p = predict.view(B, -1)
        y = mask.view(B, -1)

        if shape_mask is not None:
            vf = (shape_mask.view(B, -1) > 0).float()
        else:
            vf = torch.ones_like(y)

        # negatives among valid pixels
        neg = (1.0 - y) * vf  # [B,N]

        T = self.num_thresholds
        thresholds = torch.linspace(0.0, 1.0, T, device=p.device).view(1, T, 1)
        pred_bin = (p.unsqueeze(1) >= thresholds).float() * vf.unsqueeze(1)  # [B,T,N]

        FP = (pred_bin * neg.unsqueeze(1)).sum(dim=-1)                       # [B,T]
        TN = ((1.0 - pred_bin) * neg.unsqueeze(1)).sum(dim=-1)               # [B,T]

        far = self._far_from_counts(FP, TN)                                  # [B,T]
        min_far, _ = far.min(dim=1)
        return min_far

    def batch_update(self, predict, mask, shape_mask=None, *args, **kwargs):
        self._check_pixel_level_params(predict, mask)

        # choose score direction
        if self.mode == "origin":
            p = predict
        elif self.mode == "reverse":
            p = 1.0 - predict
        elif self.mode == "double":
            # For FAR: "double" is ambiguous.
            # Reasonable choice:
            #   return min(FAR(origin), FAR(reverse))  -> best (lowest false alarms) direction per image
            far_o = self._far_sweep_min(predict, mask, shape_mask) if self.sweep else \
                    self._far_at_threshold(predict, mask, shape_mask, self.threshold)
            far_r = self._far_sweep_min(1.0 - predict, mask, shape_mask) if self.sweep else \
                    self._far_at_threshold(1.0 - predict, mask, shape_mask, self.threshold)
            return torch.minimum(far_o, far_r)
        else:
            raise RuntimeError(f"PixelFAR: unknown mode {self.mode}")

        if self.sweep:
            return self._far_sweep_min(p, mask, shape_mask)
        else:
            return self._far_at_threshold(p, mask, shape_mask, self.threshold)

    def remain_update(self, predict, mask, shape_mask=None, *args, **kwargs):
        return self.batch_update(predict, mask, shape_mask, *args, **kwargs)

    def epoch_update(self):
        return None

    def recovery(self):
        return None


if __name__ == "__main__":
    test_origin_image_f1()