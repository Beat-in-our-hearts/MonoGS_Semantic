import torch
import numpy as np
from typing import Dict, List, Union

__all__ = ['SegmentationMetric', 'pixel_Accuracy', 'mean_Intersection_over_Union']

class SegmentationMetric(object):
    """Computes pixAcc and mIoU metric scores
    """

    def __init__(self, nclass):
        super(SegmentationMetric, self).__init__()
        self.nclass = nclass
        self.reset()

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_correct = 0
        self.total_label = 0
        self.total_pixAcc = 0
        self.total_area_inter = np.array([0] * self.nclass)
        self.total_area_union = np.array([0] * self.nclass)
        self.gt_label = np.array([0] * self.nclass)
        self.total_miou = 0
        
    def update(self, preds, labels):
        cur_pix_acc, correct, labeled = pixel_Accuracy(preds, labels)
        self.total_correct += correct
        self.total_label += labeled
        
        cur_miou, iou_array, area_inter, area_union, area_lab = mean_Intersection_over_Union(preds, labels, self.nclass)
        self.total_area_inter += area_inter
        self.total_area_union += area_union
        self.gt_label += area_lab
        return cur_pix_acc, cur_miou
        
    def get(self):
        """
        Gets the current evaluation result.
        Returns: tuple of float
            pixAcc and mIoU
        """
        pixAcc = 1.0 * self.total_correct / (1e-10 + self.total_label)
        # self.non_zero = self.total_union > 0
        self.non_zero = self.gt_label > 0
        self.non_zero_total_inter = self.total_area_inter[self.non_zero]
        self.non_zero_total_union = self.total_area_union[self.non_zero]

        IoU = 1.0 * self.non_zero_total_inter / (self.non_zero_total_union + 1e-10)
        mIoU = np.mean(IoU)
        return pixAcc, mIoU

def pixel_Accuracy(predict:Union[torch.Tensor, np.ndarray], target:Union[torch.Tensor, np.ndarray]):
    """PixAcc"""
    assert predict.dtype == target.dtype, "Predict and Target should have the same dtype"
    if isinstance(predict, torch.Tensor):
        assert predict.dim() == 4 and target.dim() == 4, "tensor shape: 1 C H W"
        pixel_labeled = torch.sum(target > 0).item()
        pixel_correct = torch.sum((predict == target) * (target > 0)).item()
    else:
        pixel_labeled = np.sum(target > 0)
        pixel_correct = np.sum((predict == target) * (target > 0))
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    pixel_accuracy = 1.0 * pixel_correct / pixel_labeled
    return pixel_accuracy, pixel_correct, pixel_labeled

def mean_Intersection_over_Union(predict:Union[torch.Tensor, np.ndarray], target:Union[torch.Tensor, np.ndarray], nclass:int):
    """mIoU"""
    assert predict.dtype == target.dtype, "Predict and Target should have the same dtype"
    if isinstance(predict, torch.Tensor):
        assert predict.dim() == 4 and target.dim() == 4, "tensor shape: 1 C H W"
        mini = 1
        maxi = nclass
        nbins = nclass
        predict = predict.float() * (target > 0).float()
        target = target.float()
        intersection = predict * (predict == target).float()

        area_inter = torch.histc(intersection.cpu(), bins=nbins, min=mini, max=maxi)
        area_pred = torch.histc(predict.cpu(), bins=nbins, min=mini, max=maxi)
        area_lab = torch.histc(target.cpu(), bins=nbins, min=mini, max=maxi)
        area_union = area_pred + area_lab - area_inter
        assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"
        iou_array = area_inter / (area_union+1e-10)
        valid_mask = area_lab > 0
        miou = torch.mean(iou_array[valid_mask]).item()
        iou_array = iou_array.cpu().numpy()
        area_inter = area_inter.cpu().numpy()
        area_union = area_union.cpu().numpy()
        area_lab = area_lab.cpu().numpy()
    else:
        intersection = predict * (predict == target)
        (area_inter, _) = np.histogram(intersection, bins=nclass, range=(1, nclass))
        (area_pred, _) = np.histogram(predict, bins=nclass, range=(1, nclass))
        (area_lab, _) = np.histogram(target, bins=nclass, range=(1, nclass))
        area_union = area_pred + area_lab - area_inter
        assert np.sum(area_inter > area_union) == 0, "Intersection area should be smaller than Union area"
        iou_array = area_inter / (area_union+1e-10)
        valid_mask = area_lab > 0
        miou = np.mean(iou_array[valid_mask])
    return miou, iou_array, area_inter, area_union, area_lab
