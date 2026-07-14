from typing import Optional, List

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from segmentation_models_pytorch.losses._functional import soft_dice_score, to_tensor, focal_loss_with_logits
from segmentation_models_pytorch.losses.constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE
from functools import partial

__all__ = ["WeightedLoss"]


class WeightedLoss(_Loss):
    def __init__(
        self,
        mode: str,
        dice_str: float,
        bce_str: float,
        focal_str: float,
        classes: Optional[List[int]] = None,
        ignore_index: Optional[int] = None,
        from_logits: bool = True,
        dice_log_loss: bool = False,
        dice_smooth: float = 0.0,
        dice_eps: float = 1e-7,
        bce_weight: Optional[torch.Tensor] = None,
        bce_reduction: str = "mean",
        bce_smooth_factor: Optional[float] = None,
        bce_pos_weight: Optional[torch.Tensor] = None,
        focal_alpha: Optional[float] = None,
        focal_gamma: Optional[float] = 2.0,
        focal_reduction: Optional[str] = "mean",
        focal_normalized: bool = False,
        focal_reduced_threshold: Optional[float] = None,
    ):
        """Dice loss for image segmentation task.
        It supports binary, multiclass and multilabel cases

        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            classes:  List of classes that contribute in loss computation. By default, all channels are included.
            log_loss: If True, loss computed as `- log(dice_coeff)`, otherwise `1 - dice_coeff`
            from_logits: If True, assumes input is raw logits
            smooth: Smoothness constant for dice coefficient (a)
            ignore_index: Label that indicates ignored pixels (does not contribute to loss)
            eps: A small epsilon for numerical stability to avoid zero division error
                (denominator will be always greater or equal to eps)

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        assert dice_str >= 0.0 and dice_str <= 1.0
        assert bce_str >= 0.0 and bce_str <= 1.0
        assert focal_str >= 0.0 and focal_str <= 1.0
        assert dice_str + bce_str + focal_str == 1.0
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super(WeightedLoss, self).__init__()
        self.mode = mode
        if classes is not None:
            assert mode != BINARY_MODE, (
                "Masking classes is not supported with mode=binary"
            )
            classes = to_tensor(classes, dtype=torch.long)

        self.dice_str = dice_str
        self.bce_str = bce_str
        self.focal_str = focal_str
        self.classes = classes
        self.ignore_index = ignore_index
        self.from_logits = from_logits
        self.dice_smooth = dice_smooth
        self.dice_eps = dice_eps
        self.dice_log_loss = dice_log_loss
        self.bce_reduction = bce_reduction
        self.bce_smooth_factor = bce_smooth_factor
        self.register_buffer("bce_weight", bce_weight)
        self.register_buffer("bce_pos_weight", bce_pos_weight)
        self.focal_loss_fn = partial(
            focal_loss_with_logits,
            alpha=focal_alpha,
            gamma=focal_gamma,
            reduced_threshold=focal_reduced_threshold,
            reduction=focal_reduction,
            normalized=focal_normalized,
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert y_true.size(0) == y_pred.size(0)

        if self.dice_str > 0:
            dice_loss = self.dice_forward(y_pred, y_true) * self.dice_str
        else:
            dice_loss = 0
        if self.bce_str > 0:
            bce_loss = self.bce_forward(y_pred, y_true) * self.bce_str
        else:
            bce_loss = 0
        if self.focal_str > 0:
            focal_loss = self.focal_forward(y_pred, y_true) * self.focal_str
        else:
            focal_loss = 0
        
        loss = dice_loss + bce_loss + focal_loss

        return loss

    def aggregate_dice_loss(self, loss):
        return loss.mean()

    def compute_score(
        self, output, target, smooth=0.0, eps=1e-7, dims=None
    ) -> torch.Tensor:
        return soft_dice_score(output, target, smooth, eps, dims)
    
    def dice_forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            if self.mode == MULTICLASS_MODE:
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        if self.mode == BINARY_MODE:
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        if self.mode == MULTICLASS_MODE:
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask.unsqueeze(1)

                y_true = F.one_hot(
                    (y_true * mask).to(torch.long), num_classes
                )  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1)  # N, C, H*W
            else:
                y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1)  # N, C, H*W

        if self.mode == MULTILABEL_MODE:
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        scores = self.compute_score(
            y_pred, y_true.type_as(y_pred), smooth=self.dice_smooth, eps=self.dice_eps, dims=dims
        )

        if self.dice_log_loss:
            loss = -torch.log(scores.clamp_min(self.dice_eps))
        else:
            loss = 1.0 - scores

        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        if self.classes is not None:
            loss = loss[self.classes]

        return self.aggregate_dice_loss(loss)

    def bce_forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: torch.Tensor of shape (N, C, H, W)
            y_true: torch.Tensor of shape (N, H, W)  or (N, 1, H, W)

        Returns:
            loss: torch.Tensor
        """

        if self.smooth_factor is not None:
            soft_targets = (1 - y_true) * self.smooth_factor + y_true * (
                1 - self.smooth_factor
            )
        else:
            soft_targets = y_true

        loss = F.binary_cross_entropy_with_logits(
            y_pred,
            soft_targets,
            self.bce_weight,
            pos_weight=self.bce_pos_weight,
            reduction="none",
        )

        if self.ignore_index is not None:
            not_ignored_mask = y_true != self.ignore_index
            loss *= not_ignored_mask.type_as(loss)

        if self.bce_reduction == "mean":
            loss = loss.mean()

        if self.bce_reduction == "sum":
            loss = loss.sum()

        return loss
    
    def focal_forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.mode in {BINARY_MODE, MULTILABEL_MODE}:
            y_true = y_true.view(-1)
            y_pred = y_pred.view(-1)

            if self.ignore_index is not None:
                # Filter predictions with ignore label from loss computation
                not_ignored = y_true != self.ignore_index
                y_pred = y_pred[not_ignored]
                y_true = y_true[not_ignored]

            loss = self.focal_loss_fn(y_pred, y_true)

        elif self.mode == MULTICLASS_MODE:
            num_classes = y_pred.size(1)
            loss = 0

            # Filter anchors with -1 label from loss computation
            if self.ignore_index is not None:
                not_ignored = y_true != self.ignore_index

            for cls in range(num_classes):
                cls_y_true = (y_true == cls).long()
                cls_y_pred = y_pred[:, cls, ...]

                if self.ignore_index is not None:
                    cls_y_true = cls_y_true[not_ignored]
                    cls_y_pred = cls_y_pred[not_ignored]

                loss += self.focal_loss_fn(cls_y_pred, cls_y_true)

        return loss