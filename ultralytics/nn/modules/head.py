# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Model head modules."""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.tal import dist2bbox, make_anchors
from ultralytics.utils.torch_utils import smart_inference_mode

from .block import DFL, BNContrastiveHead, ContrastiveHead
from .conv import Conv, DWConv

__all__ = "Detect", "OptiSARNetPlusPlusDetect"


class Detect(nn.Module):
    """YOLO Detect head for object detection models.

    This class implements the detection head used in YOLO models for predicting bounding boxes and class probabilities.
    It supports both training and inference modes, with optional end-to-end detection capabilities.

    Attributes:
        dynamic (bool): Force grid reconstruction.
        export (bool): Export mode flag.
        format (str): Export format.
        end2end (bool): End-to-end detection mode.
        max_det (int): Maximum detections per image.
        shape (tuple): Input shape.
        anchors (torch.Tensor): Anchor points.
        strides (torch.Tensor): Feature map strides.
        legacy (bool): Backward compatibility for v3/v5/v8/v9 models.
        xyxy (bool): Output format, xyxy or xywh.
        nc (int): Number of classes.
        nl (int): Number of detection layers.
        reg_max (int): DFL channels.
        no (int): Number of outputs per anchor.
        stride (torch.Tensor): Strides computed during build.
        cv2 (nn.ModuleList): Convolution layers for box regression.
        cv3 (nn.ModuleList): Convolution layers for classification.
        dfl (nn.Module): Distribution Focal Loss layer.
        one2one_cv2 (nn.ModuleList): One-to-one convolution layers for box regression.
        one2one_cv3 (nn.ModuleList): One-to-one convolution layers for classification.

    Methods:
        forward: Perform forward pass and return predictions.
        forward_end2end: Perform forward pass for end-to-end detection.
        bias_init: Initialize detection head biases.
        decode_bboxes: Decode bounding boxes from predictions.
        postprocess: Post-process model predictions.

    Examples:
        Create a detection head for 80 classes
        >>> detect = Detect(nc=80, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = detect(x)
    """

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    legacy = False  # backward compatibility for v3/v5/v8/v9 models
    xyxy = False  # xyxy or xywh output

    def __init__(self, nc: int = 80, reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize the YOLO detection layer with specified number of classes and channels.

        Args:
            nc (int): Number of classes.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = reg_max  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        self.end2end = end2end
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for v5/v5/v8/v9/11 backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3)

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module = None, cls_head: torch.nn.Module = None
    ) -> dict[str, torch.Tensor]:
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if box_head is None or cls_head is None:  # for fused inference
            return dict()
        bs = x[0].shape[0]  # batch size
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        return dict(boxes=boxes, scores=scores, feats=x)

    def forward(
        self, x: list[torch.Tensor]
    ) -> dict[str, torch.Tensor] | torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        preds = self.forward_head(x, **self.one2many)
        if self.end2end:
            x_detach = [xi.detach() for xi in x]
            one2one = self.forward_head(x_detach, **self.one2one)
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps.

        Args:
            x (dict[str, torch.Tensor]): List of feature maps from different detection layers.

        Returns:
            (torch.Tensor): Concatenated tensor of decoded bounding boxes and class probabilities.
        """
        # Inference path
        dbox = self._get_decode_boxes(x)
        return torch.cat((dbox, x["scores"].sigmoid()), 1)

    def _get_decode_boxes(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Get decoded boxes based on anchors and strides."""
        shape = x["feats"][0].shape  # BCHW
        if self.format != "imx" and (self.dynamic or self.shape != shape):
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x["feats"], self.stride, 0.5))
            self.shape = shape

        dbox = self.decode_bboxes(self.dfl(x["boxes"]), self.anchors.unsqueeze(0)) * self.strides
        return dbox

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        for i, (a, b) in enumerate(zip(self.one2many["box_head"], self.one2many["cls_head"])):  # from
            a[-1].bias.data[:] = 2.0  # box
            b[-1].bias.data[: self.nc] = math.log(
                5 / self.nc / (640 / self.stride[i]) ** 2
            )  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for i, (a, b) in enumerate(zip(self.one2one["box_head"], self.one2one["cls_head"])):  # from
                a[-1].bias.data[:] = 2.0  # box
                b[-1].bias.data[: self.nc] = math.log(
                    5 / self.nc / (640 / self.stride[i]) ** 2
                )  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor, xywh: bool = True) -> torch.Tensor:
        """Decode bounding boxes from predictions."""
        return dist2bbox(
            bboxes,
            anchors,
            xywh=xywh and not self.end2end and not self.xyxy,
            dim=1,
        )

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        boxes, scores = preds.split([4, self.nc], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        return torch.cat([boxes, scores, conf], dim=-1)

    def get_topk_index(self, scores: torch.Tensor, max_det: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get top-k indices from scores.

        Args:
            scores (torch.Tensor): Scores tensor with shape (batch_size, num_anchors, num_classes).
            max_det (int): Maximum detections per image.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): Top scores, class indices, and filtered indices.
        """
        batch_size, anchors, nc = scores.shape  # i.e. shape(16,8400,84)
        # Use max_det directly during export for TensorRT compatibility (requires k to be constant),
        # otherwise use min(max_det, anchors) for safety with small inputs during Python inference
        k = max_det if self.export else min(max_det, anchors)
        ori_index = scores.max(dim=-1)[0].topk(k)[1].unsqueeze(-1)
        scores = scores.gather(dim=1, index=ori_index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(k)
        idx = ori_index[torch.arange(batch_size)[..., None], index // nc]  # original index
        return scores[..., None], (index % nc)[..., None].float(), idx

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = None


class SAVPE(nn.Module):
    def __init__(self, ch, c3, embed):
        super().__init__()
        self.cv1 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3)) for x in ch)
        self.cv1[1].append(nn.Upsample(scale_factor=2))
        self.cv1[2].append(nn.Upsample(scale_factor=4))

        self.cv2 = nn.ModuleList(nn.Sequential(Conv(x, c3, 1)) for x in ch)
        self.cv2[1].append(nn.Upsample(scale_factor=2))
        self.cv2[2].append(nn.Upsample(scale_factor=4))

        self.c = 16
        self.cv3 = nn.Conv2d(3 * c3, embed, 1)
        self.cv4 = nn.Conv2d(3 * c3, self.c, 3, padding=1)
        self.cv5 = nn.Conv2d(1, self.c, 3, padding=1)
        self.cv6 = nn.Sequential(Conv(2 * self.c, self.c, 3), nn.Conv2d(self.c, self.c, 3, padding=1))

    def forward(self, x, vp):
        y = [self.cv2[i](xi) for i, xi in enumerate(x)]
        y = self.cv4(torch.cat(y, dim=1))

        x = [self.cv1[i](xi) for i, xi in enumerate(x)]
        x = self.cv3(torch.cat(x, dim=1))

        B, C, H, W = x.shape

        Q = vp.shape[1]

        x = x.view(B, C, -1)

        y = y.reshape(B, 1, self.c, H, W).expand(-1, Q, -1, -1, -1).reshape(B * Q, self.c, H, W)
        vp = vp.reshape(B, Q, 1, H, W).reshape(B * Q, 1, H, W)

        y = self.cv6(torch.cat((y, self.cv5(vp)), dim=1))

        y = y.reshape(B, Q, self.c, -1)
        vp = vp.reshape(B, Q, 1, -1)

        score = y * vp + torch.logical_not(vp) * torch.finfo(y.dtype).min

        score = F.softmax(score, dim=-1, dtype=torch.float).to(score.dtype)

        aggregated = score.transpose(-2, -3) @ x.reshape(B, self.c, C // self.c, -1).transpose(-1, -2)

        return F.normalize(aggregated.transpose(-2, -3).reshape(B, Q, -1), dim=-1, p=2)


class LRPCHead(nn.Module):  # only used during inference
    def __init__(self, vocab, pf, loc, enabled=True):
        super().__init__()
        if enabled:
            self.vocab = self.conv2linear(vocab)
        else:
            self.vocab = vocab
        self.pf = pf
        self.loc = loc
        self.enabled = enabled

    def conv2linear(self, conv):
        assert isinstance(conv, nn.Conv2d) and conv.kernel_size == (1, 1)
        linear = nn.Linear(conv.in_channels, conv.out_channels)
        linear.weight.data = conv.weight.view(conv.out_channels, -1).data
        linear.bias.data = conv.bias.data
        return linear

    def forward(self, cls_feat, loc_feat, conf, max_det):
        if self.enabled:
            pf_score = self.pf(cls_feat)[0, 0].flatten(0)
            mask = pf_score.sigmoid() > conf

            cls_feat = self.vocab(cls_feat.flatten(2).transpose(-1, -2)[:, mask])
            return (self.loc(loc_feat), cls_feat.transpose(-1, -2)), mask
        else:
            cls_feat = self.vocab(cls_feat)
            loc_feat = self.loc(loc_feat)
            return (loc_feat, cls_feat.flatten(2)), torch.ones(
                cls_feat.shape[2] * cls_feat.shape[3], device=cls_feat.device, dtype=torch.bool
            )


class OptiSARNetPlusPlusDetect(Detect):
    is_fused = False

    """Head for integrating YOLO detection models with semantic understanding from text embeddings."""

    def __init__(self, nc=80, embed=512, with_bn=False, num_regions=0, reg_max=16, end2end=False, ch=()):
        """Initialize YOLO detection layer with nc classes and layer channels ch.

        Args:
            nc (int): Number of classes.
            embed (int): Embedding dimension.
            with_bn (bool): Whether to use batch normalization.
            num_regions (int): Number of regions for auxiliary task.
            reg_max (int): Maximum number of DFL channels. Default: 16.
            end2end (bool): Whether to use end-to-end detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=ch)
        c3 = max(ch[0], min(self.nc, 100))
        assert c3 <= embed
        assert with_bn
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, embed, 1),
                )
                for x in ch
            )
        )

        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)
        self.num_regions = num_regions
        if self.num_regions > 0:
            from ultralytics.nn.modules.spatial_heads import SpatialGridHead

            base_grid_size = int(num_regions**0.5)
            assert base_grid_size * base_grid_size == num_regions, (
                f"num_regions must be a perfect square, got {num_regions}"
            )

            default_strides = [8, 16, 32]

            self.cv_region = nn.ModuleList()
            for i in range(len(ch)):
                if self.stride[i].item() > 0:
                    current_stride = self.stride[i].item()
                else:
                    if i < len(default_strides):
                        current_stride = default_strides[i]
                    else:
                        current_stride = default_strides[-1] * (2 ** (i - len(default_strides) + 1))

                stride_scale = current_stride / 8.0
                grid_rows = max(2, int(base_grid_size / (stride_scale**0.5)))
                grid_cols = grid_rows

                self.cv_region.append(
                    SpatialGridHead(embed, grid_rows=grid_rows, grid_cols=grid_cols, hidden_channels=embed // 2)
                )

        self.reprta = Residual(SwiGLUFFN(embed, embed))
        self.savpe = SAVPE(ch, c3, embed)
        self.embed = embed

        # Rebuild the one-to-one branch after replacing the classification head.
        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @smart_inference_mode()
    def fuse(self, txt_feats):
        return

    def get_tpe(self, tpe):
        if tpe is None:
            return None
        return F.normalize(self.reprta(tpe), dim=-1, p=2)

    def get_vpe(self, x, vpe):
        if vpe.ndim == 4:
            vpe = self.savpe(x, vpe)
        assert vpe.ndim == 3
        return vpe

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: nn.ModuleList = None,
        cls_head: nn.ModuleList = None,
        cls_proj: nn.ModuleList = None,
        region_head: nn.ModuleList = None,
    ) -> dict[str, torch.Tensor]:
        """Run either detection branch and return boxes, scores, features, and optional region predictions."""
        if box_head is None or cls_head is None or cls_proj is None:
            return dict()

        cls_pe = x[-1]
        feats = x[: self.nl]

        if self.training:
            assert len(x) == self.nl + 1, (
                f"Expected {self.nl + 1} inputs ({self.nl} features and one class embedding), got {len(x)}"
            )
            assert cls_pe.ndim == 3, (
                f"cls_pe must be 3D (B, num_classes, embed), got shape {tuple(cls_pe.shape)}"
            )
            for i, feat in enumerate(feats):
                assert feat.ndim == 4, f"feats[{i}] must be 4D (B, C, H, W), got shape {tuple(feat.shape)}"

        bs = feats[0].shape[0]  # batch size

        box_outputs = []
        for i in range(self.nl):
            box_out = box_head[i](feats[i]).reshape(bs, 4 * self.reg_max, -1)
            box_outputs.append(box_out)
        boxes = torch.cat(box_outputs, dim=-1)

        cls_feats = [cls_head[i](feats[i]) for i in range(self.nl)]
        score_outputs = []
        for i, cf in enumerate(cls_feats):
            score_out = cls_proj[i](cf, cls_pe)
            score_out = score_out.view(bs, score_out.shape[1], -1)
            score_outputs.append(score_out)
        scores = torch.cat(score_outputs, dim=-1)

        if self.training:
            total_anchors_from_feats = sum([f.shape[2] * f.shape[3] for f in feats])
            total_anchors_from_boxes = boxes.shape[2]
            total_anchors_from_scores = scores.shape[2]
            if (
                total_anchors_from_boxes != total_anchors_from_feats
                or total_anchors_from_scores != total_anchors_from_feats
            ):
                raise RuntimeError(
                    "Inconsistent anchor counts: "
                    f"features={total_anchors_from_feats}, boxes={total_anchors_from_boxes}, "
                    f"scores={total_anchors_from_scores}"
                )

        result = dict(boxes=boxes, scores=scores, feats=feats)

        if region_head is not None and self.num_regions > 0:
            all_grid_logits = []
            all_grid_maps = []

            for i in range(self.nl):
                grid_output = region_head[i](cls_feats[i])
                all_grid_logits.append(grid_output["grid_logits"])
                all_grid_maps.append(grid_output["grid_map"])

            result["regions"] = {
                "grid_logits_list": all_grid_logits,
                "grid_maps": all_grid_maps,
            }

        return result

    def forward(self, x, cls_pe, return_mask=False):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        has_lrpc = hasattr(self, "lrpc")

        # LRPC mode (fused inference)
        if has_lrpc:
            masks = []
            for i in range(self.nl):
                assert self.is_fused
                cls_feat = self.cv3[i](x[i])
                loc_feat = self.cv2[i](x[i])
                assert isinstance(self.lrpc[i], LRPCHead)
                x[i], mask = self.lrpc[i](cls_feat, loc_feat, self.conf, self.max_det)
                masks.append(mask)

            # Inference path for LRPC mode
            shape = x[0][0].shape
            if self.dynamic or self.shape != shape:
                self.anchors, self.strides = (
                    x.transpose(0, 1) for x in make_anchors([b[0] for b in x], self.stride, 0.5)
                )
                self.shape = shape
            box = torch.cat([xi[0].view(shape[0], self.reg_max * 4, -1) for xi in x], 2)
            cls = torch.cat([xi[1] for xi in x], 2)
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
            mask = torch.cat(masks)
            dbox = dbox[:, :, mask]
            y = torch.cat((dbox, cls.sigmoid()), 1)

            if not return_mask:
                return y if self.export else (y, x)
            else:
                return (y, mask) if self.export else ((y, x), mask)

        # Standard mode: use forward_head
        # Append the class prompt embedding to the feature list.
        x_with_pe = x + [cls_pe]

        # One-to-many branch
        one2many = self.forward_head(
            x_with_pe,
            box_head=self.cv2,
            cls_head=self.cv3,
            cls_proj=self.cv4,
            region_head=self.cv_region if self.num_regions > 0 else None,
        )

        # One-to-one branch for optional end-to-end detection.
        preds = one2many
        if self.end2end:
            x_detach = [xi.detach() for xi in x]
            x_detach_with_pe = x_detach + [cls_pe]  # append cls_pe
            one2one = self.forward_head(
                x_detach_with_pe,
                box_head=self.one2one_cv2,
                cls_head=self.one2one_cv3,
                cls_proj=self.one2one_cv4,
                region_head=None,  # the one2one branch does not use region
            )
            preds = {"one2many": one2many, "one2one": one2one}

        if self.training:
            return preds

        # Inference path.
        # Run inference with one2one or one2many predictions
        inference_preds = preds["one2one"] if self.end2end else preds
        y = self._inference(inference_preds)

        # In end-to-end mode, top-k selection is done via postprocess
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)

        return y if self.export else (y, preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Inference path: decode bounding boxes and class probabilities.

        Args:
            x: dict containing boxes, scores, feats

        Returns:
            Decoded prediction tensor.
        """
        dbox = self._get_decode_boxes(x)
        return torch.cat((dbox, x["scores"].sigmoid()), 1)

    def _get_decode_boxes(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode bounding boxes from anchors and strides."""
        shape = x["feats"][0].shape  # BCHW
        if self.format != "imx" and (self.dynamic or self.shape != shape):
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x["feats"], self.stride, 0.5))
            self.shape = shape

        dbox = self.decode_bboxes(self.dfl(x["boxes"]), self.anchors.unsqueeze(0)) * self.strides
        return dbox

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        for a, b, c, s in zip(m.cv2, m.cv3, m.cv4, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            # b[-1].bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
            b[-1].bias.data[:] = 0.0
            c.bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)

        # Initialize the one2one branch
        if self.end2end:
            for a, b, c, s in zip(m.one2one_cv2, m.one2one_cv3, m.one2one_cv4, m.stride):
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[:] = 0.0
                c.bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        """
        Select top-k predictions for end-to-end detection without NMS.

        Args:
            preds (torch.Tensor): raw predictions with shape (batch_size, num_anchors, 4 + nc);
                                  the last dimension is [x, y, w, h, class_probs]
            max_det (int): maximum number of detections per image
            nc (int, optional): number of classes. Defaults to 80

        Returns:
            (torch.Tensor): processed predictions with shape (batch_size, min(max_det, num_anchors), 6);
                           the last dimension is [x, y, w, h, max_class_prob, class_index]
        """
        batch_size, anchors, _ = preds.shape  # e.g. shape(16, 8400, 84)
        boxes, scores = preds.split([4, nc], dim=-1)
        scores, conf, idx = OptiSARNetPlusPlusDetect.get_topk_index(scores, max_det, export=False)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        return torch.cat([boxes, scores, conf], dim=-1)

    @staticmethod
    def get_topk_index(scores: torch.Tensor, max_det: int, export: bool = False) -> tuple:
        """
        Return top-k flattened score indices.

        Args:
            scores (torch.Tensor): score tensor with shape (batch_size, num_anchors, num_classes)
            max_det (int): maximum number of detections
            export (bool): whether this is export mode

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): top scores, class indices, filtered indices
        """
        batch_size, anchors, nc = scores.shape  # e.g. shape(16, 8400, 84)
        # In export mode use max_det directly for TensorRT compatibility; otherwise use min(max_det, anchors)
        k = max_det if export else min(max_det, anchors)
        ori_index = scores.max(dim=-1)[0].topk(k)[1].unsqueeze(-1)
        scores = scores.gather(dim=1, index=ori_index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(k)
        idx = ori_index[torch.arange(batch_size)[..., None], index // nc]  # original indices
        return scores[..., None], (index % nc)[..., None].float(), idx


class SwiGLUFFN(nn.Module):
    def __init__(self, gc, ec, e=4) -> None:
        super().__init__()
        self.w12 = nn.Linear(gc, e * ec)
        self.w3 = nn.Linear(e * ec // 2, ec)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class Residual(nn.Module):
    def __init__(self, m) -> None:
        super().__init__()
        self.m = m
        nn.init.zeros_(self.m.w3.bias)
        # For models with l scale, please change the initialization to
        # nn.init.constant_(self.m.w3.weight, 1e-6)
        nn.init.zeros_(self.m.w3.weight)

    def forward(self, x):
        return x + self.m(x)
