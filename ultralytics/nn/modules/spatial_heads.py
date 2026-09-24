import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialGridHead(nn.Module):
    """
    Grid prediction head: predicts which grid region each feature-map location belongs to.
    This is a self-supervised method that needs no GT bboxes, only the mapping from
    feature-map locations to source-image positions.

    Supports multiple scales: different detection layers can use different grid sizes.
    """

    def __init__(self, in_channels, grid_rows=3, grid_cols=3, hidden_channels=None):
        """
        Args:
            in_channels: number of input feature channels
            grid_rows: number of grid rows
            grid_cols: number of grid columns
            hidden_channels: number of hidden channels
        """
        super().__init__()
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.num_grids = grid_rows * grid_cols

        if hidden_channels is None:
            self.grid_head = nn.Conv2d(in_channels, self.num_grids, 1)
        else:
            self.grid_head = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, self.num_grids, 1),
            )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) feature map
        Returns:
            dict containing:
                - grid_logits: (B, H*W, num_grids) for loss computation
                - grid_map: (B, num_grids, H, W) spatial structure preserved
                - grid_rows: number of grid rows (for loss computation)
                - grid_cols: number of grid columns (for loss computation)
        """
        grid_map = self.grid_head(x)  # (B, num_grids, H, W)

        B, _, H, W = grid_map.shape
        grid_logits = grid_map.reshape(B, self.num_grids, -1).permute(0, 2, 1)  # (B, H*W, num_grids)

        return {
            "grid_logits": grid_logits,
            "grid_map": grid_map,
            "grid_rows": self.grid_rows,
            "grid_cols": self.grid_cols,
        }


class SpatialGridLoss(nn.Module):
    """
    Grid prediction loss: predicts the grid region of each feature-map location.
    Supports:
    1. Self-supervised learning (no GT bboxes required)
    2. Foreground/background weighting (higher weight on foreground regions)
    3. Multi-scale adaptivity (different grid sizes per scale)
    """

    def __init__(self, grid_rows=3, grid_cols=3, fg_weight=2.0, bg_weight=1.0):
        """
        Args:
            grid_rows: number of grid rows
            grid_cols: number of grid columns
            fg_weight: foreground weight (default 2.0)
            bg_weight: background weight (default 1.0)
        """
        super().__init__()
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.num_grids = grid_rows * grid_cols
        self.fg_weight = fg_weight
        self.bg_weight = bg_weight

    def get_pixel_grid_labels(self, feat_h, feat_w, img_h, img_w, device):
        """
        Assign the grid label of each feature-map location.

        Args:
            feat_h, feat_w: feature-map size
            img_h, img_w: source image size
            device: device

        Returns:
            pixel_labels: (feat_h, feat_w) grid label per location
        """
        scale_h = img_h / feat_h
        scale_w = img_w / feat_w

        y_coords = torch.arange(feat_h, dtype=torch.float32, device=device) * scale_h + scale_h / 2
        x_coords = torch.arange(feat_w, dtype=torch.float32, device=device) * scale_w + scale_w / 2

        y_coords = y_coords / img_h
        x_coords = x_coords / img_w

        row_idx = (y_coords * self.grid_rows).long().clamp(0, self.grid_rows - 1)
        col_idx = (x_coords * self.grid_cols).long().clamp(0, self.grid_cols - 1)

        row_idx = row_idx.view(-1, 1).expand(feat_h, feat_w)
        col_idx = col_idx.view(1, -1).expand(feat_h, feat_w)

        pixel_labels = row_idx * self.grid_cols + col_idx
        return pixel_labels

    def create_foreground_mask(self, feat_h, feat_w, gt_bboxes, fg_mask_1d, anchor_points, stride, device):
        """
        Create the foreground mask from GT bboxes and task-assignment results.

        Args:
            feat_h, feat_w: feature-map size
            gt_bboxes: (B, N, 4) GT boxes (xyxy format, normalized)
            fg_mask_1d: (B, H*W) 1D foreground mask (from task assignment)
            anchor_points: (H*W, 2) anchor coordinates
            stride: feature-map stride
            device: device

        Returns:
            fg_mask_2d: (B, H, W) 2D foreground mask
        """
        B = fg_mask_1d.shape[0]

        fg_mask_2d = fg_mask_1d.reshape(B, feat_h, feat_w).float()  # (B, H, W)

        return fg_mask_2d

    def forward(self, pred_dict, img_size, fg_mask=None):
        """
        Args:
            pred_dict: dict containing
                - grid_map: (B, num_grids, H, W)
                - grid_rows, grid_cols: grid size (optional)
            img_size: tuple (img_h, img_w) source image size
            fg_mask: (B, H, W) foreground mask, optional (foreground weighting is used when provided)

        Returns:
            loss: scalar loss
            accuracy: prediction accuracy (for monitoring)
        """
        if "grid_map" in pred_dict:
            grid_map = pred_dict["grid_map"]  # (B, num_grids, H, W)
            B, _, H, W = grid_map.shape

            grid_rows = pred_dict.get("grid_rows", self.grid_rows)
            grid_cols = pred_dict.get("grid_cols", self.grid_cols)
        elif "grid_logits" in pred_dict:
            grid_logits = pred_dict["grid_logits"]  # (B, H*W, num_grids)
            B, HW, num_grids = grid_logits.shape
            H = W = int(HW**0.5)
            grid_map = grid_logits.permute(0, 2, 1).reshape(B, num_grids, H, W)
            grid_rows = pred_dict.get("grid_rows", self.grid_rows)
            grid_cols = pred_dict.get("grid_cols", self.grid_cols)
        else:
            raise ValueError("pred_dict must contain 'grid_map' or 'grid_logits'")

        img_h, img_w = img_size

        if grid_rows != self.grid_rows or grid_cols != self.grid_cols:
            temp_loss_fn = SpatialGridLoss(grid_rows, grid_cols, self.fg_weight, self.bg_weight)
            pixel_labels = temp_loss_fn.get_pixel_grid_labels(H, W, img_h, img_w, grid_map.device)
        else:
            pixel_labels = self.get_pixel_grid_labels(H, W, img_h, img_w, grid_map.device)  # (H, W)

        pixel_labels = pixel_labels.unsqueeze(0).expand(B, -1, -1)  # (B, H, W)

        if fg_mask is not None:
            weight_map = torch.where(fg_mask > 0.5, self.fg_weight, self.bg_weight)  # (B, H, W)

            loss = F.cross_entropy(grid_map, pixel_labels, reduction="none")  # (B, H, W)

            weighted_loss = (loss * weight_map).sum() / weight_map.sum()
            loss = weighted_loss
        else:
            loss = F.cross_entropy(grid_map, pixel_labels)

        with torch.no_grad():
            pred_labels = grid_map.argmax(dim=1)  # (B, H, W)
            correct = (pred_labels == pixel_labels).float()

            if fg_mask is not None:
                fg_correct = (correct * fg_mask).sum() / fg_mask.sum().clamp(min=1)
                bg_correct = (correct * (1 - fg_mask)).sum() / (1 - fg_mask).sum().clamp(min=1)
                accuracy = {
                    "overall": correct.mean(),
                    "foreground": fg_correct,
                    "background": bg_correct,
                }
            else:
                accuracy = {"overall": correct.mean()}

        return loss, accuracy
