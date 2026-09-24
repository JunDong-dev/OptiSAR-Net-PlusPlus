# Ultralytics YOLO 🚀, AGPL-3.0 license


from copy import copy

from ultralytics.data import build_yolo_dataset
from ultralytics.models import yolo
from ultralytics.nn.tasks import OptiSARNetPlusPlusModel
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.torch_utils import de_parallel

from .val import OptiSARNetPlusPlusDetectValidator


class OptiSARNetPlusPlusTrainer(yolo.detect.DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return OptiSARNetPlusPlusModel initialized with specified config and weights."""
        # NOTE: This `nc` here is the max number of different text samples in one image, rather than the actual `nc`.
        # NOTE: Following the official config, nc hard-coded to 80 for now.
        model = OptiSARNetPlusPlusModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=3,
            nc=min(self.data["nc"], 80),
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        # Check whether the model uses MoE
        has_moe = self._check_model_has_moe()
        # Append "region" to loss_names when the region auxiliary head is present
        base_model = de_parallel(self.model)
        has_region = getattr(base_model.model[-1], "num_regions", 0) > 0
        if has_moe and has_region:
            self.loss_names = "box", "cls", "dfl", "moe", "region"  # order: box, cls, dfl, moe, region
        elif has_moe:
            self.loss_names = "box", "cls", "dfl", "moe"
        elif has_region:
            self.loss_names = "box", "cls", "dfl", "region"
        else:
            self.loss_names = "box", "cls", "dfl"

        return OptiSARNetPlusPlusDetectValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def _check_model_has_moe(self):
        """Check whether the model contains a MoE-based C3k2 module."""
        from ultralytics.nn.modules import PLoRA_MoE
        from ultralytics.utils.torch_utils import de_parallel

        model = de_parallel(self.model)

        for module in model.modules():
            if isinstance(module, PLoRA_MoE):
                return True
        return False

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(
            self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs, multi_modal=mode == "train"
        )

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        batch["txt_feats"] = batch["texts"].to(self.device)
        return batch
