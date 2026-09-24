# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.data import YOLOConcatDataset, build_grounding, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.optisar_net_plusplus import OptiSARNetPlusPlusTrainer
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.torch_utils import de_parallel


class OptiSARNetPlusPlusTrainerFromScratch(OptiSARNetPlusPlusTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build a YOLO dataset.

        Args:
            img_path (List[str] | str): folder path(s) containing images.
            mode (str): `train` or `val`; users can customize different augmentations per mode.
            batch (int, optional): batch size, used for `rect` mode. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        if mode != "train":
            # In val mode, use build_grounding if img_path is a dict (grounding data)
            if isinstance(img_path, dict):
                return build_grounding(
                    self.args, img_path["img_path"], img_path["json_file"], batch, stride=gs, mode=mode, data=self.data
                )
            else:
                # Otherwise (yolo data as a string), use build_yolo_dataset
                return build_yolo_dataset(
                    self.args, img_path, batch, self.data, mode=mode, rect=False, stride=gs, load_vp=False
                )
        dataset = [
            build_yolo_dataset(self.args, im_path, batch, self.training_data[im_path], stride=gs, multi_modal=True)
            if isinstance(im_path, str)
            else build_grounding(self.args, im_path["img_path"], im_path["json_file"], batch, stride=gs)
            for im_path in img_path
        ]
        return YOLOConcatDataset(dataset) if len(dataset) > 1 else dataset[0]

    def get_dataset(self):
        """
        Extract train/val paths from the data dict (if present).

        Returns None if the data format is not recognized.
        """
        final_data = {}
        data_yaml = self.args.data
        assert data_yaml.get("train", False), "training dataset not found"  # object365.yaml
        assert data_yaml.get("val", False), "validation dataset not found"  # lvis.yaml
        data = {k: [check_det_dataset(d) for d in v.get("yolo_data", [])] for k, v in data_yaml.items()}
        for s in ["train", "val"]:
            final_data[s] = [d["train" if s == "train" else "val"] for d in data[s]]
            # Keep the grounding data (if any)
            grounding_data = data_yaml[s].get("grounding_data")
            if grounding_data is None:
                continue
            grounding_data = grounding_data if isinstance(grounding_data, list) else [grounding_data]
            for g in grounding_data:
                assert isinstance(g, dict), f"Grounding data must be provided as a dict, got {type(g)}"
            final_data[s] += grounding_data
        # Note: `nc` and `names` must be set for training to work
        final_data["nc"] = data["val"][0]["nc"]
        final_data["names"] = data["val"][0]["names"]
        # Note: add lvis path
        final_data["path"] = data["val"][0]["path"]
        self.data = final_data
        self.training_data = {}
        for d in data["train"]:
            self.training_data[d["train"]] = d
        return final_data["train"], final_data["val"][0]

    def plot_training_labels(self):
        """Do not plot labels."""
        pass

    def final_eval(self):
        if not self.args.val:
            return
        val = self.args.data["val"]["yolo_data"][0]
        self.validator.args.data = val
        self.validator.args.split = "minival" if isinstance(val, str) and "lvis" in val else "val"
        return super().final_eval()
