# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel, OptiSARNetPlusPlusModel
from ultralytics.utils import ROOT, yaml_load


class YOLO(Model):
    """YOLO (You Only Look Once) object detection model."""

    def __init__(self, model="yolo11n.pt", task=None, verbose=False):
        """Initialize YOLO model, switching to OptiSAR-Net++ for matching model files."""
        path = Path(model)
        if "optisar" in path.stem.lower() and path.suffix in {".pt", ".yaml", ".yml"}:
            new_instance = OptiSARNetPlusPlus(path, task=task, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        else:
            # Continue with default YOLO initialization
            super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
        }


class OptiSARNetPlusPlus(Model):
    """OptiSAR-Net++ visual grounding model."""

    def __init__(self, model="configs/optisar-net-plusplus-66m.yaml", task=None, verbose=False) -> None:
        """
        Initialize OptiSAR-Net++ from a checkpoint or model configuration.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """
        super().__init__(model=model, task=task, verbose=verbose)

        # Assign default COCO class names when there are no custom names
        if not hasattr(self.model, "names"):
            self.model.names = yaml_load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": OptiSARNetPlusPlusModel,
                "validator": yolo.optisar_net_plusplus.OptiSARNetPlusPlusDetectValidator,
                "predictor": yolo.detect.DetectionPredictor,
                "trainer": yolo.optisar_net_plusplus.OptiSARNetPlusPlusTrainer,
            },
        }

    def get_text_pe(self, texts):
        assert isinstance(self.model, OptiSARNetPlusPlusModel)
        return self.model.get_text_pe(texts)

    def get_visual_pe(self, img, visual):
        assert isinstance(self.model, OptiSARNetPlusPlusModel)
        return self.model.get_visual_pe(img, visual)

    def set_vocab(self, vocab, names):
        assert isinstance(self.model, OptiSARNetPlusPlusModel)
        self.model.set_vocab(vocab, names=names)

    def get_vocab(self, names):
        assert isinstance(self.model, OptiSARNetPlusPlusModel)
        return self.model.get_vocab(names)

    def set_classes(self, classes, embeddings):
        """
        Set classes.

        Args:
            classes (List(str)): A list of categories i.e. ["person"].
        """
        assert isinstance(self.model, OptiSARNetPlusPlusModel)
        self.model.set_classes(classes, embeddings)
        # Remove background if it's given
        assert " " not in classes
        self.model.names = classes

        # Reset method class names
        # self.predictor = None  # reset predictor otherwise old names remain
        if self.predictor:
            self.predictor.model.names = classes
