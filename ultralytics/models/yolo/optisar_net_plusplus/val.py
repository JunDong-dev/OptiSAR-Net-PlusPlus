import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.tasks import OptiSARNetPlusPlusModel
from ultralytics.utils import LOGGER, TQDM
from ultralytics.utils.torch_utils import select_device, smart_inference_mode


class ImageCache:
    """Image cache manager."""

    def __init__(self, max_size=0):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def preload(self, image_paths):
        """Pre-load all images into memory."""
        LOGGER.info(f"Pre-loading {len(image_paths)} images into memory...")

        for path in TQDM(image_paths, desc="Caching images"):
            if path not in self.cache:
                try:
                    image = Image.open(path).convert("RGB")
                    image.load()
                    self.cache[path] = {"image": image, "size": image.size}
                except Exception as e:
                    LOGGER.warning(f"Failed to load {path}: {e}")
                    self.cache[path] = None

        LOGGER.info(f"Cached {len(self.cache)} images.")

    def get(self, path):
        if path in self.cache:
            self.hits += 1
            self.cache.move_to_end(path)
            cached = self.cache[path]
            if cached is None:
                return None, (0, 0)
            return cached["image"], cached["size"]

        self.misses += 1

        try:
            image = Image.open(path).convert("RGB")
            image.load()
            size = image.size

            self.cache[path] = {"image": image, "size": size}

            if self.max_size > 0 and len(self.cache) > self.max_size:
                self.cache.popitem(last=False)

            return image, size

        except Exception as e:
            LOGGER.warning(f"Error loading {path}: {e}")
            return None, (0, 0)

    def stats(self):
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {"cached_images": len(self.cache), "hits": self.hits, "misses": self.misses, "hit_rate": hit_rate}


class TextEmbeddingCache:
    """Text-embedding cache manager."""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.cache = {}  # caption -> text_pe tensor

    def precompute_embeddings(self, captions, batch_size=64):
        """
        Pre-compute text embeddings for all unique captions.

        Args:
            captions: list of all captions
            batch_size: batch size
        """
        # Collect unique captions
        unique_captions = list(set(captions))
        LOGGER.info(f"Total captions: {len(captions)}")
        LOGGER.info(f"Unique captions: {len(unique_captions)}")
        LOGGER.info("Pre-computing text embeddings...")

        # Call once first to cache the CLIP model
        with torch.no_grad():
            _ = self.model.model.get_text_pe(["dummy"], cache_clip_model=True)

        # Compute text embeddings in batches
        for i in TQDM(range(0, len(unique_captions), batch_size), desc="Computing text embeddings"):
            batch_captions = unique_captions[i : i + batch_size]

            with torch.no_grad():
                # Get the embeddings of this batch
                batch_tpe = self.model.model.get_text_pe(batch_captions, cache_clip_model=True)

                # Store the embedding of each caption
                # batch_tpe shape: [batch_size, embed_dim] or [1, batch_size, embed_dim]
                for j, caption in enumerate(batch_captions):
                    if batch_tpe.dim() == 3:
                        # shape: [1, batch_size, embed_dim] -> take [1, 1, embed_dim]
                        self.cache[caption] = batch_tpe[:, j : j + 1, :].clone()
                    else:
                        # shape: [batch_size, embed_dim] -> take [1, embed_dim] and expand
                        self.cache[caption] = batch_tpe[j : j + 1, :].unsqueeze(0).clone()

        LOGGER.info(f"Cached {len(self.cache)} text embeddings.")

    def get_embedding(self, caption):
        """Get the text embedding of a single caption."""
        if caption in self.cache:
            return self.cache[caption]

        # Compute on the fly if not cached
        with torch.no_grad():
            tpe = self.model.model.get_text_pe([caption], cache_clip_model=True)
        self.cache[caption] = tpe
        return tpe

    def set_class_from_cache(self, caption):
        """Set classes using the cached embedding."""
        tpe = self.get_embedding(caption)
        self.model.model.set_classes([caption], tpe)

    def stats(self):
        return {
            "cached_embeddings": len(self.cache),
        }


class OptiSARNetPlusPlusValidatorMixin:
    def _get_best_box(self, result):
        """Get the highest-confidence box from detection results."""
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return None, 0.0

        confs = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()

        max_idx = np.argmax(confs)
        best_box = self._xyxy_to_xywh(xyxy[max_idx].tolist())
        best_conf = confs[max_idx]

        return best_box, best_conf

    def _xywh_to_xyxy(self, box):
        """Convert [x, y, w, h] to [x1, y1, x2, y2]"""
        return [box[0], box[1], box[0] + box[2], box[1] + box[3]]

    def _xyxy_to_xywh(self, box):
        """Convert box from [x1, y1, x2, y2] to [x, y, w, h]"""
        return [box[0], box[1], box[2] - box[0], box[3] - box[1]]

    def _calculate_iou_metrics(self, box1_xyxy, box2_xyxy):
        """Compute IoU from xyxy boxes (NumPy implementation)."""
        b1_x1, b1_y1, b1_x2, b1_y2 = box1_xyxy
        b2_x1, b2_y1, b2_x2, b2_y2 = box2_xyxy

        inter_x1 = max(b1_x1, b2_x1)
        inter_y1 = max(b1_y1, b2_y1)
        inter_x2 = min(b1_x2, b2_x2)
        inter_y2 = min(b1_y2, b2_y2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

        union_area = b1_area + b2_area - inter_area

        if union_area == 0:
            return 0.0, 0.0, 0.0

        iou = inter_area / union_area
        return iou, inter_area, union_area

    def _evaluate_visual_grounding(
        self, predictions, ground_truths, image_sizes, iou_thresholds=[0.5, 0.6, 0.7, 0.8, 0.9]
    ):
        """Evaluate the visual grounding task."""
        assert len(predictions) == len(ground_truths) == len(image_sizes)

        M = len(predictions)
        precision_at_threshold = {f"Pr@{t}": 0 for t in iou_thresholds}
        total_intersection = 0
        total_union = 0
        sum_iou = 0
        ious = []

        for i in range(M):
            pred_box_xywh = predictions[i]
            gt_box_xywh = ground_truths[i]
            img_w, img_h = image_sizes[i]

            if pred_box_xywh is None:
                # Missed detection: IoU is 0 but still counted in the denominator
                iou = 0.0
                inter_area = 0.0
                gt_xyxy = self._xywh_to_xyxy(gt_box_xywh)
                gt_x1 = np.clip(gt_xyxy[0], 0, img_w)
                gt_y1 = np.clip(gt_xyxy[1], 0, img_h)
                gt_x2 = np.clip(gt_xyxy[2], 0, img_w)
                gt_y2 = np.clip(gt_xyxy[3], 0, img_h)
                union_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
            else:
                pred_xyxy = self._xywh_to_xyxy(pred_box_xywh)
                gt_xyxy = self._xywh_to_xyxy(gt_box_xywh)

                p_x1 = np.clip(pred_xyxy[0], 0, img_w)
                p_y1 = np.clip(pred_xyxy[1], 0, img_h)
                p_x2 = np.clip(pred_xyxy[2], 0, img_w)
                p_y2 = np.clip(pred_xyxy[3], 0, img_h)

                g_x1 = np.clip(gt_xyxy[0], 0, img_w)
                g_y1 = np.clip(gt_xyxy[1], 0, img_h)
                g_x2 = np.clip(gt_xyxy[2], 0, img_w)
                g_y2 = np.clip(gt_xyxy[3], 0, img_h)

                iou, inter_area, union_area = self._calculate_iou_metrics(
                    [p_x1, p_y1, p_x2, p_y2], [g_x1, g_y1, g_x2, g_y2]
                )

            ious.append(iou)
            sum_iou += iou
            total_intersection += inter_area
            total_union += union_area

            for threshold in iou_thresholds:
                if iou >= threshold:
                    precision_at_threshold[f"Pr@{threshold}"] += 1

        results = {}
        for key in precision_at_threshold:
            results[key] = precision_at_threshold[key] / M if M > 0 else 0

        results["meanIoU"] = sum_iou / M if M > 0 else 0
        results["cumIoU"] = total_intersection / total_union if total_union > 0 else 0
        results["total_samples"] = M
        results["avg_iou"] = np.mean(ious) if len(ious) > 0 else 0

        return results, ious

    @smart_inference_mode()
    def evaluate_visual_grounding_dataset(self, model=None, trainer=None, save_dir=None):
        """Evaluate a visual-grounding dataset and save the metrics."""
        images_dir = self.args.grounding_images_dir
        annotations_file = self.args.grounding_annotations
        if not images_dir or not annotations_file:
            raise ValueError("grounding_images_dir and grounding_annotations must be configured for validation")
        LOGGER.info(f"Grounding images: {images_dir}")
        LOGGER.info(f"Grounding annotations: {annotations_file}")
        checkpoint_path = getattr(self.args, "checkpoint", None)

        if not checkpoint_path and trainer is not None:
            ckpt_candidate = getattr(trainer, "last", None)
            if ckpt_candidate:
                checkpoint_path = str(ckpt_candidate)
                self.args.checkpoint = checkpoint_path

        if not checkpoint_path:
            raise ValueError("Grounding eval requires args.checkpoint to be set to a saved weight file.")
        LOGGER.info(f"Grounding checkpoint: {checkpoint_path}")

        runtime_device = getattr(self, "device", None)
        if runtime_device is None:
            runtime_device = str(getattr(self.args, "device", "0")).split(",")[0]
        device_str = str(runtime_device)

        from ultralytics import OptiSARNetPlusPlus

        LOGGER.info(f"Loading OptiSAR-Net++ checkpoint for grounding eval: {checkpoint_path}")
        optisar_wrapper = OptiSARNetPlusPlus(checkpoint_path)
        optisar_wrapper.eval()

        optisar_wrapper.to(runtime_device)

        if not Path(annotations_file).exists():
            raise FileNotFoundError(f"Grounding annotations file not found: {annotations_file}")
        with open(annotations_file, "r") as f:
            data = json.load(f)

        images_dict = {img["id"]: img for img in data["images"]}
        annotations_by_image = {}
        for ann in data["annotations"]:
            image_id = ann["image_id"]
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(ann)

        tasks = []
        unique_image_paths = set()
        all_captions = []

        for image_id, annotations in annotations_by_image.items():
            image_info = images_dict[image_id]
            image_path = str(Path(images_dir) / image_info["file_name"])
            unique_image_paths.add(image_path)

            for ann in annotations:
                caption = ann["caption"]
                all_captions.append(caption)
                tasks.append(
                    {
                        "image_id": image_id,
                        "image_info": image_info,
                        "image_path": image_path,
                        "caption": caption,
                        "gt_box": ann["bbox"],
                        "ann_id": ann["id"],
                    }
                )

        all_predictions = []
        all_ground_truths = []
        all_image_sizes = []

        conf = 0.0001
        iou_threshold = 0.9
        text_batch_size = 64

        LOGGER.info(f"Evaluating visual grounding on {len(tasks)} samples...")
        LOGGER.info(f"Unique images: {len(unique_image_paths)}")

        text_cache = TextEmbeddingCache(optisar_wrapper, device_str)
        text_cache.precompute_embeddings(all_captions, batch_size=text_batch_size)

        image_cache = ImageCache(max_size=0)
        image_cache.preload(list(unique_image_paths))

        LOGGER.info(f"Total evaluation pairs: {len(tasks)}")
        LOGGER.info(f"Unique images: {len(unique_image_paths)}")
        LOGGER.info(f"Unique captions: {len(text_cache.cache)}")
        LOGGER.info("Processing...")

        pbar = TQDM(total=len(tasks), desc="Visual Grounding Evaluation")

        for task in tasks:
            image_path = task["image_path"]
            caption = task["caption"]
            gt_box = task["gt_box"]

            image, (img_w, img_h) = image_cache.get(image_path)

            if image is None:
                all_predictions.append(None)
                all_ground_truths.append(gt_box)
                all_image_sizes.append((0, 0))
                pbar.update(1)
                continue

            text_cache.set_class_from_cache(caption)

            with torch.no_grad():
                results = optisar_wrapper.predict(
                    image, verbose=False, conf=conf, iou=iou_threshold, device=device_str, stream=False
                )

            result = results[0] if isinstance(results, list) else results
            pred_box, pred_conf = self._get_best_box(result)

            all_predictions.append(pred_box)
            all_ground_truths.append(gt_box)
            all_image_sizes.append((img_w, img_h))

            pbar.update(1)

        pbar.close()

        cache_stats = image_cache.stats()
        text_stats = text_cache.stats()

        metrics, ious = self._evaluate_visual_grounding(all_predictions, all_ground_truths, all_image_sizes)

        LOGGER.info("Visual grounding evaluation results:")
        LOGGER.info(f"Total samples: {metrics['total_samples']}")
        LOGGER.info(f"meanIoU: {metrics['meanIoU']:.4f}")
        LOGGER.info(f"cumIoU:  {metrics['cumIoU']:.4f}")
        for k, v in metrics.items():
            if k.startswith("Pr@"):
                LOGGER.info(f"{k}: {v:.4f}")

        save_dir = Path(save_dir or getattr(self, "save_dir", "."))
        save_dir.mkdir(parents=True, exist_ok=True)

        history_file = save_dir / "vg_metrics.json"
        history = []
        if history_file.exists():
            try:
                with open(history_file, "r") as f:
                    existing_data = json.load(f)
                    if isinstance(existing_data, list):
                        history = existing_data
                    else:
                        history = [existing_data]
            except (json.JSONDecodeError, Exception) as e:
                LOGGER.warning(f"Failed to read existing vg_metrics.json: {e}, starting fresh.")
                history = []

        import datetime

        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        current_result = {
            "timestamp": timestamp_str,
            "checkpoint": checkpoint_path if checkpoint_path else "unknown",
            "dataset": str(annotations_file),
            "batch_size": 1,
            "conf": conf,
            "iou_threshold": iou_threshold,
            "text_batch_size": text_batch_size,
            "cache_stats": cache_stats,
            "text_cache_stats": text_stats,
            "metrics": metrics,
        }
        history.append(current_result)

        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)

        ious_file = save_dir / f"vg_ious_{timestamp_str}.npy"
        np.save(ious_file, np.array(ious, dtype=np.float32))

        return metrics

    @smart_inference_mode()
    def get_visual_pe(self, model):
        assert isinstance(model, OptiSARNetPlusPlusModel)
        data_loader, names = self.get_lvis_train_vps_loader(model)
        visual_pe = torch.zeros(len(names), model.model[-1].embed, device=self.device)
        cls_visual_num = torch.zeros(len(names))

        desc = "Get visual prompt embeddings from samples"

        for batch in data_loader:
            cls = batch["cls"].squeeze(-1).to(torch.int).unique()
            count = torch.bincount(cls, minlength=len(names))
            cls_visual_num += count

        cls_visual_num = cls_visual_num.to(self.device)

        pbar = TQDM(data_loader, total=len(data_loader), desc=desc)
        for batch in pbar:
            batch = self.preprocess(batch)
            preds = model.get_visual_pe(batch["img"], visual=batch["visuals"])
            assert preds.shape[0] == 1

            cls = batch["cls"].squeeze(-1).to(torch.int).unique(sorted=True)
            assert len(cls) == 1
            visual_pe[cls] += preds[0][cls] / cls_visual_num[cls]

        visual_pe[cls_visual_num != 0] = F.normalize(visual_pe[cls_visual_num != 0], dim=-1, p=2)
        visual_pe[cls_visual_num == 0] = 0
        return visual_pe.unsqueeze(0)

    def preprocess(self, batch):
        batch = super().preprocess(batch)
        if "visuals" in batch:
            batch["visuals"] = batch["visuals"].to(batch["img"].device)
        return batch

    def get_lvis_train_vps_loader(self, model):
        lvis_train_vps_data = check_det_dataset("lvis_train_vps.yaml")
        lvis_train_vps_loader = build_dataloader(
            build_yolo_dataset(
                self.args,
                lvis_train_vps_data.get("val"),
                1,
                lvis_train_vps_data,
                mode="val",
                stride=max(int(model.stride.max()), 32),
                rect=False,
                load_vp=True,
            ),
            1,
            self.args.workers,
            shuffle=False,
            rank=-1,
        )
        return lvis_train_vps_loader, lvis_train_vps_data["names"]

    def add_prefix_for_metric(self, stats, prefix):
        prefix_stats = {}
        for k, v in stats.items():
            if k.startswith("metrics"):
                prefix_stats[f"{prefix}_{k}"] = v
            else:
                prefix_stats[k] = v
        return prefix_stats

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """Run visual-grounding validation during training or standalone evaluation."""
        LOGGER.info("OptiSAR-Net++ Visual Grounding Evaluation Mode")

        if trainer is not None:
            self.device = trainer.device

            if hasattr(trainer, "save_model"):
                trainer.save_model()
            ckpt_candidate = getattr(trainer, "last", None)
            if ckpt_candidate:
                self.args.checkpoint = str(ckpt_candidate)
                LOGGER.info(f"Using checkpoint: {self.args.checkpoint}")
        else:
            self.device = select_device(self.args.device, self.args.batch)

        if model is not None:
            if isinstance(model, (str, Path)):
                self.args.checkpoint = str(model)
                LOGGER.info(f"Model path provided: {self.args.checkpoint}")
            else:
                if hasattr(model, "eval"):
                    model.eval().to(self.device)

        vg_metrics = self.evaluate_visual_grounding_dataset(trainer=trainer, model=model, save_dir=self.save_dir)

        stats = {}

        if vg_metrics is not None:
            vg_pref = {f"vg_{k}": v for k, v in vg_metrics.items()}
            stats.update(vg_pref)

            if "meanIoU" in vg_metrics:
                stats["fitness"] = vg_metrics["meanIoU"]
                LOGGER.info(f"Fitness set to vg_meanIoU: {vg_metrics['meanIoU']:.6f}")
            else:
                stats["fitness"] = 0.0
                LOGGER.warning("vg_metrics is missing meanIoU; fitness is set to 0.0")
        else:
            stats["fitness"] = 0.0
            LOGGER.warning("vg_metrics is None; fitness is set to 0.0")

        stats["vg_evaluation_only"] = True

        LOGGER.info("Visual grounding validation summary:")
        LOGGER.info(f"Fitness (meanIoU): {stats['fitness']:.6f}")

        if vg_metrics:
            LOGGER.info(f"cumIoU:           {vg_metrics.get('cumIoU', 0):.6f}")
            LOGGER.info(f"Pr@0.5:          {vg_metrics.get('Pr@0.5', 0):.6f}")
            LOGGER.info(f"Pr@0.7:          {vg_metrics.get('Pr@0.7', 0):.6f}")
            LOGGER.info(f"Total samples:   {vg_metrics.get('total_samples', 0)}")

        return stats


class OptiSARNetPlusPlusDetectValidator(OptiSARNetPlusPlusValidatorMixin, DetectionValidator):
    pass
