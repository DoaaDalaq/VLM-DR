import os.path as osp
import torch.nn as nn

from .builder import MODELS
from .CLIPDR import load_clip_to_cpu


@MODELS.register_module()
class ImageOnlyRN50(nn.Module):
    """Ablation: CLIP ResNet-50 image encoder + linear 5-grade head, no text."""

    def __init__(self, text_encoder_name="RN50", image_encoder_name="RN50", prompt_learner_cfg=None, **kwargs):
        super().__init__()
        clip_model = load_clip_to_cpu(
            text_encoder_name,
            image_encoder_name,
            root=osp.join(osp.dirname(osp.realpath(__file__)), "..", "..", ".cache", "clip"),
        )
        clip_model.float()
        self.num_ranks = (prompt_learner_cfg or {}).get("num_ranks", 5)
        self.image_encoder = clip_model.visual
        self.classifier = nn.Linear(clip_model.text_projection.shape[1], self.num_ranks)
        # no text branch
        self.prompt_learner = None
        self.text_encoder = None
        self.logit_scale = None

    def forward(self, images):
        f = self.image_encoder(images)
        logits = self.classifier(f)
        # same features passed twice -> CORAL loss becomes 0 (no text side)
        return logits, f, f