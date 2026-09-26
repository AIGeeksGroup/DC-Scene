"""Unified dense captioning interface for DC-Scene's two paper backbones."""
from torch import nn


class CaptionNet(nn.Module):
    def __init__(self, args, dataset_config, train_dataset):
        super().__init__()
        self.backbone = args.backbone
        from dc_scene.quality import fingerprint
        self.signature = {key: getattr(args, key) for key in (
            "backbone", "dataset", "use_color", "use_normal", "use_height", "use_multiview", "max_des_len")}
        self.signature["vocabulary_sha256"] = fingerprint(train_dataset.tokenizer.word2idx)
        if args.backbone == "3dcoca":
            from models.coca3d.CoCa3d import detector
            self.detector = detector(args, dataset_config, train_dataset)
            self.captioner = None
        elif args.backbone == "vote2cap_detrpp":
            from models.detector_Vote2Cap_DETRv2.detector import detector
            from models.captioner_dccv2.captioner import captioner
            self.detector = detector(args, dataset_config)
            self.captioner = captioner(args, train_dataset)
        else:
            raise ValueError(args.backbone)

    def forward(self, inputs, is_eval=False):
        outputs = self.detector(inputs, is_eval=is_eval)
        if self.captioner is not None:
            outputs = self.captioner(outputs, inputs, is_eval=is_eval)
        return outputs
