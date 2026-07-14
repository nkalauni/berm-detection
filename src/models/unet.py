import segmentation_models_pytorch as smp


class BermUNet(smp.Unet):
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=4,
        classes=1,
    ):
        super().__init__(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )


def build_model(cfg: dict) -> BermUNet:
    return BermUNet(
        encoder_name=cfg.get("encoder", "resnet34"),
        encoder_weights=cfg.get("encoder_weights", "imagenet"),
        in_channels=cfg.get("in_channels", 4),
        classes=cfg.get("num_classes", 1),
    )
