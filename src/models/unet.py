import torch
import segmentation_models_pytorch as smp

# Index of the raw R/G/B bands within the 12-channel stack (see
# scripts/build_feature_stack.py:FULL_12_CHANNELS). Used to inflate the
# ImageNet-pretrained first conv layer onto a non-3-channel input.
FULL_12_CHANNELS = [
    "resid15", "resid45", "openness", "profile_curvature",
    "multidirectional_hillshade", "depression_depth",
    "omega", "log_flowacc",
    "red", "green", "blue", "savi_anomaly",
]
RGB_CHANNEL_INDICES = [FULL_12_CHANNELS.index(c) for c in ("red", "green", "blue")]


class BermUNet(smp.Unet):
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=4,
        classes=1,
        rgb_channel_indices=None,
    ):
        # Build with in_channels=3 first so smp actually loads the pretrained
        # first conv, then inflate it ourselves -- passing in_channels=N!=3
        # directly to smp makes it random-init the first conv, discarding
        # the pretrained weights entirely.
        inflate = encoder_weights is not None and in_channels != 3 and rgb_channel_indices is not None
        super().__init__(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3 if inflate else in_channels,
            classes=classes,
            activation=None,
        )
        if inflate:
            _inflate_first_conv(self.encoder, in_channels, rgb_channel_indices)


def _inflate_first_conv(encoder, in_channels: int, rgb_channel_indices: list) -> None:
    """
    Copy the pretrained RGB first-conv weights into the R/G/B channel slots;
    initialize every other channel's kernel as the mean of the pretrained
    RGB kernels divided by the number of non-RGB channels. Random-initing
    all channels would throw away the pretraining that's the whole reason
    for keeping raw RGB in the stack.
    """
    first_conv_name, first_conv = next(
        (name, m) for name, m in encoder.named_modules() if isinstance(m, torch.nn.Conv2d)
    )
    old_weight = first_conv.weight.data  # (out_c, 3, kh, kw)
    out_c, _, kh, kw = old_weight.shape

    n_other = in_channels - len(rgb_channel_indices)
    other_fill = old_weight.mean(dim=1, keepdim=True) / max(n_other, 1)  # (out_c, 1, kh, kw)

    new_weight = other_fill.repeat(1, in_channels, 1, 1).clone()
    for slot, rgb_idx in enumerate(rgb_channel_indices):
        new_weight[:, rgb_idx] = old_weight[:, slot]

    new_conv = torch.nn.Conv2d(
        in_channels, out_c, kernel_size=(kh, kw),
        stride=first_conv.stride, padding=first_conv.padding, bias=first_conv.bias is not None,
    )
    new_conv.weight.data = new_weight
    if first_conv.bias is not None:
        new_conv.bias.data = first_conv.bias.data.clone()

    parent = encoder
    *path, leaf = first_conv_name.split(".")
    for p in path:
        parent = getattr(parent, p)
    setattr(parent, leaf, new_conv)


def build_model(cfg: dict) -> BermUNet:
    rgb_indices = cfg.get("rgb_channel_indices")
    return BermUNet(
        encoder_name=cfg.get("encoder", "resnet34"),
        encoder_weights=cfg.get("encoder_weights", "imagenet"),
        in_channels=cfg.get("in_channels", 4),
        classes=cfg.get("num_classes", 1),
        rgb_channel_indices=rgb_indices,
    )
