from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def make_train_transform(patch_size=None, jitter_strength=0.3):
    """Return a training transform pipeline.

    Args:
        patch_size: If given, randomly crop to this square size (pad if needed).
        jitter_strength: ColorJitter magnitude for brightness and contrast.
    """
    ops = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=jitter_strength, contrast=jitter_strength),
    ]
    if patch_size is not None:
        ops.append(transforms.RandomCrop(patch_size, pad_if_needed=True))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(ops)


def make_inference_transform(patch_size=None):
    """Return an inference transform pipeline.

    Args:
        patch_size: If given, centre-crop to this square size before inference.
    """
    ops = []
    if patch_size is not None:
        ops.append(transforms.CenterCrop(patch_size))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(ops)
