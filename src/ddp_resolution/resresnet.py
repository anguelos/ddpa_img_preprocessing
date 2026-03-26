"""ResNet-18 based scan-resolution regression model."""

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
import torchvision as tv

import mentor as mtr


class ResResNet(mtr.Mentee):
    """ResNet-18 backbone for scan-resolution (PPI) regression.

    The standard ResNet-18 classification head is replaced with a single
    linear unit that predicts ``log2(ppi)``.  Using log-space makes the
    loss scale-invariant and guarantees positive PPI at decode time via
    ``2 ** prediction``.

    Parameters
    ----------
    pretrained : bool, optional
        Initialise the backbone with ImageNet weights.  Default is *True*.

    Attributes
    ----------
    resnet : torchvision.models.ResNet
        Backbone with its ``fc`` layer replaced by a scalar regression head.

    Examples
    --------
    >>> model = ResResNet()
    >>> img = torch.zeros(1, 3, 224, 224)
    >>> log2_ppi = model(img)
    >>> log2_ppi.shape
    torch.Size([1, 1])
    >>> ppi = 2 ** log2_ppi.item()
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = tv.models.ResNet18_Weights.DEFAULT if pretrained else None
        self.resnet: tv.models.ResNet = tv.models.resnet18(weights=weights)
        in_features: int = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Identity()

        self.regression_head = torch.nn.Sequential(
            torch.nn.Linear(in_features, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(256, 128),
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(128, 1),
        )

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Predict ``log2(ppi)`` for a batch of images.

        Parameters
        ----------
        x : torch.Tensor
            Normalised image batch of shape ``(N, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Shape ``(N, 1)`` — predicted ``log2(ppi)`` values.

        Examples
        --------
        >>> model = ResResNet()
        >>> out = model(torch.zeros(2, 3, 224, 224))
        >>> out.shape
        torch.Size([2, 1])
        """
        x = self.resnet(x)              # (N, in_features)
        x = self.regression_head(x)    # (N, 1)
        return x

    # ------------------------------------------------------------------
    # Mentee interface
    # ------------------------------------------------------------------

    def training_step(
        self,
        sample: Tuple[Tensor, Any],
        loss_fn: Optional[Any] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute Huber loss between predicted and GT log2-PPI.

        Parameters
        ----------
        sample : tuple of (Tensor, scalar)
            ``(img, ppi)`` as returned by :class:`~ddp_resolution.ResDs`.
            *ppi* may be a float or a 1-D tensor.
        loss_fn : callable, optional
            Ignored; Huber loss is always used.

        Returns
        -------
        tuple of (Tensor, dict)
            ``(loss, {"log2_ppi_mae": ..., "ppi_mae": ...})``

        Raises
        ------
        ValueError
            If *ppi* contains non-positive values.
        """
        img, ppi = sample
        img = img.to(self.device)
        ppi = torch.as_tensor(ppi, dtype=torch.float32).to(self.device).view(-1)

        log2_pred: Tensor = self(img).squeeze(1)           # (N,)
        log2_gt: Tensor = torch.log2(ppi)

        loss: Tensor = F.huber_loss(log2_pred, log2_gt)

        with torch.no_grad():
            log2_mae: float = (log2_pred - log2_gt).abs().mean().item()
            ppi_mae: float = (2.0 ** log2_pred - ppi).abs().mean().item()

        return loss, {"log2_ppi_mae": log2_mae, "ppi_mae": ppi_mae}

    def validation_step(
        self,
        sample: Tuple[Tensor, Any],
        loss_fn: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Evaluate on one validation sample without gradient tracking.

        Parameters
        ----------
        sample : tuple of (Tensor, scalar)
            ``(img, ppi)`` as returned by :class:`~ddp_resolution.ResDs`.
        loss_fn : callable, optional
            Ignored.

        Returns
        -------
        dict
            Same keys as :meth:`training_step`: ``log2_ppi_mae`` and
            ``ppi_mae``.
        """
        _, metrics = self.training_step(sample, loss_fn)
        return metrics

    def preprocess(self, raw_input: Union[str, "PIL.Image.Image"]) -> Tensor:
        """Load and normalise a raw image for inference.

        Parameters
        ----------
        raw_input : str or PIL.Image.Image
            File path or already-opened PIL image.

        Returns
        -------
        torch.Tensor
            Shape ``(1, 3, H, W)`` — ready to pass to :meth:`forward`.
        """
        from PIL import Image
        from ddp_resolution.res_ds import make_inference_transform

        if isinstance(raw_input, str):
            raw_input = Image.open(raw_input).convert("RGB")
        t = make_inference_transform()
        return t(raw_input).unsqueeze(0)

    def decode(self, model_output: Tensor) -> float:
        """Convert raw model output to a PPI value.

        Parameters
        ----------
        model_output : torch.Tensor
            Shape ``(1, 1)`` or ``(1,)`` — output of :meth:`forward`.

        Returns
        -------
        float
            Estimated pixels-per-inch.
        """
        return float(2.0 ** model_output.squeeze())

    # ------------------------------------------------------------------
    # Checkpoint metadata
    # ------------------------------------------------------------------

    def get_output_schema(self) -> Dict[str, Any]:
        """Describe the model output for checkpoint metadata.

        Returns
        -------
        dict
            ``{"type": "regression", "target": "ppi", "space": "log2"}``.
        """
        return {"type": "regression", "target": "ppi", "space": "log2"}

    def get_preprocessing_info(self) -> Dict[str, Any]:
        """Describe the expected preprocessing for checkpoint metadata.

        Returns
        -------
        dict
            ImageNet normalisation statistics and channel order.
        """
        from ddp_resolution.res_ds import IMAGENET_MEAN, IMAGENET_STD

        return {
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
            "channels": "RGB",
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main_res_train() -> None:
    """CLI entry-point for training :class:`ResResNet`.

    Loads a resolution dataset via :class:`~ddp_resolution.ResDs`,
    trains for *epochs* epochs, validates after each epoch, and saves
    the best checkpoint to *out_model*.
    """
    import sys
    import fargv
    from ddp_resolution.res_ds import ResDs, make_train_transform, make_inference_transform

    p = {
        "fsdb_root": "",
        "gt_glob": "**/*.res.gt.json",
        "image_crop": ("img", "Img:WritableArea", "Wr:OldText"),
        "patch_size": 512,
        "val_ratio": 0.2,
        "seed": 42,
        "epochs": 30,
        "lr": 1e-4,
        "step_size": 10,
        "gamma": 0.5,
        "pseudo_batch_size": 3,
        "batch_size": 6,
        "num_workers": 4,
        "pretrained": True,
        "resume": "./tmp/resresnet.pt",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "verbose": False,
    }
    args, _ = fargv.fargv(p)

    import os
    resume_exists = os.path.isfile(args.resume)

    # --- dataset ---------------------------------------------------------
    ds = ResDs.from_root(
        args.fsdb_root,
        args.gt_glob,
        image_crop=args.image_crop,
    )
    train_ds, val_ds = ds.random_split(1.0 - args.val_ratio, seed=args.seed)
    train_ds.input_transform = make_train_transform(patch_size=args.patch_size)
    val_ds.input_transform = make_inference_transform(patch_size=args.patch_size)

    if args.verbose:
        print(f"Train: {len(train_ds)}  Val: {len(val_ds)}", file=sys.stderr)

    # --- model -----------------------------------------------------------
    if resume_exists:
        model, optimizer, lr_scheduler = ResResNet.resume_training(
            args.resume,
            device=args.device,
            lr=args.lr,
            step_size=args.step_size,
            gamma=args.gamma,
        )
        if args.verbose:
            print(f"Resumed from {args.resume} at epoch {model.current_epoch}", file=sys.stderr)
    else:
        model = ResResNet(pretrained=args.pretrained)
        model.to(torch.device(args.device))
        train_objs = model.create_train_objects(
            lr=args.lr,
            step_size=args.step_size,
            gamma=args.gamma,
        )
        optimizer = train_objs["optimizer"]
        lr_scheduler = train_objs["lr_scheduler"]

    # --- training loop ---------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.resume)), exist_ok=True)
    while model.current_epoch < args.epochs:
        train_metrics = model.train_epoch(
            train_ds,
            optimizer,
            lr_scheduler=lr_scheduler,
            pseudo_batch_size=args.pseudo_batch_size,
            num_workers=args.num_workers,
            verbose=args.verbose,
            batch_size = args.batch_size
        )
        val_metrics = model.validate_epoch(
            val_ds,
            num_workers=args.num_workers,
            verbose=args.verbose,
        )
        model.save(args.resume, optimizer=optimizer, lr_scheduler=lr_scheduler)
        if args.verbose:
            print(
                f"Epoch {model.current_epoch}/{args.epochs}  "
                f"train mae={train_metrics['ppi_mae']:.4f}  "
                f"val mae={val_metrics['ppi_mae']:.4f}  "
                f"saved to {args.resume}",
                file=sys.stderr,
            )
