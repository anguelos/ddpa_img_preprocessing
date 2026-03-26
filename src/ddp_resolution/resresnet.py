"""ResNet-based resolution estimator (work in progress)."""

import torch
from torch import Tensor, nn
import torchvision as tv

import torch_mentor as mtr


class ResResNet(mtr.Mentee):
    """ResNet-18 backbone adapted for scan-resolution regression.

    .. note::
        This model is a placeholder.  The ``forward`` method currently
        implements a residual block rather than the full regression head
        and is **not yet functional**.

    Parameters
    ----------
    None

    Attributes
    ----------
    resnet : torchvision.models.ResNet
        Pre-trained ResNet-18 backbone (ImageNet weights).

    Examples
    --------
    >>> model = ResResNet()  # doctest: +SKIP
    """

    def __init__(self) -> None:
        super().__init__()
        self.resnet = tv.models.resnet.resnet18(
            weights=tv.models.ResNet18_Weights.DEFAULT
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through a single residual block.

        .. warning::
            This implementation is incomplete and will raise
            :exc:`AttributeError` at runtime because ``conv1``, ``bn1``,
            etc. are not yet defined on this class.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map of shape ``(N, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Output feature map of the same spatial shape as *x*.
        """
        identity: Tensor = x

        out: Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
