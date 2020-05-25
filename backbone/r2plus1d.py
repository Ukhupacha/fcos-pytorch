import torch.nn as nn
import torchvision


class R2PlusOneD(nn.Module):
    """
    R(2+1)D base network model used in our work
    """

    def __init__(self, is_pretrained=True):
        super(R2PlusOneD, self).__init__()
        # We use the pretrained model provided by Torchvision
        self._base_model = torchvision.models.video.r2plus1d_18(
            pretrained=is_pretrained
        )
        # We remove temporal pooling to preserve the number of frames
        # We also dilate the outputs to preserve spatial resolution
        self._temporal_unpool_and_dilate()
        self._prune_pool_fc()

    def _temporal_unpool_and_dilate(self):
        """
        The following were specifically written to remove temporal pooling
        and dilate the output resolution. This causes the output stride to be 4.
        :return: None
        """
        self._base_model.layer2[0].conv1[0][3].stride = (1, 1, 1)
        self._base_model.layer2[0].downsample[0].stride = (1, 2, 2)
        self._base_model.layer3[0].conv1[0][3].stride = (1, 1, 1)
        self._base_model.layer3[0].conv1[0][0].stride = (1, 2, 2)
        self._base_model.layer3[0].downsample[0].stride = (1, 2, 2)
        self._base_model.layer4[0].conv1[0][3].stride = (1, 1, 1)
        self._base_model.layer4[0].conv1[0][0].stride = (1, 2, 2)
        self._base_model.layer4[0].downsample[0].stride = (1, 2, 2)
        return None

    def _prune_pool_fc(self):
        required_layers = list(self._base_model.children())[:-2]
        self._base_model = nn.Sequential(*required_layers)
        return None

    def forward(self, input_batch):
        out = self._base_model(input_batch)
        return out
