""" Full assembly of the parts to form the complete network 

NOTE: this code is mainly taken from: https://github.com/milesial/Pytorch-UNet
"""
import os
import numpy as np
from skimage import feature
import cv2
from util.constants import SAMPLE_SHAPE

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class PytorchUnetCellModel():
    """
    U-NET model for cell detection implemented with the Pytorch library

    NOTE: this model does not utilize the tissue patch but rather
    only the cell patch.

    Parameters
    ----------
    metadata: Dict
        Dataset metadata in case you wish to compute statistics

    """
    def __init__(self, metadata):
        self.device = torch.device('cuda:0')
        self.metadata = metadata
        self.resize_to = (512, 512) # The model is trained with 512 resolution
        # RGB images and 2 class prediction
        self.n_classes =  3 # Two cell classes and background

        self.unet = UNet(n_channels=3, n_classes=self.n_classes)
        self.load_checkpoint()
        self.unet = self.unet.to(self.device)
        self.unet.eval()

    def load_checkpoint(self):
        """Loading the trained weights to be used for validation"""
        _curr_path = os.path.split(__file__)[0]
        _path_to_checkpoint = os.path.join(_curr_path, "checkpoints/ocelot_unet.pth")
        state_dict = torch.load(_path_to_checkpoint, map_location=torch.device('cpu'))
        self.unet.load_state_dict(state_dict, strict=True)
        print("Weights were successfully loaded!")

    def prepare_input(self, cell_patch):
        """This function prepares the cell patch array to be forwarded by
        the model

        Parameters
        ----------
        cell_patch: np.ndarray[uint8]
            Cell patch with shape [1024, 1024, 3] with values from 0 - 255

        Returns
        -------
            torch.tensor of shape [1, 3, 1024, 1024] where the first axis is the batch
            dimension
        """
        cell_patch = torch.from_numpy(cell_patch).permute((2, 0, 1)).unsqueeze(0)
        cell_patch = cell_patch.to(self.device).type(torch.cuda.FloatTensor)
        cell_patch = cell_patch / 255 # normalize [0-1]
        if self.resize_to is not None:
            cell_patch= F.interpolate(
                    cell_patch, size=self.resize_to, mode="bilinear", align_corners=True
            ).detach()
        return cell_patch
        
    def find_cells(self, heatmap):
        """This function detects the cells in the output heatmap

        Parameters
        ----------
        heatmap: torch.tensor
            output heatmap of the model,  shape: [1, 3, 1024, 1024]

        Returns
        -------
            List[tuple]: for each predicted cell we provide the tuple (x, y, cls, score)
        """
        arr = heatmap[0,:,:,:].cpu().detach().numpy()
        # arr = np.transpose(arr, (1, 2, 0)) # CHW -> HWC

        bg, pred_wo_bg = np.split(arr, (1,), axis=0) # Background and non-background channels
        bg = np.squeeze(bg, axis=0)
        obj = 1.0 - bg

        arr = cv2.GaussianBlur(obj, (0, 0), sigmaX=3)
        peaks = feature.peak_local_max(
            arr, min_distance=3, exclude_border=0, threshold_abs=0.0
        ) # List[y, x]

        maxval = np.max(pred_wo_bg, axis=0)
        maxcls_0 = np.argmax(pred_wo_bg, axis=0)

        # Filter out peaks if background score dominates
        peaks = np.array([peak for peak in peaks if bg[peak[0], peak[1]] < maxval[peak[0], peak[1]]])
        if len(peaks) == 0:
            return []

        # Get score and class of the peaks
        scores = maxval[peaks[:, 0], peaks[:, 1]]
        peak_class = maxcls_0[peaks[:, 0], peaks[:, 1]]

        predicted_cells = [(x, y, c + 1, float(s)) for [y, x], c, s in zip(peaks, peak_class, scores)]

        return predicted_cells

    def post_process(self, logits):
        """This function applies some post processing to the
        output logits
        
        Parameters
        ----------
        logits: torch.tensor
            Outputs of U-Net

        Returns
        -------
            torch.tensor after post processing the logits
        """
        if self.resize_to is not None:
            logits = F.interpolate(logits, size=SAMPLE_SHAPE[:2],
                mode='bilinear', align_corners=False
            )
        return torch.softmax(logits, dim=1)

    def __call__(self, cell_patch, tissue_patch, pair_id):
        """This function detects the cells in the cell patch using Pytorch U-Net.

        Parameters
        ----------
        cell_patch: np.ndarray[uint8]
            Cell patch with shape [1024, 1024, 3] with values from 0 - 255
        tissue_patch: np.ndarray[uint8] 
            Tissue patch with shape [1024, 1024, 3] with values from 0 - 255
        pair_id: str
            Identification number of the patch pair

        Returns
        -------
            List[tuple]: for each predicted cell we provide the tuple (x, y, cls, score)
        """
        cell_patch = self.prepare_input(cell_patch)
        logits = self.unet(cell_patch)
        heatmap = self.post_process(logits)
        return self.find_cells(heatmap)
