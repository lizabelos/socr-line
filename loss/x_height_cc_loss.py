from random import randint

import torch

from coder.baseline_decoder import BaselineDecoder
from coder.baseline_encoder import BaselineEncoder
from utils.image import show_numpy_image
from utils.image import show_connected_components, connected_components
from utils.logger import print_normal


class XHeightCCLoss(torch.nn.Module):
    """An absolute position Loss"""

    def __init__(self, loss_type="mse", hysteresis_minimum=0.5, hysteresis_maximum=0.5, thicknesses=2,
                 height_importance=1.0):
        """

        :param s: grid division, assuming we have only 1 bounding box per cell
        """
        super().__init__()

        self.add_activation = None
        self.loss_type = loss_type
        self.mse = torch.nn.MSELoss()

        if loss_type == "mse":
            print_normal("Using MSE Loss with Hysteresis=(" + str(hysteresis_minimum) + "," + str(hysteresis_maximum) + "), thicknesses=" + str(thicknesses) + ", height_importance=" + str(height_importance))
            self.loss = torch.nn.MSELoss()
        elif loss_type == "bce":
            print_normal("Using Binary Cross Entropy Loss Hysteresis=(" + str(hysteresis_minimum) + "," + str(hysteresis_maximum) + "), thicknesses=" + str(thicknesses) + ", height_importance=" + str(height_importance))
            self.loss = torch.nn.BCELoss()
            # self.mse = torch.nn.BCEWithLogitsLoss()
        elif loss_type == "norm":
            self.loss = None
        else:
            raise AssertionError

        self.hysteresis_minimum = hysteresis_minimum
        self.hysteresis_maximum = hysteresis_maximum
        self.thicknesses = thicknesses

        self.height_factor = 1.0
        self.height_importance = height_importance
        self.decoder = BaselineDecoder(self.height_factor, self.hysteresis_minimum, self.hysteresis_maximum)
        self.encoder = BaselineEncoder(self.height_factor, self.thicknesses)

    def forward(self, predicted, y_true):
        predicted = predicted.permute(1, 0, 2, 3).contiguous()
        y_true = y_true.permute(1, 0, 2, 3).contiguous()

        if self.height_importance == 0:
            if self.loss_type == "norm":
                return torch.mean(torch.abs(predicted[0] - y_true[0]))
            else:
                return self.loss(predicted[0], y_true[0])
        else:
            if self.loss_type == "norm":
                return torch.mean(torch.abs(predicted[0] - y_true[0])) + self.height_importance * self.mse(predicted[1], y_true[1])
            else:
                return self.loss(predicted[0], y_true[0]) + self.height_importance * self.mse(predicted[1], y_true[1])

    def document_to_ytrue(self, image_size, base_lines):
        return self.encoder.encode(image_size, base_lines)

    def show_ytrue(self, image, y_true):
        y_true = y_true[0]
        show_numpy_image(image, invert_axes=True)
        show_connected_components(connected_components(y_true))
        show_numpy_image(y_true, invert_axes=False)

    def ytrue_to_lines(self, image, predicted, with_images=True):
        return self.decoder.decode(image, predicted, with_images, degree=3, brut_points=True)
