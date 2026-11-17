
import unittest

import torch

from src.models.srcnn import SRCNN


class TestSRCNN(unittest.TestCase):

    def setUp(self):
        self.model = SRCNN()
        self.model.eval()

    def test_output_dimensions(self):
        input_tensor = torch.randn(1, 3, 256, 256)

        with torch.no_grad():
            output = self.model(input_tensor)

        self.assertEqual(
            output.shape,
            (1, 3, 244, 244)
        )

    def test_batch_processing(self):
        input_tensor = torch.randn(4, 3, 256, 256)

        with torch.no_grad():
            output = self.model(input_tensor)

        self.assertEqual(
            output.shape,
            (4, 3, 244, 244)
        )

    def test_model_parameter_count(self):
        total_parameters = sum(
            parameter.numel()
            for parameter in self.model.parameters()
        )

        self.assertEqual(total_parameters, 20099)

    def test_invalid_channel_count(self):
        input_tensor = torch.randn(1, 1, 256, 256)

        with self.assertRaises(ValueError):
            self.model(input_tensor)

    def test_invalid_input_dimensions(self):
        input_tensor = torch.randn(3, 256)

        with self.assertRaises(ValueError):
            self.model(input_tensor)

    def test_too_small_input(self):
        input_tensor = torch.randn(1, 3, 12, 12)

        with self.assertRaises(ValueError):
            self.model(input_tensor)


if __name__ == "__main__":
    unittest.main(verbosity=2)
