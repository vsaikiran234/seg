import unittest
import numpy as np
from segformer_script import prediction_to_vis

class TestSegformerScript(unittest.TestCase):
    def test_prediction_to_vis_shape(self):
        # Create a dummy prediction mask
        prediction = np.zeros((10, 10), dtype=np.uint8)
        prediction[0:5, 0:5] = 1
        prediction[5:10, 0:5] = 2
        prediction[0:5, 5:10] = 3
        prediction[5:10, 5:10] = 4

        vis_image = prediction_to_vis(prediction)
        self.assertEqual(vis_image.size, (10, 10))
        self.assertEqual(vis_image.mode, "RGB")

    def test_prediction_to_vis_type(self):
        prediction = np.zeros((5, 5), dtype=np.uint8)
        vis_image = prediction_to_vis(prediction)
        from PIL.Image import Image as PILImage
        self.assertIsInstance(vis_image, PILImage)

if __name__ == "__main__":
    unittest.main()