import io
import os
import sys
import unittest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import app
from yolo_models import _should_accept_leaf_detection


class YoloEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_yolo_endpoint_accepts_image(self):
        image_bytes = io.BytesIO()
        Image.new('RGB', (224, 224), color='green').save(image_bytes, format='JPEG')
        image_bytes.seek(0)

        response = self.client.post(
            '/predict/yolo',
            data={'image': (image_bytes, 'leaf.jpg')},
            content_type='multipart/form-data',
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn('status', payload)
        self.assertIn('step1_is_rice_leaf', payload)

    def test_strict_leaf_detection_rejects_non_leaf_objects(self):
        self.assertFalse(_should_accept_leaf_detection(
            class_name='finger',
            confidence=0.60,
            box_area_ratio=0.10,
            image_bytes=b'fake-image-bytes'
        ))

        self.assertFalse(_should_accept_leaf_detection(
            class_name='not_rice_leaf',
            confidence=0.95,
            box_area_ratio=0.20,
            image_bytes=b'fake-image-bytes'
        ))

        self.assertTrue(_should_accept_leaf_detection(
            class_name='rice_leaf',
            confidence=0.90,
            box_area_ratio=0.20,
            image_bytes=b'fake-image-bytes'
        ))


if __name__ == '__main__':
    unittest.main()
