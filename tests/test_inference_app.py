"""Regression checks for mask-only conditioning and web app launches."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/land-test-matplotlib')
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import inference_ldm_app as web
from utils.utils_lidc3D import read_N_masks_for_inference


class InferenceAppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        masks = self.root / 'patient' / 'mask'
        masks.mkdir(parents=True)
        self.mask = np.zeros((8, 8, 8), dtype=np.float32)
        self.mask[1, 1, 1] = 0.5
        self.mask[2, 2, 2] = 1
        np.save(masks / 'sample.npy', self.mask)

    def tearDown(self):
        self.temp.cleanup()

    def test_nodule_encoder_uses_torch_integer_classes(self):
        masks, _ = read_N_masks_for_inference(1, str(self.root), 'nodule', True, 0)
        self.assertEqual(masks.shape, (1, 2, 8, 8, 8))
        self.assertEqual(masks[0, 1, 2, 2, 2], 1)
        self.assertEqual(masks[0, 0, 1, 1, 1], 1)

    def test_lung_encoder_and_saved_mask_without_ct_files(self):
        masks, _ = read_N_masks_for_inference(1, str(self.root), 'nodule+lung', True, 0)
        self.assertEqual(masks[0, 2, 1, 1, 1], 1)
        self.assertEqual(masks[0, 1, 2, 2, 2], 1)
        np.testing.assert_array_equal(web.load_input_mask(str(self.root), 'nodule+lung', 0)[0], self.mask)

    def test_texture_encoder_preserves_lung_class(self):
        with patch('numpy.random.choice', return_value=3):
            masks, textures = read_N_masks_for_inference(1, str(self.root), 'nodule+lung+texture', True, 0)
        self.assertEqual(masks[0, 6, 1, 1, 1], 1)
        self.assertEqual(masks[0, 3, 2, 2, 2], 1)
        self.assertEqual(textures.tolist(), [3])

    def test_texture_without_encoder_preserves_normalized_lung(self):
        with patch('numpy.random.choice', return_value=3):
            masks, _ = read_N_masks_for_inference(1, str(self.root), 'nodule+lung+texture', False, 0)
        self.assertAlmostEqual(float(masks[0, 0, 1, 1, 1]), 0.1)
        self.assertAlmostEqual(float(masks[0, 0, 2, 2, 2]), 0.6)

    def test_mask_batch_cannot_exceed_dataset(self):
        with self.assertRaisesRegex(ValueError, 'dataset has 1 masks'):
            read_N_masks_for_inference(2, str(self.root), 'nodule+lung', False, 0)

    def test_routes_reject_invalid_parameters_and_unknown_jobs(self):
        client = web.app.test_client()
        self.assertEqual(client.get('/').status_code, 200)
        self.assertEqual(client.get('/run_status/unknown').status_code, 404)
        response = client.post('/generate', data={'n_images': 0})
        self.assertEqual(response.status_code, 400)
        self.assertIn(b'n_images must be between', response.data)
        self.assertEqual(client.post('/generate', data={'model_key': 'unknown'}).status_code, 400)

    def test_generation_saves_actual_conditioning_and_partial_batch(self):
        for index in range(1, 3):
            mask = self.mask.copy()
            mask[3, 3, 3] = index / 10.0
            np.save(self.root / 'patient' / 'mask' / f'sample{index}.npy', mask)
        calls = []
        def pipeline(**kwargs):
            calls.append((kwargs['start_indx'], kwargs['batch_size']))
            count = kwargs['batch_size']
            images = np.zeros((count, 1, 8, 8, 8), dtype=np.float32)
            masks, _ = read_N_masks_for_inference(count, str(self.root), 'nodule+lung', False, kwargs['start_indx'])
            return images, masks, None
        pipeline.unet = SimpleNamespace(device='cpu')
        web.generate_images_ldm_web(pipeline, str(self.root / 'run'), 8, 2, 1,
                                    'nodule+lung', str(self.root))
        self.assertEqual(calls, [(0, 2), (2, 1), (0, 2), (2, 1), (0, 2)])
        self.assertEqual(len(list((self.root / 'run' / 'output_images').glob('*.npy'))), 8)
        for index in range(8):
            saved = np.load(self.root / 'run' / 'input_masks' / f'mask_{index:05d}.npy')
            expected, _ = read_N_masks_for_inference(1, str(self.root), 'nodule+lung', False, index % 3)
            np.testing.assert_array_equal(saved, expected[0])

    def test_reused_masks_warning_is_visible_in_status_window(self):
        cfg = dict(web.MODEL_CONFIGS['ldm_nodule_lung'], model_path=str(self.root), mask_dataset=str(self.root))
        with patch.dict(web.MODEL_CONFIGS, {'ldm_nodule_lung': cfg}), \
             patch.object(web, 'WEB_OUTPUTS_DIR', str(self.root / 'outputs')), \
             patch.object(web, 'Thread'):
            run_id = web.GeneratorApp().create_run('ldm_nodule_lung', 12, 2, 1)
        try:
            client = web.app.test_client()
            logs = client.get(f'/run_status/{run_id}').get_json()['logs']
            self.assertIn('WARNING:', logs)
            self.assertIn('reused cyclically', logs)
            self.assertIn('contact the software authors', logs)
            self.assertIn(b'contact the software authors', client.get(f'/run/{run_id}').data)
        finally:
            web.RUNS.pop(run_id)

    def test_empty_mask_dataset_is_rejected(self):
        empty = self.root / 'empty'
        empty.mkdir()
        cfg = dict(web.MODEL_CONFIGS['ldm_nodule_lung'], model_path=str(self.root), mask_dataset=str(empty))
        with patch.dict(web.MODEL_CONFIGS, {'ldm_nodule_lung': cfg}):
            with self.assertRaisesRegex(ValueError, 'No conditioning masks'):
                web.GeneratorApp().create_run('ldm_nodule_lung', 12, 2, 1)

    def test_cli_paths_and_server_options(self):
        previous_choices = list(web.MODEL_CHOICES)
        try:
            with patch.dict(web.MODEL_CONFIGS, {'ldm_nodule_lung': dict(web.MODEL_CONFIGS['ldm_nodule_lung'])}), \
                 patch.object(web, 'HOST'), patch.object(web, 'PORT'), \
                 patch.object(web, 'DEVICE'), patch.object(web, 'WEB_OUTPUTS_DIR'), \
                 patch.object(web, 'LATENTS_DIR'), patch.object(web, 'CREATE_VIDEOS'), \
                 patch.object(web.app, 'run') as serve:
                web.main(['--host', '127.0.0.1', '--model-path', str(self.root), '--mask-mode', 'none',
                          '--outputs-dir', str(self.root / 'outputs'),
                          '--port', '7890', '--device', 'cpu', '--no-videos'])
                self.assertEqual(serve.call_args.kwargs['port'], 7890)
                self.assertEqual(web.MODEL_CONFIGS['ldm_nodule_lung']['mask_mode'], 'none')
                self.assertIsNone(web.MODEL_CONFIGS['ldm_nodule_lung']['mask_dataset'])
                self.assertFalse(web.CREATE_VIDEOS)
        finally:
            web.MODEL_CHOICES[:] = previous_choices


if __name__ == '__main__':
    unittest.main()
