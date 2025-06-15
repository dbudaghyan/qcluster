import inspect
import unittest
from unittest.mock import MagicMock

import torch
# from pacmap import PaCMAP
from umap import UMAP

from qcluster.algorithms.feature_extractors import (  # pacmap_reduction,
    create_embeddings, pca_reduction, umap_reduction)


class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        """Set up for the tests."""
        self.texts = ["This is a test sentence.", "Another test sentence."]
        # Create dummy embeddings with 30 samples and 768 features to
        # avoid issues with PaCMAP on small datasets.
        self.embeddings = torch.randn(30, 768)

    def test_create_embeddings(self):
        """Test the create_embeddings function for both CPU and non-CPU tensors."""
        # Mock the SentenceTransformer model
        mock_model = MagicMock()

        # 1. Test case for embeddings returned on a non-CPU device
        mock_gpu_tensor = MagicMock(spec=torch.Tensor)
        mock_gpu_tensor.is_cpu = False
        cpu_tensor = torch.randn(len(self.texts), 768)
        mock_gpu_tensor.cpu.return_value = cpu_tensor
        mock_model.encode.return_value = mock_gpu_tensor

        result_embeddings = create_embeddings(self.texts, mock_model)

        mock_model.encode.assert_called_with(self.texts, convert_to_tensor=True)
        mock_gpu_tensor.cpu.assert_called_once()
        self.assertIs(result_embeddings, cpu_tensor)

        # 2. Test case for embeddings already on the CPU
        mock_model.reset_mock()
        mock_cpu_tensor = MagicMock(spec=torch.Tensor)
        mock_cpu_tensor.is_cpu = True
        mock_model.encode.return_value = mock_cpu_tensor

        result_embeddings = create_embeddings(self.texts, mock_model)

        mock_cpu_tensor.cpu.assert_not_called()
        self.assertIs(result_embeddings, mock_cpu_tensor)

    def test_pca_reduction(self):
        """Test the pca_reduction function."""
        n_components = 5
        reduced_embeddings = pca_reduction(self.embeddings, n_components=n_components)
        self.assertIsInstance(reduced_embeddings, torch.Tensor)
        self.assertEqual(
            reduced_embeddings.shape, (self.embeddings.shape[0], n_components)
        )
        self.assertFalse(torch.isnan(reduced_embeddings).any())

    def test_umap_reduction(self):
        """Test the umap_reduction function."""
        n_components = 5
        # n_neighbors must be smaller than the number of samples (30)
        reduced_embeddings = umap_reduction(
            self.embeddings, n_components=n_components, n_neighbors=5
        )
        self.assertIsInstance(reduced_embeddings, torch.Tensor)
        self.assertEqual(
            reduced_embeddings.shape, (self.embeddings.shape[0], n_components)
        )
        self.assertFalse(torch.isnan(reduced_embeddings).any())

    # The module has an issue, test fails https://github.com/YingfanWang/PaCMAP/issues/94
    # def test_pacmap_reduction(self):
    #     """Test the pacmap_reduction function."""
    #     n_components = 5
    #     # n_neighbors must be smaller than the number of samples (30)
    #     reduced_embeddings = pacmap_reduction(
    #         self.embeddings, n_components=n_components, n_neighbors=2
    #     )
    #     self.assertIsInstance(reduced_embeddings, torch.Tensor)
    #     self.assertEqual(
    #         reduced_embeddings.shape, (self.embeddings.shape[0], n_components)
    #     )
    #     self.assertFalse(torch.isnan(reduced_embeddings).any())

    def test_umap_reduction_signature_conformity(self):
        """Test if `umap_reduction` signature conforms to umap.UMAP.__init__."""
        local_sig = inspect.signature(umap_reduction)
        upstream_sig = inspect.signature(UMAP.__init__)

        local_params = local_sig.parameters
        upstream_params = upstream_sig.parameters

        local_param_names = set(local_params.keys()) - {"embeddings"}
        upstream_param_names = set(upstream_params.keys()) - {"self"}

        missing_params = local_param_names - upstream_param_names
        self.assertEqual(
            len(missing_params),
            0,
            f"Parameters {missing_params} not found in UMAP.__init__",
        )

        for name in local_param_names:
            if name in upstream_params:
                local_param = local_params[name]
                upstream_param = upstream_params[name]
                if local_param.default != inspect.Parameter.empty:
                    self.assertEqual(
                        local_param.default,
                        upstream_param.default,
                        f"Default value mismatch for '{name}'",
                    )

    # def test_pacmap_reduction_signature_conformity(self):
    #     """Test if `pacmap_reduction` signature conforms to pacmap.PaCMAP.__init__."""
    #     local_sig = inspect.signature(pacmap_reduction)
    #     upstream_sig = inspect.signature(PaCMAP.__init__)
    #
    #     local_params = local_sig.parameters
    #     upstream_params = upstream_sig.parameters
    #
    #     # Handle naming differences (e.g., mn_ratio vs. MN_ratio)
    #     param_mapping = {
    #         "n_components": "n_components",
    #         "n_neighbors": "n_neighbors",
    #         "mn_ratio": "MN_ratio",
    #         "fp_ratio": "FP_ratio",
    #     }
    #
    #     local_param_names = set(local_params.keys()) - {"embeddings"}
    #
    #     for local_name in local_param_names:
    #         self.assertIn(
    #             local_name,
    #             param_mapping,
    #             f"Parameter '{local_name}' has no mapping to PaCMAP.__init__",
    #         )
    #         upstream_name = param_mapping[local_name]
    #         self.assertIn(
    #             upstream_name,
    #             upstream_params,
    #             f"Mapped parameter '{upstream_name}' not in PaCMAP.__init__",
    #         )
    #
    #         local_param = local_params[local_name]
    #         upstream_param = upstream_params[upstream_name]
    #         if local_param.default != inspect.Parameter.empty:
    #             self.assertEqual(
    #                 local_param.default,
    #                 upstream_param.default,
    #                 f"Default value mismatch for '{local_name}' -> '{upstream_name}'",
    #             )


if __name__ == "__main__":
    unittest.main()
