import torch

from python.utils.torch_utils import compare_expected_actual


def test_compare_expected_actual():
    shape = [32, 32, 32, 32]
    expected = -torch.ones(shape)
    actual = torch.zeros(shape)
    assert (compare_expected_actual(expected, actual) - 1) < 1e-6

