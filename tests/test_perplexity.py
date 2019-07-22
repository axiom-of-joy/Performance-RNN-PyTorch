"""
This script tests re-implementations of cross-entropy loss.

This test helped me develop code for computing the perplexity of the
quantized and unquantized versions of Performance RNN. The test may be
run with pytest -vv --pudb test_perplexity.py. Note: These tests were used for
development purposes, but do not test code appearing in this repository.

Author: Alexander Song
"""

import torch
import torch.nn as nn
import numpy as np
import numpy.testing as nptest
import pytest

@pytest.mark.parametrize("batch_size,num_classes", [
    (4, 6),
    (10, 20),
    (15, 3)
])
def test_cross_entropy_loss_implementation1(batch_size, num_classes):
    """
    Tests a re-implementation of PyTorch's cross-entropy loss.
    """

    ce_loss = nn.CrossEntropyLoss()
    lsm = nn.LogSoftmax(dim=1)
    nnl = nn.NLLLoss(reduction='sum')
    input = torch.randn(batch_size, num_classes)
    target = torch.empty(batch_size, dtype=torch.long).random_(num_classes)
    loss1 = ce_loss(input, target)

    pred = lsm(input)
    loss2 = nnl(pred, target)
    nptest.assert_array_almost_equal(loss1, loss2 / batch_size)


@pytest.mark.parametrize("batch_size,num_classes", [
    (4, 6),
    (10, 20),
    (15, 3)
])
def test_cross_entropy_loss_implementation2(batch_size, num_classes):
    """
    Tests a second re-implementation of cross-entropy loss.
    """

    ce_loss = nn.CrossEntropyLoss()
    lsm = nn.LogSoftmax(dim=1)
    nnl = nn.NLLLoss(reduction='mean')
    input = torch.randn(batch_size, num_classes)
    target = torch.empty(batch_size, dtype=torch.long).random_(num_classes)
    loss1 = ce_loss(input, target)

    pred = lsm(input)
    loss2 = nnl(pred, target)
    nptest.assert_array_almost_equal(loss1, loss2)
