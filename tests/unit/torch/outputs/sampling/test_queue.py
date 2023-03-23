import pytest
import torch

from merlin.models.torch.outputs.sampling.queue import FIFOQueue


class TestFIFOQueue:
    def test_enqueue_dequeue_single_element(self):
        queue = FIFOQueue(capacity=5, dims=[3])

        test_tensor = torch.tensor([1, 2, 3])
        queue.enqueue(test_tensor)
        queue.enqueue(test_tensor)
        result = queue.list_all()

        assert torch.equal(result, torch.stack([test_tensor] * 2))

    def test_enqueue_batch(self):
        queue = FIFOQueue(capacity=5, dims=[3])

        test_tensor = torch.rand((4, 3))
        queue.enqueue(test_tensor)
        queue.enqueue(test_tensor.clone())

        result = queue.list_all()
        assert torch.equal(result[1:], test_tensor)
        assert queue.size == 5

    def test_enqueue_dequeue_overflow(self):
        queue = FIFOQueue(capacity=3, dims=[3])

        test_tensor = torch.rand((4, 3))
        queue.enqueue(test_tensor)

        result = queue.list_all()
        assert torch.equal(result, test_tensor[-queue.capacity :])

    def test_list_all_empty_queue(self):
        queue = FIFOQueue(capacity=5, dims=[3])
        result = queue.list_all()

        assert torch.equal(result, torch.tensor([]))

    def test_fifo_queue_enqueue_invalid_shape_1(self):
        fifo = FIFOQueue(capacity=3, dims=[2, 2])
        invalid_tensor = torch.randn(3)  # Invalid shape

        with pytest.raises(RuntimeError, match="Invalid shape for enqueueing"):
            fifo.enqueue(invalid_tensor)

    def test_fifo_queue_enqueue_invalid_shape_2(self):
        fifo = FIFOQueue(capacity=3, dims=[2, 2])
        invalid_tensor = torch.randn(2, 2, 3)  # Invalid shape

        with pytest.raises(RuntimeError, match="Invalid shape for enqueueing"):
            fifo.enqueue(invalid_tensor)
