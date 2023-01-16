import torch
import torch.nn.functional as F
# from instruct_goose.utils import logits_to_logprob


# def test_logits_to_logprobs():
#     logits = torch.tensor([1, 2, 3, 4, 5]).float()

#     logprobs = logits_to_logprob(logits)

#     assert logprobs == F.log_softmax(logits, dim=-1)