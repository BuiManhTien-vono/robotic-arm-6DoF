import unittest
import gc
import operator as op
import functools
import torch
from torch.autograd import Variable, Function
try:
  from knn_pytorch import knn_pytorch
except ImportError:
  knn_pytorch = None
# import knn_pytorch
def knn(ref, query, k=1):
  """ Compute k nearest neighbors for each query point.
  """
  device = ref.device
  ref = ref.float().to(device)
  query = query.float().to(device)
  if knn_pytorch is None:
    ref_points = ref.transpose(1, 2).contiguous()
    query_points = query.transpose(1, 2).contiguous()
    _, inds = torch.topk(torch.cdist(query_points, ref_points), k=k, dim=-1, largest=False)
    return inds.permute(0, 2, 1).contiguous() + 1
  inds = torch.empty(query.shape[0], k, query.shape[2]).long().to(device)
  knn_pytorch.knn(ref, query, inds)
  return inds
