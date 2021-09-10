# -*- coding: utf-8 -*-

# import torch
import random
import inspect
import numpy as np
from itertools import islice, repeat
import os
import torch
import torch.nn as nn
from torch.nn.functional import pad
from main_utils import append_to_file

def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    mask = (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))
    return mask


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0) 
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def generate_ast_parents_matrix(parent_idx, sequence, padding_idx, stepwise_decoding = False):
    """Generate parent type and value ptr matrix shape: batch X length X length"""
    length = sequence.shape[0]
    batch = parent_idx.shape[0]
    if not stepwise_decoding:
        assert parent_idx.shape[1] * 2 == length # batch X (length/2)
        mtrx = torch.stack([parent_idx*2, parent_idx*2, parent_idx*2, parent_idx*2 ],dim=2) # batch x (length/2) X 4
        mtrx = mtrx.view(batch,length,-1) # batch x length x 2
        mtrx[:,:,1] += 1
        filled_mtrx = torch.full(size = (batch,length,length),fill_value = padding_idx,dtype=torch.int).type_as(mtrx)
        idx1 = torch.stack([torch.arange(batch)]*length,dim=1).view(batch*length)
        idx2 = torch.stack([torch.arange(length)]*batch,dim=0).view(batch*length)
        filled_mtrx = filled_mtrx.index_put(indices = [idx1,idx2,mtrx[:,:,0].view(batch*length)], values = mtrx[:,:,0].view(batch*length))
        filled_mtrx = filled_mtrx.index_put(indices = [idx1,idx2,mtrx[:,:,1].view(batch*length)], values = mtrx[:,:,1].view(batch*length))
        unmasked_sequence = sequence.permute(1,2,0)
        filled_mtrx = unmasked_sequence.masked_fill(filled_mtrx==padding_idx,padding_idx) #caveat: dummy for root has index 1 which is also padding index. Ignored
    else:
        filled_mtrx = torch.full(size = (batch,1,length+1),fill_value = padding_idx,dtype=torch.int).type_as(parent_idx)
        idx1 = torch.stack([torch.arange(batch)]*1,dim=1).view(batch)
        idx2 = torch.stack([torch.arange(1)]*batch,dim=0).view(batch)
        filled_mtrx = filled_mtrx.index_put(indices = [idx1,idx2,parent_idx], values = parent_idx)
        filled_mtrx = filled_mtrx.index_put(indices = [idx1,idx2,parent_idx+1], values = parent_idx+1)
        filled_mtrx = filled_mtrx[:,:,:length]
        unmasked_sequence = sequence.unsqueeze(2).permute(1,2,0)
        filled_mtrx = unmasked_sequence.masked_fill(filled_mtrx == padding_idx, padding_idx)

    filled_mtrx = filled_mtrx.type_as(parent_idx)
    return filled_mtrx


def ast_matmul(x, z):
    """Helper function for graph attention.
    x: batch x head x (qlen or 1) x dim
    z: batch x (1 or qlen) x (step or klen) x dim
    Returns:
    score: batch x head x qlen x (klen or step)
    """
    # filename = 'a'
    # print('Query shape',x.shape)
    # print('Parent embedding shape',z.shape)
    assert x.shape[0] == z.shape[0] and x.shape[-1] == z.shape[-1] and z.shape[1] == x.shape[2]
    # append_to_file('X', filename)
    # append_to_file(x[:10], filename)
    # append_to_file('Z', filename)
    # append_to_file(z[:10], filename)
    x = x.unsqueeze(-1)
    z = z.unsqueeze(1)
    score = torch.matmul(z,x)
    score = score.squeeze(-1)
    # append_to_file('Score', filename)
    # append_to_file(score[:10], filename)
    return score

""" Misc classes """

# At the moment this class is only used by embeddings.Embeddings look-up tables
class Elementwise(nn.ModuleList):
    """
    A simple network container.
    Parameters are a list of modules.
    Inputs are a 3d Tensor whose last dimension is the same length
    as the list.
    Outputs are the result of applying modules to inputs elementwise.
    An optional merge parameter allows the outputs to be reduced to a
    single Tensor.
    """

    def __init__(self, merge=None, *args):
        assert merge in [None, 'first', 'concat', 'sum', 'mlp']
        self.merge = merge
        super(Elementwise, self).__init__(*args)

    def forward(self, inputs):
        #create list of tensors by splitting last dimension
        inputs_ = [feat.squeeze(2) for feat in inputs.split(1, dim=2)]
        #apply each module to corresponding tensor splitted above
        assert len(self) == len(inputs_)
        outputs = [f(x) for f, x in zip(self, inputs_)]
        if self.merge == 'first':
            return outputs[0]
        elif self.merge == 'concat' or self.merge == 'mlp':
            return torch.cat(outputs, 2)
        elif self.merge == 'sum':
            return sum(outputs)
        else:
            return outputs


class Cast(nn.Module):
    """
    Basic layer that casts its input to a specific data type. The same tensor
    is returned if the data type is already correct.
    """

    def __init__(self, dtype):
        super(Cast, self).__init__()
        self._dtype = dtype

    def forward(self, x):
        return x.to(self._dtype)
