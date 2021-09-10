import torch
import torch.nn as nn
import numpy as np

class UtilityGuide(nn.Module):
  def __init__(self,documentation_data,filters,num_filters,embedder,d_model,vocab_size,pad_idx):
    super(UtilityGuide,self).__init__()
    self.utility_idx = documentation_data[0]
    self.utility_description = documentation_data[1]
    self.pad_idx = pad_idx
    self.utility_description_lengths = (self.utility_description[:,:,0] != self.pad_idx).sum(dim=1)
    self.embedder = embedder
    embedding_dim = d_model
    self.vocab_size = vocab_size
    self.WINDOW_SIZES = filters
    self.NUM_FILTERS = num_filters
    self.permutation_ = torch.randperm(len(self.WINDOW_SIZES) * self.NUM_FILTERS)
    self.convNets = nn.ModuleList([nn.Conv2d(in_channels = 1,out_channels = self.NUM_FILTERS, kernel_size = (window_size,embedding_dim), padding = ((window_size-1),0)) for window_size in self.WINDOW_SIZES])
    # Instantiate CNN

  def forward(self,invocations,utility_targets=None,return_distribution=False):
    # invocations: (maxlen X batch X 2)
    # returns : predictions- (batch X value_vocab_size) , loss- (1)
    guidance_distribution=None
    batch_size = invocations.shape[1]
    invocation_embedding = self.embedder(invocations).transpose(1,0).unsqueeze(1) #batch X 1 X maxlen X dim
    final_embedding = []
    invocation_top_idx = []
    for convNet in self.convNets:
      s = convNet(invocation_embedding).squeeze(3)
      s,idx = torch.max(s,dim=2)
      if return_distribution:
        invocation_top_idx.append(idx)
      s = torch.tanh(s)
      final_embedding.append(s)
    final_embedding = torch.cat(final_embedding,dim=1) #batch X dim
    final_embedding = final_embedding[:,self.permutation_]

    description_embedding = self.embedder(self.utility_description.to(invocations.device)).unsqueeze(1) #num_class X 1 X max_len X dim
    final_desc_embedding = []
    description_top_idx = []
    for convNet in self.convNets:
      s = convNet(description_embedding).squeeze(3)
      s,idx = torch.max(s,dim=2)
      if return_distribution:
        description_top_idx.append(idx)
      s = torch.tanh(s)
      final_desc_embedding.append(s)
    final_desc_embedding = torch.cat(final_desc_embedding,dim=1) #num_class X dim
    
    u = final_desc_embedding.unsqueeze(0).repeat(final_embedding.shape[0],1,1) #batch X num_classes X dim
    v = final_embedding.unsqueeze(1).repeat(1,final_desc_embedding.shape[0],1) #batch X num_classes X dim
    
    #Compute Loss
    loss = None
    if utility_targets is not None:
        loss = self.guide_loss(u, v, utility_targets)

    #Compute predictions
    cos = nn.CosineSimilarity(dim=2)
    sparse_predictions = cos(u,v) # (batch , num_classes)

    predictions = torch.zeros(batch_size,self.vocab_size).type_as(u)
    fi = torch.arange(batch_size).unsqueeze(1).repeat(1,self.utility_idx.shape[0]).view(-1)
    si = self.utility_idx.repeat(batch_size)
    predictions.index_put_(indices = [fi,si],values = sparse_predictions.view(-1))

    #Process guidance
    if return_distribution:
      invocation_top_idx = torch.cat(invocation_top_idx,dim=1)
      description_top_idx = torch.cat(description_top_idx,dim=1)
      invocation_lengths = (invocations[:,:,0] != self.pad_idx).sum(dim=0)
      guidance_distribution = self._process_distribution(invocation_lengths,invocation_top_idx,description_top_idx,u_x_v = u*v)
    
    return predictions,loss,guidance_distribution
  
  def guide_loss(self,invocation_embedding,description_embedding,utility_targets):
    # utility_targets: batch X max_util_len (Padded)
    # description_embedding and invocation_embedding : batch X num_classes X dim
    batch = utility_targets.shape[0]
    num_classes = description_embedding.shape[1]

    #create a mask over batch X num_classes with 1 at positions where utiltiy_targets match utility_idx
    uidx = self.utility_idx.unsqueeze(0).repeat(batch,1).unsqueeze(1) # batch X 1 X num_classes
    pos_mask = (uidx == utility_targets.unsqueeze(2)) #batch X max_util_len x num_classes
    pos_mask = pos_mask.any(dim=1) #batch X num_classes
    #use this mask to select similarity values. Thsese are positive embeddings and targets
    pos_u_emb = invocation_embedding[pos_mask] #(batch*max_util_len) X dim
    pos_v_emb = description_embedding[pos_mask] #(batch*max_util_len) X dim

    neg_mask = torch.ones_like(pos_mask).bool()
    neg_mask[pos_mask] = False # clashing positions
    assert not ((neg_mask & pos_mask).any().item())

    neg_u_emb = invocation_embedding[neg_mask]
    neg_v_emb = description_embedding[neg_mask]
    u_emb = torch.cat([pos_u_emb,neg_u_emb],dim=0).view(-1,pos_u_emb.shape[1])
    v_emb = torch.cat([pos_v_emb,neg_v_emb],dim=0).view(-1,pos_v_emb.shape[1])

    pos_targets = torch.ones(pos_u_emb.shape[0]).type_as(utility_targets)
    neg_targets = torch.ones(neg_u_emb.shape[0]).type_as(utility_targets) * -1
    targets = torch.cat([pos_targets,neg_targets])
    loss_fn = nn.CosineEmbeddingLoss(margin=0)
    loss = loss_fn(u_emb,v_emb,targets)
    return loss
  
  def _process_distribution(self,invocation_lengths,invocation_top_idx,description_top_idx,u_x_v):
    attention_matrices = []
    batch = invocation_top_idx.shape[0]
    num_class = description_top_idx.shape[0]
    for b in range(batch):
      example_attention_matrices = []
      for c in range(num_class):
        example_attention_matrices.append(self.generate_alignment_matrix(u_x_v[b,c],invocation_top_idx[b],\
                                  description_top_idx[c],invocation_lengths[b].item(),\
                                  self.utility_description_lengths[c].item()))
      attention_matrices.append(example_attention_matrices)
    return attention_matrices


  def generate_alignment_matrix(self,dot_product,inv_top_idx,desc_top_idx,invocation_len,description_len,topk=30):
    sorted_idx = np.argsort(dot_product.cpu().detach().numpy())
    sorted_idx = sorted_idx[-topk:]
    filter_type = sorted_idx//self.NUM_FILTERS
    iv_filter_idx = inv_top_idx[sorted_idx].cpu().detach().numpy() #(topk,)
    ds_filter_idx = desc_top_idx[sorted_idx].cpu().detach().numpy() #(topk,)
    alignment = np.zeros((invocation_len,description_len))
    for filter_t,iv_idx,ds_idx in zip(filter_type,iv_filter_idx,ds_filter_idx):
      for r_window_step in range(self.WINDOW_SIZES[filter_t]):
        row_idx = iv_idx - r_window_step
        for c_window_step in range(self.WINDOW_SIZES[filter_t]):
          col_idx = ds_idx - c_window_step
          if row_idx >=0 and row_idx < alignment.shape[0] and col_idx>=0 and col_idx < alignment.shape[1]:
            alignment[row_idx,col_idx] += 1

    return alignment