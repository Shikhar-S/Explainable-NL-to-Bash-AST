""" Multi-Head Attention module """
import math
import torch
import torch.nn as nn
from typing_extensions import final
from model.utils import ast_matmul
from main_utils import get_logger
logger = get_logger()
from main_utils import append_to_file
import pickle
log_D = {'pre_net_query':[],'post_net_query':[],'pre_net_key':[],'pre_net_value':[],'post_net_key':[],'post_net_value':[],'post_cache_query':[],'sqrt_query':[],'qk':[],'ast_score':[],'tot_score':[],'masked_score':[],'drop_attn':[],'context':[],'output':[],'parent_embeddings':[],'ast_parents_matrix':[]}
debug = False


def aeq(x,y):
    assert x==y, f'{x} != {y}'

class ParentEmbeddings(nn.Module):
    def __init__(self,vocab_sizes,dim_per_head,padding_idx):
        super(ParentEmbeddings, self).__init__()
        self.structure_embedding_layer = nn.Embedding(vocab_sizes[0],dim_per_head,padding_idx=padding_idx)
        self.value_embedding_layer = nn.Embedding(vocab_sizes[1],dim_per_head,padding_idx=padding_idx)

    def forward(self,matrix):
        structure_matrix = matrix[:,:,::2]
        value_matrix = matrix[:,:,1::2]
        assert structure_matrix.shape[2]==value_matrix.shape[2] or structure_matrix.shape[2] == value_matrix.shape[2]+1
        structure_embedding = self.structure_embedding_layer(structure_matrix)
        value_embedding = self.value_embedding_layer(value_matrix)
        #combine two B X L X (L/2 or [step/2,step/2-1]) X dim tensors at dim=2 interleaved
        structure_embedding = structure_embedding.transpose(2,-1)
        value_embedding = value_embedding.transpose(2,-1)
        undo_length_match = False
        if value_embedding.shape[-1]!=structure_embedding.shape[-1]:
            undo_length_match = True
            value_embedding = torch.cat([value_embedding,torch.zeros_like(structure_embedding[:,:,:,:1])],dim=-1)
        
        assert structure_embedding.shape[0] == value_embedding.shape[0]\
            and structure_embedding.shape[2] == value_embedding.shape[2]\
            and structure_embedding.shape[1] == value_embedding.shape[1]\
            and structure_embedding.shape[-1] == value_embedding.shape[-1]

        final_embedding = torch.stack([structure_embedding,value_embedding],dim=-1)
        final_embedding = final_embedding.view(structure_embedding.shape[0],\
            value_embedding.shape[1],\
            structure_embedding.shape[2],\
            structure_embedding.shape[-1] + value_embedding.shape[-1])
        final_embedding = final_embedding.transpose(2,-1)
        if undo_length_match:
            final_embedding = final_embedding[:,:,:-1,:]
        return final_embedding

class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention module from "Attention is All You Need"

    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.

    Also includes several additional tricks.

    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, head_count, model_dim, dropout=0.1,
                 graph_input=False, vocab_sizes = None, padding_idx = 1):
        try:
            assert model_dim % head_count == 0
        except AssertionError as err:
            logger.error('Model dimension not divisible by head count',err)
            raise err
        super(MultiHeadedAttention, self).__init__()

        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)

        self.graph_input = graph_input
        self.vocab_sizes = vocab_sizes
        self.padding_idx = padding_idx
        if graph_input:
            # self.ast_embeddings = nn.Embedding(max_seq_len + 1, self.dim_per_head, padding_idx = max_seq_len)
            self.ast_embeddings = ParentEmbeddings(self.vocab_sizes,self.dim_per_head,padding_idx = self.padding_idx)

    def forward(self, key, value, query, ast_parents_matrix = None, mask=None,
                layer_cache=None, attn_type=None):
        """
        Compute the context vector and the attention vectors.

        Args:
           key (FloatTensor): set of `key_len`
               key vectors ``(batch, key_len, dim)``
           value (FloatTensor): set of `key_len`
               value vectors ``(batch, key_len, dim)``
           query (FloatTensor): set of `query_len`
               query vectors  ``(batch, query_len, dim)``
           mask: binary mask 1/0 indicating which keys have
               zero / non-zero attention ``(batch, query_len, key_len)``
        Returns:
           (FloatTensor, FloatTensor):

           * output context vectors ``(batch, query_len, dim)``
           * Attention vector in heads ``(batch, head, query_len, key_len)``.
        """
        # # TRAIN CHECKS
        # batch, k_len, d = key.size()
        # batch_, k_len_, d_ = value.size()
        # aeq(batch, batch_)
        # aeq(k_len, k_len_)
        # aeq(d, d_)
        # batch_, q_len, d_ = query.size()
        # aeq(batch, batch_)
        # aeq(d, d_)
        # if mask is not None:
        #     batch_, q_len_, k_len_ = mask.size()
        #     aeq(batch_, batch)
        #     aeq(k_len_, k_len)
        # #    aeq(q_len_, q_len)
        # # END TRAIN CHECKS
        filename = 'ib'
        logging_condition = attn_type == 'self' and ast_parents_matrix is not None and debug

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            """Projection: Break the last dimension into (heads,dim per head) then bring head forward to second dim."""
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)
            # -1 is for sequence length

        def unshape(x):
            """Compute inverse of shape"""
            return x.transpose(1, 2).contiguous() \
                    .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if attn_type == "self":
                if logging_condition:
                    log_D['pre_net_query'].append(query.clone())
                    log_D['pre_net_key'].append(key.clone())
                    log_D['pre_net_value'].append(value.clone())
                # if logging_condition:
                #     append_to_file('Pre forward net', filename)
                #     append_to_file(query[:10,:,40], filename)
                #     append_to_file(key[:10,:,40], filename)
                #     append_to_file(value[:10,:,40], filename)
                query, key, value = self.linear_query(query),\
                                    self.linear_keys(query),\
                                    self.linear_values(query)
                if logging_condition:
                    log_D['post_net_query'].append(query.clone())
                    log_D['post_net_key'].append(key.clone())
                    log_D['post_net_value'].append(value.clone())
                # if logging_condition:
                #     append_to_file('Post forward net', filename)
                #     append_to_file(query[:10,:,40], filename)
                #     append_to_file(key[:10,:,40], filename)
                #     append_to_file(value[:10,:,40], filename)
                key = shape(key)
                value = shape(value)
                if layer_cache["self_keys"] is not None:
                    key = torch.cat(
                        (layer_cache["self_keys"], key),
                        dim=2)
                if layer_cache["self_values"] is not None:
                    value = torch.cat(
                        (layer_cache["self_values"], value),
                        dim=2)
                layer_cache["self_keys"] = key
                layer_cache["self_values"] = value
            elif attn_type == "context":
                query = self.linear_query(query)
                if layer_cache["memory_keys"] is None:
                    key, value = self.linear_keys(key),\
                                 self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
                else:
                    key, value = layer_cache["memory_keys"],\
                               layer_cache["memory_values"]
                layer_cache["memory_keys"] = key
                layer_cache["memory_values"] = value
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query) # batch x seq_len x model_dimension
            key = shape(key)
            value = shape(value) # batch x heads x seq_len x dim_per_head
        
        if self.graph_input and attn_type == "self": 
            key_len = key.size(2)
            parent_embeddings = self.ast_embeddings(ast_parents_matrix[:,:,:key_len])
            if logging_condition:
                log_D['ast_parents_matrix'].append(ast_parents_matrix.clone())
            # batch x key_len x max_keylen->key_len x dim
           
        query = shape(query)
        if logging_condition:
            log_D['post_cache_query'].append(query.clone())
        # if logging_condition:
        #     append_to_file('Query', filename)
        #     append_to_file(query[:10], filename)

        key_len = key.size(2)
        query_len = query.size(2)
        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        # batch x num_heads x query_len x key_len
        if logging_condition:
            log_D['sqrt_query'].append(query.clone())
        # if logging_condition:
        #     append_to_file('Query_sqrt', filename)
        #     append_to_file(query[:10], filename)
        query_key = torch.matmul(query, key.transpose(2, 3)) # batch x heads x qlen x klen
        if logging_condition:
            log_D['qk'].append(query_key.clone())
        # if logging_condition:
        #     append_to_file('Query_Key', filename)
        #     append_to_file(query_key[:10], filename)
        if self.graph_input and attn_type == "self":
            if logging_condition:
                log_D['parent_embeddings'].append(parent_embeddings.clone())
            score_ast = ast_matmul(query, parent_embeddings)
            if logging_condition:
                log_D['ast_score'].append(score_ast.clone())
            # if logging_condition:
            #     append_to_file('AST Score', filename)
            #     append_to_file(score_ast[:10], filename)
            scores = query_key + score_ast
        else:
            scores = query_key

        
        scores = scores.float()
        if logging_condition:
            log_D['tot_score'].append(scores.clone())
        # if logging_condition:
        #     append_to_file('Score', filename)
        #     append_to_file(scores[:10], filename)
        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1, T_values]
            scores = scores.masked_fill(mask, -1e18)
        if logging_condition:
            log_D['masked_score'].append(scores.clone())
        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores).to(query.dtype)
        drop_attn = self.dropout(attn)
        if logging_condition:
            log_D['drop_attn'].append(drop_attn.clone())
        context_original = torch.matmul(drop_attn, value) #batch x heads x len x dim_per_head
        context = unshape(context_original) #batch x len x model_dim
        if logging_condition:
            log_D['context'].append(context.clone())
        output = self.final_linear(context)
        if logging_condition:
            log_D['output'].append(output.clone())
        # # TRAIN CHECK
        # batch_, q_len_, d_ = output.size()
        # aeq(q_len, q_len_)
        # aeq(batch, batch_)
        # aeq(d, d_)

        # Return multi-head attn
        # re-assure that shape matches
        attns = attn \
            .view(batch_size, head_count,
                  query_len, key_len)
        # if logging_condition:
        #     append_to_file(output[:10], filename)
        
        if logging_condition:
            with open('/home/antpc/Desktop/Seq2BashAST/'+filename+'.pkl','wb') as f:
                pickle.dump(log_D,f)

        return output, attns

    def update_dropout(self, dropout):
        self.dropout.p = dropout