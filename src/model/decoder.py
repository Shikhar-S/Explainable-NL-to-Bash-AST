import torch
import torch.nn as nn
import pickle

from model.multi_headed_attn import MultiHeadedAttention
from model.position_ffn import PositionwiseFeedForward, ActivationFunction
from model.utils import sequence_mask, generate_ast_parents_matrix
from main_utils import append_to_file
       
log_d = {'inputs':[],'inputs_norm':[],'pre_res_query':[],'post_res_query':[],'query_norm':[],'mid':[],'attns':[],'memory_bank':[],'output':[]}
debug = False

class TransformerDecoderLayer(nn.Module):
    """Transformer Decoder layer block in Pre-Norm style.
    Pre-Norm style is an improvement w.r.t. Original paper's Post-Norm style,
    providing better converge speed and performance. This is also the actual
    implementation in tensor2tensor and also avalable in fairseq.
    """
    def __init__(self,
                d_model,
                heads,
                d_ff,
                dropout,
                attention_dropout,
                graph_input=False,
                vocab_sizes = None,
                padding_idx=1,
                pos_ffn_activation_fn = ActivationFunction.relu):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(
            heads,
            d_model,
            dropout=attention_dropout,
            graph_input=graph_input,
            vocab_sizes = vocab_sizes, 
            padding_idx = padding_idx,
        )
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout,
                                            pos_ffn_activation_fn
                                            )
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)
        self.context_attn = MultiHeadedAttention(
            heads, d_model, dropout=attention_dropout
        )
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)

    def update_dropout(self, dropout, attention_dropout):
        self.context_attn.update_dropout(attention_dropout)
        self.self_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.drop.p = dropout

    def forward(
        self,
        inputs,
        memory_bank,
        src_pad_mask,
        tgt_pad_mask,
        ast_parents_matrix=None,
        layer_cache=None,
        future=False,
    ):
        """A naive forward pass for transformer decoder.

        # T: could be 1 in the case of stepwise decoding or tgt_len

        Args:
            inputs (FloatTensor): ``(batch_size, T, model_dim)``
            memory_bank (FloatTensor): ``(batch_size, src_len, model_dim)``
            src_pad_mask (bool): ``(batch_size, 1, src_len)``
            tgt_pad_mask (bool): ``(batch_size, 1, T)``
            layer_cache (dict or None): cached layer info when stepwise decode
            step (int or None): stepwise decoding counter
            future (bool): If set True, do not apply future_mask.

        Returns:
            (FloatTensor, FloatTensor):

            * output ``(batch_size, T, model_dim)``
            * attns ``(batch_size, head, T, src_len)``
        
        Returns:
            (FloatTensor, FloatTensor, FloatTensor or None):

            * output ``(batch_size, T, model_dim)``
            * top_attn ``(batch_size, T, src_len)``
            * attn_align ``(batch_size, T, src_len)`` or None

        """
        filename = 'b'
        dec_mask = None

        if inputs.size(1) > 1:
            # masking is necessary when sequence length is greater than one
            dec_mask = self._compute_dec_mask(tgt_pad_mask, future)
        # append_to_file('Inputs', filename)
        # append_to_file(inputs[:10], filename)
        if debug:
            log_d['inputs'].append(inputs.clone())

        inputs_norm = self.layer_norm_1(inputs)
        if debug:
            log_d['inputs_norm'].append(inputs_norm.clone())
        # append_to_file('Normed inputs', filename)
        # append_to_file(inputs_norm[:10], filename)
        #Self attention on decoder target input
        query, _ = self.self_attn(
                inputs_norm,
                inputs_norm,
                inputs_norm,
                ast_parents_matrix=ast_parents_matrix,
                mask=dec_mask,
                layer_cache=layer_cache,
                attn_type="self",
            )
        if debug:
            log_d['pre_res_query'].append(query.clone())
        # append_to_file('query', filename)
        # append_to_file(query[:10], filename)
        # residual connection from inputs
        query = self.drop(query) + inputs
        # append_to_file(query[:10], filename)
        if debug:
            log_d['post_res_query'].append(query.clone())
        query_norm = self.layer_norm_2(query)
        #Cross attention on encoder output
        # append_to_file(query_norm[:10], filename)
        if debug:
            log_d['query_norm'].append(query_norm.clone())

        mid, attns = self.context_attn(
            memory_bank,
            memory_bank,
            query_norm,
            mask=src_pad_mask,
            layer_cache=layer_cache,
            attn_type="context",
        )
        # append_to_file(mid[:10], filename)
        # append_to_file(attns[:10], filename)
        if debug:
            log_d['mid'].append(mid.clone())
            log_d['attns'].append(attns.clone())
            log_d['memory_bank'].append(memory_bank.clone())
        # residual connection from query
        output = self.feed_forward(self.drop(mid) + query)
        # append_to_file('Layer Outputs', filename)
        if debug:
            log_d['output'].append(output.clone())
        # append_to_file(output[:10], filename)
        # attns: batch x heads x q_len x k_len
        # output: batch x q_len x model_dim
        top_attn = attns[:, 0, :, :].contiguous()
        #top_attn is attention from first attention head! -> batch x q_len x k_len
        attn_align = attns.mean(dim=1)
        if debug:
            with open('/home/antpc/Desktop/Seq2BashAST/'+filename+'.pkl','wb') as f:
                pickle.dump(log_d,f)    
        return output, top_attn, attn_align

    def _compute_dec_mask(self, tgt_pad_mask, future):
        tgt_len = tgt_pad_mask.size(-1)
        if not future:  # apply future_mask, result mask in (B, T, T)
            future_mask = torch.ones(
                [tgt_len, tgt_len],
                device=tgt_pad_mask.device,
                dtype=torch.uint8,
            )
            future_mask = future_mask.triu_(1).view(1, tgt_len, tgt_len)
            # BoolTensor was introduced in pytorch 1.2
            try:
                future_mask = future_mask.bool()
            except AttributeError:
                pass
            dec_mask = torch.gt(tgt_pad_mask + future_mask, 0)
        else:  # only mask padding, result mask in (B, 1, T)
            dec_mask = tgt_pad_mask
        return dec_mask


class TransformerDecoder(nn.Module):
    """The Transformer decoder from "Attention is All You Need".

    Args:
        num_layers (int): number of decoder layers.
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout in residual, self-attn(dot) and feed-forward
        attention_dropout (float): dropout in context_attn (and self-attn(avg))
        embeddings (onmt.modules.Embeddings):
            embeddings to use, should have positional encodings
        max_relative_positions (int):
            Max distance between inputs in relative positions representations
    """

    def __init__(
        self,
        num_layers,
        d_model,
        heads,
        d_ff,
        dropout,
        attention_dropout,
        embeddings_structure,
        embeddings_value,
        graph_input,
        vocab_sizes,
        padding_idx,
        pos_ffn_activation_fn=ActivationFunction.relu,
    ):
        super(TransformerDecoder, self).__init__()
        assert not graph_input or vocab_sizes is not None
        
        self.embeddings_structure = embeddings_structure
        self.embeddings_value = embeddings_value
        # Decoder State
        self.state = {}
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.vocab_sizes = vocab_sizes
        self.padding_idx = padding_idx
        self.graph_input = graph_input

        self.transformer_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model,
                    heads,
                    d_ff,
                    dropout,
                    attention_dropout,
                    graph_input=graph_input,
                    vocab_sizes = vocab_sizes,
                    padding_idx = padding_idx,
                    pos_ffn_activation_fn=pos_ffn_activation_fn,
                )
                for i in range(num_layers)
            ]
        )
        
    def map_state(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        if self.state["src"] is not None:
            self.state["src"] = fn(self.state["src"], 1)
        if self.state["cache"] is not None:
            _recursive_map(self.state["cache"])

    def detach_state(self):
        self.state["src"] = self.state["src"].detach()
    
    def init_state(self, src):
        """Initialize decoder state.
            src : max_len X src X features
        """
        self.state["src"] = src
        self.state["cache"] = None

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings_structure.update_dropout(dropout)
        self.embeddings_value.update_dropout(dropout)
        for layer in self.transformer_layers:
            layer.update_dropout(dropout, attention_dropout)

    def forward(self, sequence, ast_parents_matrix, tree_coordinates=None, memory_bank=None, step=None, **kwargs):
        """Decode, possibly stepwise.
            sequence :  len X batch X 1
            tgt : len X batch X features
            memory_bank : len X batch X features
            tree_coordinates: batch X len or 1 X batch X 1
            parent_idx: batch X (len or 1)
        """
        # filename  = 'a'
        # append_to_file(sequence[0,:10,0], filename)
        # append_to_file(parent_idx[:10,0], filename)
        # append_to_file(tree_coordinates[0,:10,0], filename)
        # append_to_file(memory_bank[:,:10,:5], filename)
        # append_to_file(step, filename)
        if step == 0:
            self._init_cache()

        if step is None:
            seq_len = sequence.shape[0]
            tgt_structure = sequence[::2,:,:]
            tgt_value = sequence[1::2,:,:]
            if tree_coordinates is not None:
                tgt_structure = torch.cat([tgt_structure,tree_coordinates.transpose(1,0).unsqueeze(-1)],dim=-1)
                tgt_value = torch.cat([tgt_value,tree_coordinates.transpose(1,0).unsqueeze(-1)],dim=-1)
            emb_structure = self.embeddings_structure(tgt_structure, step=step)
            emb_value = self.embeddings_value(tgt_value, step=step)
            emb = torch.stack((emb_structure,emb_value),dim=1)
            emb = emb.view(seq_len, emb.shape[2],emb.shape[3]).contiguous()
        else:
            if tree_coordinates is not None:
                dec_input = torch.cat([sequence,tree_coordinates],dim=-1)
            if step%2==0:
                emb = self.embeddings_structure(dec_input, step=step//2) 
            else:
                emb = self.embeddings_value(dec_input, step=step//2)
            
        #######################################################
        tgt_words = sequence.transpose(0,1).squeeze(-1)
        # batch X len

        assert emb.dim() == 3  # len x batch x embedding_dim

        output = emb.transpose(0, 1).contiguous() # batch x len x embedding_dim
        src_memory_bank = memory_bank.transpose(0, 1).contiguous()
        # batch x len x embedding_dim

        pad_idx = self.embeddings_structure.word_padding_idx
        src_lens = kwargs["memory_lengths"] # (batch,)
        src_max_len = self.state["src"].shape[0]
        src_pad_mask = ~sequence_mask(src_lens, src_max_len).unsqueeze(1) # [B, 1, T_max_src]
        tgt_pad_mask = tgt_words.data.eq(pad_idx).unsqueeze(1)  # [B, 1, T_tgt]

        attn_aligns = []
        # append_to_file(output[:10,0,:5], filename)
        # append_to_file(src_pad_mask[:10], filename)
        # append_to_file(tgt_pad_mask[:10], filename)
        # append_to_file(ast_parents_matrix[:10], filename)
        for i, layer in enumerate(self.transformer_layers):
            layer_cache = (
                self.state["cache"]["layer_{}".format(i)]
                if step is not None
                else None
            )
            # append_to_file('Cache at'+str(i), filename)
            # append_to_file(layer_cache['memory_keys'][:10] if layer_cache['memory_keys'] is not None else 'none', filename)
            # append_to_file(layer_cache['memory_values'][:10] if layer_cache['memory_values'] is not None else 'none', filename)
            # append_to_file(layer_cache['self_keys'][:10] if layer_cache['self_keys'] is not None else 'none', filename)
            # append_to_file(layer_cache['self_values'][:10] if layer_cache['self_values'] is not None else 'none', filename)
            output, attn, attn_align = layer(
                output,
                src_memory_bank,
                src_pad_mask,
                tgt_pad_mask,
                ast_parents_matrix,
                layer_cache = layer_cache,
            )
            # append_to_file('Layer'+str(i), filename)
            # append_to_file(output[:10,0,:5],filename)
            # append_to_file('-'*40, filename)
            if attn_align is not None:
                attn_aligns.append(attn_align)

        output = self.layer_norm(output)
        dec_outs = output.transpose(0, 1).contiguous()
        attn = attn.transpose(0, 1).contiguous()

        attns = {"std": attn}
        return dec_outs, attns

    def _init_cache(self):
        self.state["cache"] = {}

        for i, layer in enumerate(self.transformer_layers):
            layer_cache = {"memory_keys": None, "memory_values": None}
            layer_cache["self_keys"] = None
            layer_cache["self_values"] = None
            self.state["cache"]["layer_{}".format(i)] = layer_cache