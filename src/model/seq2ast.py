from numpy import mat
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from argparse import ArgumentParser
from main_utils import str2bool, get_logger
from decoder.bash_generator import get_score
from model.utils import generate_ast_parents_matrix

logger = get_logger()

class Seq2BashAST(pl.LightningModule):
    def __init__(self, invocation_encoder, utility_guide, decoder, d_model, trg_tokenizer, alpha, beta, device):
        super(Seq2BashAST, self).__init__()
        self.translator = None
        self.invocation_encoder = invocation_encoder
        self.utility_guide = utility_guide
        self.decoder = decoder
        self.d_model = d_model
        self.pad = trg_tokenizer.pad
        self.trg_tokenizer = trg_tokenizer
        self.n_structure = self.trg_tokenizer.kind_id.vocab_len
        self.inference = False
        self.max_len = 128
        self.root = self.trg_tokenizer.get_kind_id('root')
        self.utility_id = self.trg_tokenizer.get_kind_id('utility')
        self.unk = self.trg_tokenizer.kind_id.unknown_id
        self.alpha = alpha
        self.beta = beta
        self.structure_gen = nn.Linear(d_model,trg_tokenizer.kind_id.vocab_len)
        self.value_gen = nn.Linear(d_model,trg_tokenizer.value_id.vocab_len)
        self.soften_params = nn.parameter.Parameter(torch.ones(self.trg_tokenizer.value_id.vocab_len).to(device))
    
    def init_translator(self,translator):
        self.translator = translator

    def forward(self, invocation, invocation_length, ast_traversal, ast_traversal_length, utility_target = None):
        '''
        invocation: max_len X batch X 2
        invocation_len: batch
        ast_traversal: batch X max_len X 3
        ast_traversal_length: batch
        '''
        guidance,guidance_loss,_ = self.utility_guide(invocation,utility_target)
        _,inv_encodings,lengths = self.invocation_encoder(invocation,invocation_length)
        # inv_encodings: max_len x batch X dim
        self.decoder.init_state(invocation)
        #################################INPUT FOR DECODER################################
        parent_idx = ast_traversal[:,:,2] # batch X length/2
        level = ast_traversal[:,:,3]
        seq_structure = ast_traversal[:,:,0].transpose(1,0).unsqueeze(2)
        seq_value = ast_traversal[:,:,1].transpose(1,0).unsqueeze(2)
        max_len = ast_traversal.shape[1]
        sequence = torch.stack((seq_structure,seq_value),dim=1).view(max_len*2,ast_traversal.shape[0],1) #Length X Batch X 1

        #Generate ast parent matrix
        # batch X key_len x key_len
        ast_parents_matrix = generate_ast_parents_matrix(
                parent_idx, sequence, padding_idx = self.trg_tokenizer.pad,
                stepwise_decoding = False)
        ##################################################################################
        dec_out,attn = self.decoder(sequence, ast_parents_matrix, tree_coordinates = level, memory_bank = inv_encodings, memory_lengths = lengths)
        structure_dec_out = dec_out[0::2,:,:]
        value_dec_out = dec_out[1::2,:,:]

        value_out = self.value_gen(structure_dec_out) #starts from dummy for root
        #T B C
        structure_out = self.structure_gen(value_dec_out) #starts from child of root

        guidance = guidance.unsqueeze(0) # 1 X batch X vocab
        mask = (seq_structure == self.utility_id) # maxlen X batch X 1
        guidance = guidance * mask # maxlen X batch X vocab
        value_out = value_out + guidance * self.soften_params
        
        logsoftmax = nn.LogSoftmax(dim=-1)
        structure_out = logsoftmax(structure_out)

        value_out_clone = value_out.detach().clone()
        value_out = logsoftmax(value_out)
        #mask irrelevant tokens for computing comparable loss value
        value_out_clone[:,:,self.unk] = -1e20
        value_out_clone[:,:,self.pad] = -1e20
        value_out_clone[:,:,self.trg_tokenizer.value_id.get_id('dummy')] = -1e20
        value_out_clone = logsoftmax(value_out_clone)

        return structure_out, value_out, guidance_loss, value_out_clone

    
    def _cross_attn_mask(self,k_lens,key_max_len):
        batch_size = k_lens.shape[0]
        mask_v = (torch.arange(0, key_max_len, device=k_lens.device)
                .type_as(k_lens)
                .repeat(batch_size, 1)
                .lt(k_lens.unsqueeze(1))).unsqueeze(1)
        return ~mask_v

    def update_dropout(self, dropout):
        self.invocation_encoder.update_dropout(dropout)
        self.value_decoder.update_dropout(dropout)

    ################################# LIGHTNING FUNCTIONS ###################################################
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    
    def shared_step(self,batch,batch_idx,compute_metric = False):
        (inv,inv_len),(y,y_len,y_utils)= batch
        inv = inv.transpose(1,0)
        inv_len = inv_len.squeeze(1).contiguous()
        y_len = y_len.squeeze(1)

        ast_structure, ast_values, guidance_loss, ast_v_mag = self(invocation = inv,\
            invocation_length = inv_len,\
            ast_traversal = y,\
            ast_traversal_length = y_len,
            utility_target = y_utils)
        ast_structure = ast_structure[:-1,:,:] #ignore last output from ET node's value.
        # length X batch X classes
        
        #compute loss
        s_loss_fn = nn.NLLLoss(ignore_index = self.pad)
        v_loss_fn = nn.NLLLoss(ignore_index = self.pad,reduction='sum')

        value_loss = v_loss_fn(ast_values.permute(1,2,0),y[:,:,1])
        structure_loss = s_loss_fn(ast_structure.permute(1,2,0),y[:,1:,0]) #ignore root 
        value_loss /= y_len.sum()
        
        if compute_metric:
            metric_score,err = self.compute_metric_score(inv,inv_len,y,y_len)
        else:
            metric_score = -1
            err = -1

        # return structure_loss, value_loss, metric_score,err, matching_loss
        return structure_loss, value_loss, guidance_loss, metric_score,err

    def training_step(self, batch, batch_idx):
        compute_metric = False
        structure_loss, value_loss, guidance_loss, metric_score, err = self.shared_step(batch,batch_idx,compute_metric=compute_metric)
        total_loss = structure_loss  + self.alpha * value_loss + self.beta * guidance_loss
        self.log('value_loss/training',value_loss)
        self.log('structure_loss/training',structure_loss)
        self.log('guidance_loss/training',guidance_loss)
        self.log('sequence_loss/training',structure_loss + value_loss)
        self.log('optim_loss/training',structure_loss + value_loss + guidance_loss)
        if compute_metric:
            self.log('metric/training',metric_score)
            self.log('errors/training',err)
        return total_loss

    def validation_step(self, batch, batch_idx):
        compute_metric = self.current_epoch > 5 or self.current_epoch==0
        structure_loss ,value_loss, guidance_loss, metric_score, err = self.shared_step(batch,batch_idx,compute_metric=compute_metric)
        total_loss = structure_loss  + self.alpha * value_loss + self.beta * guidance_loss
        self.log('value_loss/validation',value_loss)
        self.log('structure_loss/validation',structure_loss)
        self.log('guidance_loss/validation',guidance_loss)
        self.log('sequence_loss/validation',structure_loss + value_loss)
        self.log('optim_loss/validation',structure_loss + value_loss + guidance_loss)
        if compute_metric:
            self.log('metric/validation',metric_score)
            self.log('errors/validation',err)
        return total_loss
    
    def compute_metric_score(self, inv, inv_len, y, y_len):
        if self.translator is None:
            return -1
        
        self.inference = True
        with torch.no_grad():
            translations = self.translator.translate(inv,inv_len,y,y_len)
            truth_s = [translation.true_structure for translation in translations]
            truth_v = [translation.true_value for translation in translations]
            pred_v = [translation.pred_value for translation in translations]
            pred_s = [translation.pred_structure for translation in translations]
            inv_text = [" ".join(translation.inv) for translation in translations] 
            inv_text_tag = [" ".join(translation.inv_tag) for translation in translations]
            (scores,_),err = get_score(truth_s,truth_v,pred_s,pred_v,inv_text,inv_text_tag)
        mean_score = scores.mean()
        self.inference=False

        return mean_score, err

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--log_hist",type=str2bool,default=False)
        parser.add_argument('--num_filters',type = int, default = 200)
        parser.add_argument('--d_ff',type = int, default = 2048)
        parser.add_argument('--d_model',type = int, default = 512)
        parser.add_argument('--encoder_layers',type = int, default = 6)
        parser.add_argument('--encoder_heads',type = int,default = 8)
        parser.add_argument('--decoder_layers',type = int, default = 6)
        parser.add_argument('--decoder_heads',type = int,default = 8)
        parser.add_argument('--attention_dropout',type = float,default = 0.2)
        parser.add_argument('--dropout',type = float,default = 0)
        parser.add_argument('--alpha',type=float,default = 1)
        parser.add_argument('--beta',type=float,default = 10)
        parser.add_argument('--length_penalty',type=float,default=0.3)
        parser.add_argument('--decoding_strategy',type=str,default = 'beam')
        parser.add_argument('--accumulate_grad_batches',type = int, default = 50)
        parser.add_argument('--beam_size',type = int, default = 10)
        return parser
