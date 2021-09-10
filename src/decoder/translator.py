""" Translator Class """
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from decoder.decoding_strategy import GreedySearch, BeamSearch

from model.utils import generate_ast_parents_matrix
from main_utils import set_random_seed, get_logger
from decoder.utils import report_matrix
logger = get_logger()


class Translator(object):
    """Translate a batch of sentences with a saved model.

    Args:
        model (onmt.modules.NMTModel): NMT model to use for translation
        datamodule (pytorch_lightning.datamodule) : DataModule to use for translation
        n_best (int): How many beams to wait for.
        min_length (int): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        max_length (int): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        beam_size (int): Number of beams.
        random_sampling_topk (int): See
            :class:`onmt.translate.greedy_search.GreedySearch`.
        random_sampling_temp (float): See
            :class:`onmt.translate.greedy_search.GreedySearch`.
        stepwise_penalty (bool): Whether coverage penalty is applied every step
            or not.
        dump_beam (bool): Debugging option.
        block_ngram_repeat (int): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        ignore_when_blocking (set or frozenset): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        tgt_prefix (bool): Force the predictions begin with provided -tgt.
        data_type (str): Source data type.
        verbose (bool): Print/log every translation.
        report_time (bool): Print/log total time/frequency.
        global_scorer (onmt.translate.GNMTGlobalScorer): Translation
            scoring/reranking object.
        report_score (bool) : Whether to report scores
        logger (logging.Logger or NoneType): Logger.
    """

    def __init__(
        self,
        model,
        datamodule,
        decoding_strategy,
        n_best=1,
        min_length=2,
        max_length=128,
        ratio=0.0,
        beam_size=1,
        random_sampling_topk=0,
        random_sampling_topp=0,
        random_sampling_temp=1.0,
        stepwise_penalty=False,
        dump_beam='',
        block_ngram_repeat=0,
        ignore_when_blocking=[],
        ban_unk_token=False, 
        tgt_prefix=False,
        data_type="text",
        verbose=False,
        report_time=False,
        global_scorer=None,
        report_score=True,
        logger=None,
    ):
        self.model = model
        self._src_vocab = datamodule.src_tokenizer
        self._tgt_vocab = datamodule.trg_tokenizer
        self._tgt_eos_idx = datamodule.trg_tokenizer.kind_id.get_id('ET') #ET node 
        self._tgt_bos_idx = datamodule.trg_tokenizer.kind_id.get_id('root') #ROOT node
        self._tgt_utility_idx = datamodule.trg_tokenizer.kind_id.get_id('utility') #Utility node
        self._tgt_unk_idx = datamodule.trg_tokenizer.kind_id.unknown_id
        self._tgt_eoc_idx = datamodule.trg_tokenizer.kind_id.get_id('EC')
        self.max_level = datamodule.training_data.max_trg_level

        self.n_best = n_best
        self.min_length = min_length
        self.max_length = max_length

        self.beam_size = beam_size
        self.random_sampling_temp = random_sampling_temp
        self.sample_from_topk = random_sampling_topk
        self.sample_from_topp = random_sampling_topp

        self.ban_unk_token = ban_unk_token
        self.ratio = ratio
        self.stepwise_penalty = stepwise_penalty
        self.dump_beam = dump_beam
        self.block_ngram_repeat = block_ngram_repeat
        self.ignore_when_blocking = ignore_when_blocking
        self._exclusion_idxs = {
            self._tgt_vocab.get_id(t) for t in self.ignore_when_blocking
        }
        self.tgt_prefix = tgt_prefix
        self.data_type = data_type
        self.global_scorer = global_scorer
        self.decoding_strategy = decoding_strategy
        
        self.verbose = verbose
        self.report_time = report_time
        
        self.report_score = report_score

        self.use_filter_pred = False
        self._filter_pred = None

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None
        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": [],
            }

    def translate(
        self,
        inv,inv_len,y,y_len,
        attn_debug=False,
        has_tgt=False,
        get_guidance_distribution=False
    ):
        """
        Returns:
            (`list`, `list`, `list`)

            * all_scores is a list of `batch_size` lists of `n_best` scores
            * all_predictions is a list of `batch_size` lists
                of `n_best` predictions
        """
        # Statistics

        translations = self._translate_batch_with_strategy(inv,inv_len,y,get_guidance_distribution)
        return translations

    def _decode_and_generate(
        self,
        decoder_in,
        ast_parents_matrix,
        tree_coordinates,
        memory_bank,
        guidance,
        memory_lengths,
        step=None,
    ):
        # print(self.model.state_dict()['decoder.transformer_layers.3.feed_forward.w_2.bias'])
        # Decoder forward, takes [tgt_len, batch, nfeats] as input
        # and [src_len, batch, hidden] as memory_bank
        # in case of inference tgt_len = 1, batch = beam x batch_size
        # in case of Gold Scoring tgt_len = actual length, batch = 1 batch
        dec_out, dec_attn = self.model.decoder(
            decoder_in, ast_parents_matrix, tree_coordinates = tree_coordinates, memory_bank = memory_bank, memory_lengths=memory_lengths, step=step, max_dec_len = self.max_length
        )
        if step%2==0:
            #add guidance to logits
            logits = self.model.value_gen(dec_out)
            guidance = guidance.unsqueeze(0) # 1 X batch X vocab
            mask = (decoder_in == self._tgt_utility_idx) # 1 X batch X 1
            guidance = guidance * mask # 1 X batch X vocab
            logits = logits + guidance * self.model.soften_params.to(logits.device)
        else:
            logits = self.model.structure_gen(dec_out)
        
        if logits.shape[0]==1:
            logits = logits.squeeze(0)
        # returns [(batch_size x beam_size) , vocab ] when 1 step
        # or [ tgt_len, batch_size, vocab ] when full sentence
        
        if "std" in dec_attn:
            attn = dec_attn["std"]
        else:
            attn = None
        
        log_probs = F.log_softmax(logits, dim=-1)
        # print(log_probs)
        # print(decoder_in,parents_idx,tree_coordinates)
        return log_probs, attn
    
    def test_decode_and_generate(
        self,
        select_indices,
        parent_idx,
        tree_coordinates,
        ast_traversal,
        parallel_paths,
        step=None,
    ):
        # in case of inference tgt_len = 1, batch = beam x batch_size
        if select_indices is None:
            select_indices = torch.arange(parallel_paths * ast_traversal.shape[0])
        
        if step%2==0: #value prediction
            correct_index = ast_traversal[select_indices,(step+1)//2,1]
            logits = torch.zeros(size = (parent_idx.shape[1],self._tgt_vocab.value_id.vocab_len))
        else: #structure prediction
            correct_index = ast_traversal[select_indices,(step+1)//2,0]
            logits = torch.zeros(size = (parent_idx.shape[1],self._tgt_vocab.kind_id.vocab_len))
        correct_index = correct_index.unsqueeze(1).repeat(1,parallel_paths)
        correct_index = correct_index.view(-1)
        logits[torch.arange(parent_idx.shape[1]),correct_index] = 1e10

        # returns [(batch_size x beam_size) , vocab ] when 1 step
        # or [ tgt_len, batch_size, vocab ] when full sentence
        

        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, None


    def get_strategy(self,batch_size,attn_debug=False):
        if self.decoding_strategy == 'greedy':
            decode_strategy = GreedySearch(
                    bos=self._tgt_bos_idx,
                    eos=self._tgt_eos_idx,
                    eoc=self._tgt_eoc_idx,
                    unk=self._tgt_unk_idx,
                    batch_size = batch_size,
                    global_scorer=self.global_scorer,
                    min_length=self.min_length,
                    max_length=self.max_length,
                    block_ngram_repeat=self.block_ngram_repeat,
                    exclusion_tokens=self._exclusion_idxs,
                    return_attention = attn_debug,
                    sampling_temp=self.random_sampling_temp,
                    keep_topk=self.sample_from_topk,
                    keep_topp=self.sample_from_topp,
                    beam_size=1,
                    ban_unk_token=self.ban_unk_token,
                    max_level = self.max_level
                )
        elif self.decoding_strategy=='beam':
            decode_strategy = BeamSearch(
                    beam_size = self.beam_size,
                    batch_size= batch_size,
                    bos=self._tgt_bos_idx,
                    eos=self._tgt_eos_idx,
                    eoc=self._tgt_eoc_idx,
                    unk=self._tgt_unk_idx,
                    n_best=self.n_best,
                    global_scorer=self.global_scorer,
                    min_length=self.min_length,
                    max_length=self.max_length,
                    return_attention=attn_debug,
                    block_ngram_repeat=self.block_ngram_repeat,
                    exclusion_tokens=self._exclusion_idxs,
                    stepwise_penalty=self.stepwise_penalty,
                    ratio=self.ratio,
                    ban_unk_token=self.ban_unk_token,
                    max_level = self.max_level
                )
        else:
            raise Exception
        return decode_strategy

    def _translate_batch_with_strategy(self, invocation,invocation_length,ast_traversal,get_guidance_distribution):
        """Translate a batch of sentences step by step using cache.
        invocation : len x batch x 2
        invocation_length : len
        Returns:
            results (dict): The translation results.
        """
        batch_size = invocation.shape[1]
        decode_strategy = self.get_strategy(batch_size)
        # (0) Prep the components of the search.
        parallel_paths = decode_strategy.parallel_paths  # beam_size

        # (1) Run the encoder on the src.
        _,inv_encodings,lengths= self.model.invocation_encoder(invocation, invocation_length)
        # inv_encodings: max_len X batch X dim

        guidance, _, guidance_distribution = self.model.utility_guide(invocation,return_distribution = get_guidance_distribution)
        if get_guidance_distribution:
            return_guidance_prediction_ = guidance.detach().clone()
            return_guidance_distribution_ = guidance_distribution
        else:
            return_guidance_distribution_ = None
            return_guidance_prediction_ = None

        self.model.decoder.init_state(invocation)

        # (2) prep decode_strategy. Possibly repeat src objects.
        src_map = None
        (
            fn_map_state,
            memory_bank,
            guidance,
            memory_lengths,
            src_map
        ) = decode_strategy.initialize(inv_encodings, guidance, lengths, src_map)
        
        assert fn_map_state is not None
        self.model.decoder.map_state(fn_map_state)
        select_indices = None
        
        # (3) Begin decoding step by step:
        for step in range(decode_strategy.max_length):
            decoder_input = decode_strategy.current_predictions.view(1, -1, 1)
            parent_idx = decode_strategy.current_parents
            tree_coordinates = decode_strategy.current_tree_coordinates.view(1, -1, 1)
            # print('Tree coord---',tree_coordinates.squeeze(0).squeeze(1)[:10])
            # print('parent idx--',parent_idx.squeeze(1)[:10])
            # print('Decoder input--',decoder_input.squeeze(0).squeeze(1)[:10])
            #Generate AST Parent Matrix row
            ast_input_sequence = decode_strategy.alive_seq.clone().transpose(0,1) # seqlen X B 
            ast_parents_matrix = generate_ast_parents_matrix(parent_idx,ast_input_sequence,self._tgt_vocab.pad,stepwise_decoding=True)

            logits, attn = self._decode_and_generate(
                decoder_input,
                ast_parents_matrix,
                tree_coordinates,
                memory_bank,
                guidance,
                memory_lengths=memory_lengths,
                step=step
            )
            # if step%2==0:
            #     print(logits[:10,82],'82')
            #     print(logits[:10,5],'5')
            ##TEST
            # logits,attn = self.test_decode_and_generate(select_indices,parent_idx,tree_coordinates,ast_traversal,parallel_paths=parallel_paths,step=step)

            is_type_token = logits.shape[-1] == self._tgt_vocab.kind_id.vocab_len
            assert (is_type_token and step%2==1) or (not is_type_token and step%2==0)
            decode_strategy.advance(logits, attn, is_type_token)
            any_finished = decode_strategy.is_finished.any()
            if any_finished:
                decode_strategy.update_finished()
                if decode_strategy.done:
                    break

            select_indices = decode_strategy.select_indices
            # print('Select indices--',select_indices[:10])
            if any_finished:
                # Reorder states.
                if isinstance(memory_bank, tuple):
                    memory_bank = tuple(
                        x.index_select(1, select_indices) for x in memory_bank
                    )
                else:
                    memory_bank = memory_bank.index_select(1, select_indices)
                memory_lengths = memory_lengths.index_select(0, select_indices)
                guidance = guidance.index_select(0,select_indices)

            if parallel_paths > 1 or any_finished:
                self.model.decoder.map_state(
                    lambda state, dim: state.index_select(dim, select_indices)
                )
            # print('--'*65)

        translation_builder = TranslationBuilder(self._src_vocab,self._tgt_vocab)
        translations = translation_builder.build_batch(decode_strategy.predictions,\
                decode_strategy.scores,\
                ast_traversal, invocation,\
                invocation_length,return_guidance_distribution_,\
                return_guidance_prediction_)
        return translations


class TranslationBuilder:
    def __init__(self,src_tokenizer,trg_tokenizer):
        self.src_tokenizer =  src_tokenizer
        self.trg_tokenizer = trg_tokenizer
    
    def tokens_to_text(self,toklist,vocab,maxlen=10000):
        tokens = []
        for i,tok in enumerate(toklist):
            if i==maxlen:
                break
            tokid = tok.item()
            token = vocab.get_token(tokid)
            tokens.append(token)
            if token == 'ET':
                break
        return tokens

    def build_batch(self,pred,score,truth,invocation,invocation_length,guidance_distribution,guidance_predictions):
        translations = []
        for b in range(len(pred)):
            inv = invocation[:invocation_length[b],b,0]
            inv_tag = invocation[:invocation_length[b],b,1]
            guidance_distribution_ = guidance_distribution[b] if guidance_distribution is not None else None
            guidance_predictions_ = guidance_predictions[b] if guidance_predictions is not None else None
            text = [self.src_tokenizer.text.get_token(tokid.item()) for tokid in inv]
            text_tag = [self.src_tokenizer.pos.get_token(tokid.item()) for tokid in inv_tag]
            translation = Translation(text,text_tag, guidance_distribution_,guidance_predictions_)
            assert len(score[b]) == len(pred[b])
            for pred_score,cur_pred in zip(score[b],pred[b]):
                cur_truth = truth[b,:,:-1]
                p_s = ['root'] + self.tokens_to_text(cur_pred[1::2],self.trg_tokenizer.kind_id)
                t_s = self.tokens_to_text(cur_truth[:,0], self.trg_tokenizer.kind_id)
                p_v = self.tokens_to_text(cur_pred[::2],self.trg_tokenizer.value_id, maxlen = len(p_s)-1) + ['dummy']
                t_v = self.tokens_to_text(cur_truth[:,1], self.trg_tokenizer.value_id, maxlen = len(t_s))
                translation.append(t_s,t_v,p_s,p_v,pred_score)
            translations.append(translation)
        return translations


class Translation: 
    def __init__(self,text,text_tag,guidance_distribution = None,guidance_predictions=None):
        self.true_structure = []
        self.true_value = []
        self.pred_structure = []
        self.pred_value = []
        self.pred_score = []
        self.inv = text
        self.inv_tag = text_tag
        self.guidance_distribution = guidance_distribution
        self.guidance_predictions = guidance_predictions

    def append(self,ts,tv,ps,pv,pred_score):
        self.true_structure.append(ts)
        self.true_value.append(tv)
        self.pred_structure.append(ps)
        self.pred_value.append(pv)
        self.pred_score.append(pred_score)

    
    
