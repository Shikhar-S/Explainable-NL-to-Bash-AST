import os
import pathlib
import sys

from pytorch_lightning.core.hooks import CheckpointHooks

from decoder.decoding_strategy import GNMTGlobalScorer
from model.plite_callbacks import TensorboardLoggingCallback
from model.utility_guide import UtilityGuide
run_base_path = pathlib.Path(__file__).parent.absolute()
module_path = os.path.abspath(os.path.join(run_base_path,'clai/tellina-baseline/src'))
if module_path not in sys.path:
    sys.path.insert(0,module_path)

import torch
import argparse
import pytorch_lightning as pl
import numpy as np

from main_utils import get_logger, get_device, set_random_seed, init_weights, str2bool
from reader.datamodule import DataModule
from model.seq2ast import Seq2BashAST
from model.embedding_layer import Embeddings
from model.encoder import TransformerEncoder
from model.decoder import TransformerDecoder
from decoder.translator import Translator
from decoder.bash_generator import get_score
import pickle


def build_model(args,dm):
    inp_vocab_sz = dm.src_tokenizer.text.vocab_len
    pos_vocab_sz = dm.src_tokenizer.pos.vocab_len
    kind_vocab_sz = dm.trg_tokenizer.kind_id.vocab_len
    value_vocab_sz = dm.trg_tokenizer.value_id.vocab_len
    embedding_dim  = args.d_model
    max_trg_len = max(dm.training_data.max_trg_length,max(dm.testing_data.max_trg_length,dm.validation_data.max_trg_length)) * 2 
    max_trg_level = dm.training_data.max_trg_level + 1
    pad_idx = dm.src_tokenizer.pad
    

    assert ((dm.word_vectors is None) or (dm.word_vectors.shape[0] == inp_vocab_sz))

    input_embedding_layer = Embeddings(embedding_dim,\
                                inp_vocab_sz,\
                                pad_idx,\
                                position_encoding=True,\
                                feat_merge='sum',\
                                feat_padding_idx=[pad_idx],\
                                feat_vocab_sizes=[pos_vocab_sz],\
                                freeze_word_vecs=False,\
                                word_vectors = dm.word_vectors)

    invocation_encoder = TransformerEncoder(args.encoder_layers,\
                                args.d_model,\
                                args.encoder_heads,\
                                args.d_ff,\
                                args.dropout,
                                args.attention_dropout,
                                input_embedding_layer)

    value_embedding_layer = Embeddings(word_vec_size = embedding_dim,\
                                word_vocab_size = value_vocab_sz,\
                                word_padding_idx = pad_idx,\
                                position_encoding=True,\
                                feat_merge= 'sum',\
                                feat_padding_idx = [None],\
                                feat_vocab_sizes = [max_trg_level],\
                                freeze_word_vecs = False,\
                                word_vectors = None)

    structure_embedding_layer = Embeddings(word_vec_size = embedding_dim,\
                                word_vocab_size = kind_vocab_sz,\
                                word_padding_idx = pad_idx,\
                                position_encoding=True,\
                                feat_merge= 'sum',\
                                feat_padding_idx = [None],\
                                feat_vocab_sizes = [max_trg_level],\
                                freeze_word_vecs = False,\
                                word_vectors = None)

    decoder = TransformerDecoder(args.decoder_layers,
                                args.d_model,
                                args.decoder_heads,
                                args.d_ff,
                                args.dropout,
                                args.attention_dropout,
                                embeddings_structure = structure_embedding_layer,
                                embeddings_value = value_embedding_layer,
                                graph_input = True,
                                vocab_sizes = (kind_vocab_sz,value_vocab_sz),
                                padding_idx = pad_idx)
    
    guiding_module = UtilityGuide(dm.documentation_data,\
                                [1,2,3,4,5,6],\
                                 args.num_filters,\
                                 input_embedding_layer,\
                                 args.d_model,\
                                 value_vocab_sz,\
                                 pad_idx)

    model = Seq2BashAST(invocation_encoder = invocation_encoder,\
                        utility_guide=guiding_module,\
                        decoder = decoder,\
                        d_model = args.d_model,\
                        trg_tokenizer=dm.trg_tokenizer,\
                        alpha = args.alpha,\
                        beta = args.beta,\
                        device = args.device)
    return model

def set_translator(args,model,dm):
    max_trg_len = max(dm.training_data.max_trg_length,max(dm.testing_data.max_trg_length,dm.validation_data.max_trg_length)) * 2 
    
    global_scorer = GNMTGlobalScorer(alpha = args.length_penalty,length_penalty="wu")
    translator = Translator(model = model,
                            datamodule=dm,
                            decoding_strategy=args.decoding_strategy,
                            n_best=5,
                            min_length=4, #root and ET
                            max_length=max_trg_len,
                            random_sampling_topk=10,
                            random_sampling_topp=0.7,
                            report_score=False,
                            beam_size=args.beam_size,
                            global_scorer=global_scorer)
    model.init_translator(translator)

def get_results(args,model,dataloader):
    result = {'truth_s':[],'truth_v':[],\
                'pred_s':[],'pred_v':[],\
                'model_score':[],'invocation_text':[],\
                'invocation_tag':[],'metric':[],\
                'all_metric':[],'error':[],\
                'guidance_distribution':[],'guidance_predictions':[]}
    result['documentation_text'] = model.utility_guide.utility_description
    result['documentation_utility_to_idx'] = model.utility_guide.utility_idx
    with torch.no_grad():
        bidx = 0
        for batch in dataloader:
            bidx+=1
            if bidx%100==0:
                print(bidx)
            (inv,inv_len),(y,y_len,_)= batch
            inv = inv.transpose(1,0)
            inv_len = inv_len.squeeze(1).contiguous()
            y_len = y_len.squeeze(1)
            translations = model.translator.translate(inv,inv_len,y,y_len,get_guidance_distribution = args.guidance_distribution)

            truth_s = [translation.true_structure for translation in translations]
            truth_v = [translation.true_value for translation in translations]
            pred_v = [translation.pred_value for translation in translations]
            pred_s = [translation.pred_structure for translation in translations]
            model_score = [translation.pred_score for translation in translations]
            inv_text = [" ".join(translation.inv) for translation in translations] 
            inv_text_tag = [" ".join(translation.inv_tag) for translation in translations]
            guidance_distribution = [translation.guidance_distribution for translation in translations]
            guidance_predictions = [translation.guidance_predictions for translation in translations]
            (metric,all_metric),err = get_score(truth_s,truth_v,pred_s,pred_v,inv_text,inv_text_tag)
            result['pred_s'].extend(pred_s)
            result['pred_v'].extend(pred_v)
            result['truth_s'].extend(truth_s)
            result['truth_v'].extend(truth_v)
            result['model_score'].extend(model_score)
            result['invocation_text'].extend(inv_text)
            result['invocation_tag'].extend(inv_text_tag)
            result['metric'].extend(metric) 
            result['all_metric'].extend(all_metric)
            result['error'].append(err)
            result['guidance_distribution'].append(guidance_distribution)
            result['guidance_predictions'].append(guidance_predictions)
    return result

def predict(args):
    print('Split',args.split)
    print('Checkpoint_path',args.checkpoint_path)

    dm = DataModule(args)
    dm.setup(stage = args.mode)

    model = build_model(args,dm)
    model_loaded = torch.load(args.checkpoint_path, map_location=args.device)
    model.load_state_dict(model_loaded['state_dict'])
    model.eval()
    model.freeze()
    set_translator(args, model, dm)
    
    results = get_results(args,model,dm.val_dataloader())
    with open(f'./result_val.{args.split}.pkl','wb') as f:
        pickle.dump(results,f)
    print('Dumped val results')

    results = get_results(args,model,dm.test_dataloader())
    with open(f'./result_test.{args.split}.pkl','wb') as f:
        pickle.dump(results,f)
    print('Dumped test results')

    results = get_results(args,model,dm.train_dataloader())
    with open(f'./result_train.{args.split}.pkl','wb') as f:
        pickle.dump(results,f)
    print('Dumped train results')
       

def train(args):
    set_random_seed(args.seed,args.device == torch.device('cuda'))
    logger.info(f'Random Seed:{args.seed}')

    dm = DataModule(args)
    dm.setup(stage=args.mode)
    model = build_model(args,dm)

    if dm.word_vectors is not None:
        init_ignore = ['invocation_encoder.embeddings.make_embedding.emb_luts.0.weight','soften_params']
    else:
        init_ignore = ['soften_params']

    init_weights(model,init_ignore)
    set_translator(args, model, dm)

    #callbacks for logging/monitoring
    earlystop_callback = pl.callbacks.EarlyStopping(patience=40,verbose=True, monitor = 'metric/validation', mode = 'max')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='metric/validation',mode='max',\
                                                        filename='{epoch}')
    tb_logging_callback = TensorboardLoggingCallback(dm,args)

    run_base_path = pathlib.Path(__file__).parent.parent.absolute()
    run_dir = os.path.join(run_base_path,args.run_base_address,'split.' + str(args.split))
    logger.info('Storing model at:: '+run_dir)
    device_id = -1 if args.device == torch.device('cuda') else 0
    
    trainer = pl.Trainer.from_argparse_args(args,gpus = device_id,\
                                            default_root_dir=run_dir,\
                                            callbacks = [earlystop_callback, checkpoint_callback, tb_logging_callback],\
                                            accumulate_grad_batches=args.accumulate_grad_batches)
    trainer.fit(model=model,datamodule=dm)
    # print('loss:',-1 * checkpoint_callback.best_model_score.item())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',type = int,default=1)
    parser.add_argument('--mode',type=str , default='train')
    parser.add_argument('--device',default='auto')
    parser.add_argument('--checkpoint_path',type=str,default='')
    parser.add_argument('--verbose',type=str2bool,default = False)
    parser.add_argument('--log_graph',type=str2bool,default=False)
    parser.add_argument('--log_embeddings',type=str2bool,default=False)
    parser.add_argument('--guidance_distribution',type=str2bool,default=False)

    parser = DataModule.add_model_specific_args(parser)
    parser = Seq2BashAST.add_model_specific_args(parser)
    
    args,unparsed = parser.parse_known_args()
    args.device = get_device(args)
    global logger
    logger = get_logger()
    logger.info('__INIT_PROCESS__')
    logger.info('Arguments::' + str(args))
  
    if len(unparsed)>0:
        print('Unparsed args: %s',unparsed)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'predict':
        predict(args)
