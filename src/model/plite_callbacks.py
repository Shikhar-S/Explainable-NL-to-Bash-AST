from pytorch_lightning.callbacks import Callback
from pytorch_lightning import LightningModule

import numpy as np
from model import encoder

class TensorboardLoggingCallback(Callback):
    def __init__(self,dm,args) -> None:
        super(TensorboardLoggingCallback,self).__init__()
        if args.log_graph:
            self.graph_logging_input = self._get_graph_logging_input(dm)
        else:
            self.graph_logging_input = None
        self.args = args
        self.dm = dm
        self.embedding_logging_interval = 35
    
    def _get_graph_logging_input(self,dm):
        for batch in dm.train_dataloader():
            (inv,inv_len),(y,y_len,y_utils)= batch
            inv = inv.transpose(1,0)
            inv_len = inv_len.squeeze(1).contiguous()
            y_len = y_len.squeeze(1)
            return [inv,inv_len,y,y_len,y_utils]

    def log_embeddings(self,pl_module):
        encoder_embed = pl_module.invocation_encoder.embeddings.make_embedding.emb_luts[0].weight
        decoder_structure_embed = pl_module.decoder.embeddings_structure.make_embedding.emb_luts[0].weight
        decoder_value_embed = pl_module.decoder.embeddings_value.make_embedding.emb_luts[0].weight

        encoder_meta = [self.dm.src_tokenizer.text_get_token(i) for i in range(self.dm.src_tokenizer.text.vocab_len)]
        decoder_structure_meta = [self.dm.trg_tokenizer.get_kind_token(i) for i in range(self.dm.trg_tokenizer.kind_id.vocab_len)]
        decoder_value_meta = [self.dm.trg_tokenizer.get_value_token(i) for i in range(self.dm.trg_tokenizer.value_id.vocab_len)]
        decoder_value_meta = list(map(lambda x : '[EMPTY]' if x=='' else x,decoder_value_meta))

        assert len(encoder_meta) == encoder_embed.shape[0]
        assert len(decoder_structure_meta) == decoder_structure_embed.shape[0]
        assert len(decoder_value_meta) == decoder_value_embed.shape[0]
        pl_module.logger.experiment.add_embedding(encoder_embed, metadata=encoder_meta, tag='encoder_embeddings' ,global_step = pl_module.current_epoch)
        pl_module.logger.experiment.add_embedding(decoder_structure_embed, metadata=decoder_structure_meta, tag='decoder_structure_embeddings',global_step = pl_module.current_epoch)
        pl_module.logger.experiment.add_embedding(decoder_value_embed, metadata=decoder_value_meta, tag='decoder_value_embeddings',global_step = pl_module.current_epoch)

    def on_fit_start(self, trainer, pl_module: LightningModule) -> None:
        #Log Graph
        if self.args.log_graph:
            pl_module.logger.log_graph(pl_module, input_array = self.graph_logging_input)
        
        #Log Hyper-params
        full_dict = vars(self.args)
        log_dict = { filtered_key : full_dict[filtered_key] for filtered_key in ['accumulate_grad_batches', 'alpha',
                                                                                    'attention_dropout', 'batch',
                                                                                    'beam_size', 'beta',
                                                                                    'command_normalised', 'd_ff',
                                                                                    'd_model', 'decoder_heads',
                                                                                    'decoder_layers', 'dropout',
                                                                                    'encoder_heads', 'encoder_layers', 
                                                                                    'max_len', 'num_filters',
                                                                                    'seed', 'split']
                    }
        pl_module.logger.log_hyperparams(params = log_dict)
    
    def on_fit_end(self, trainer, pl_module: LightningModule) -> None:
        #Log Embedding Matrices
        if self.args.log_embeddings:
            self.log_embeddings(pl_module)


    def on_epoch_end(self, trainer, pl_module: LightningModule) -> None:
        if self.args.log_hist:
            for name,params in pl_module.named_parameters():
                pl_module.logger.experiment.add_histogram(name.replace('.','/'),params,pl_module.current_epoch)
