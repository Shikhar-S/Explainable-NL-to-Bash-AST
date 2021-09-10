from main_utils import get_logger
logger = get_logger()

import torch
from torch.utils.data import DataLoader
from reader.vocab import Vocab
from reader.datareader import DataReader, DynamicSampler
import pickle
from main_utils import str2bool,get_logger, check_path
from argparse import ArgumentParser
import numpy as np
import os
import pytorch_lightning as pl
import pathlib
import nltk
logger = get_logger()

            

class DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super(DataModule,self).__init__()
        self.process_data = args.process_data
        self.batch_size = args.batch
        self.max_len = args.max_len
        data_base_address = args.data_base_address
        self.split = args.split
        cmd_extension = '.cmd.raw' 
        eng_extension = '.en'
        dyn_path = pathlib.Path(__file__).parent.parent.parent.absolute()
        self.training_data_path = os.path.join(dyn_path,data_base_address,'train.' +str(self.split)+ eng_extension),\
                                os.path.join(dyn_path,data_base_address,'train.' +str(self.split) + cmd_extension)
        self.validation_data_path = os.path.join(dyn_path,data_base_address,'valid.' +str(self.split) + eng_extension),\
                                os.path.join(dyn_path,data_base_address,'valid.' +str(self.split) + cmd_extension)
        self.testing_data_path = os.path.join(dyn_path,data_base_address,'test.' +str(self.split) + eng_extension),\
                                os.path.join(dyn_path,data_base_address,'test.' +str(self.split)+ cmd_extension)

        self.extra_data_paths = [(os.path.join(dyn_path,data_base_address,datasource + eng_extension),\
                                os.path.join(dyn_path,data_base_address,datasource + cmd_extension))\
                                for datasource in args.extra_data_path]

        run_base_address = args.run_base_address
        self.run_address = os.path.join(dyn_path,run_base_address,'split.' + str(self.split))
        self.src_tokenizer = None
        self.trg_tokenizer = None
        
        if args.embedding_path != 'None':
            self.embedding_path = os.path.join(dyn_path,args.embedding_path)
        else:
            self.embedding_path = None
        self.device = args.device
        self.documentation_path = os.path.join(dyn_path,args.documentation_path)


    def get_data(self,path,dtype = 'train'):
        datapath = os.path.join(self.run_address, dtype + 'transformer_explain.pkl')
        check_path(datapath,exist_ok = True)
        if self.process_data or not os.path.exists(datapath):
            logger.info(f'Processing {dtype} data')
            if dtype == 'train':
                dataset = DataReader(path, self.max_len, self.src_tokenizer, self.trg_tokenizer,self.extra_data_paths)
            else:
                dataset = DataReader(path, self.max_len, self.src_tokenizer, self.trg_tokenizer)
            logger.info(f'Saving {dtype} data')
            with open(datapath,'wb') as f:
                pickle.dump(dataset,f)
            logger.info(f'Saved {dtype} data')
        else:
            #load from pickled file
            logger.info(f'Loading {dtype} data from pickled file')
            with open(datapath,'rb') as f:
                dataset = pickle.load(f)
        logger.info(f'Count: SRC_VOCAB text = {dataset.src_tokenizer.text.vocab_len} :: pos = {dataset.src_tokenizer.pos.vocab_len}')
        self.src_tokenizer = dataset.src_tokenizer
        self.trg_tokenizer = dataset.trg_tokenizer
        return dataset


    def setup(self, stage=None):
        if stage is None:
            logger.warning(f'Datamodule setup called with stage = {stage}')
        else:
            logger.info(f'Setting up Datamodule in {stage} stage.')

        # load training data, save dictionary, load validation and testing data
        self.training_data = self.get_data(self.training_data_path,dtype = 'train')
        self.validation_data = self.get_data(self.validation_data_path,dtype = 'valid')
        self.testing_data = self.get_data(self.testing_data_path,dtype ='test')
        self.add_documentation_to_vocab()
        self.filter_tokens(self.src_tokenizer.text,2500,self.embedding_path)
        self.word_vectors = self.load_word_vec()
        self.documentation_data = self.get_documentation_data()
    

    def get_documentation_data(self):
        def tokenize_text(text):
            if text != text:
                return []
            text = text.strip()
            pos_tags = [self.src_tokenizer.pos_get_id(tag[1]) for tag in nltk.pos_tag(text.split())]
            token_list = []
            for i,token in enumerate(text.split()):
                token_list.append([self.src_tokenizer.text_get_id(token),pos_tags[i]])
            return token_list

        utility_id = []
        documentation_text = []
        max_doc_len = 0
        #Get token ids from csv
        import pandas as pd
        doc_df = pd.read_csv(self.documentation_path)
        for id,row in doc_df.iterrows():
            utility_name = row['Utility'].strip()
            tldr_text = row['TLDRText']
            manpage_text = row['ManpageText']
            tldr_tokens = tokenize_text(tldr_text)
            manpage_tokens = tokenize_text(manpage_text)
            utility_id.append(self.trg_tokenizer.get_value_id(utility_name))
            documentation_text.append(manpage_tokens + tldr_tokens)
            max_doc_len = max(max_doc_len,len(manpage_tokens) + len(tldr_tokens))
        #Pad all doc to max length
        for i,doc in enumerate(documentation_text):
            documentation_text[i] = doc + [[self.src_tokenizer.pad,self.src_tokenizer.pad]] * (max_doc_len - len(doc))
        #Convert to Tensors
        documentation_text = torch.LongTensor(documentation_text).to(self.device)
        utility_id = torch.LongTensor(utility_id).to(self.device)
        return utility_id,documentation_text


    def add_documentation_to_vocab(self):
        def add_doc_to_vocab(text):
            if text != text: #nan
                return            
            text = text.strip()
            text_pos = [tag[1] for tag in nltk.pos_tag(text.split())]
            for token,pos_tag in zip(text.split(),text_pos):
                self.src_tokenizer.text.add_token(token)
                self.src_tokenizer.pos.add_token(pos_tag)
        import pandas as pd
        doc_df = pd.read_csv(self.documentation_path)
        for id,row in doc_df.iterrows():
            utility_name = row['Utility'].strip()
            tldr_text = row['TLDRText']
            manpage_text = row['ManpageText']
            add_doc_to_vocab(tldr_text)
            add_doc_to_vocab(manpage_text)
            self.trg_tokenizer.value_id.add_token(utility_name)


    def _get_embedding_words(self,path=None):
        if path is None:
            return []
        embedding_words = set()
        with open(path,'r',encoding='Latin-1') as vec_f:
            n,vec_dim = map(int,vec_f.readline().rstrip().split())
            for line in vec_f:
                word = line.strip().split()[0]
                embedding_words.add(word)
        logger.info(f'Loaded {len(embedding_words)} embedding words')
        return embedding_words


    def filter_tokens(self,full_vocab,k,path=None):
        logger.info('Filtering words from dataset.')
        logger.info(f'Original dictionary size {full_vocab.vocab_len}')
        embedding_words = self._get_embedding_words(path)
        token_counter = full_vocab.token_counter
        top_k_words = set(map(lambda z: z[0],token_counter.most_common(k)))
        filtered_embeddding_words = set()
        for word in embedding_words:
            if word in token_counter.keys():
                filtered_embeddding_words.add(word)
        filtered_words = list(top_k_words.union(filtered_embeddding_words))
        filtered_words.sort() 
        logger.info(f'Words remaining {len(filtered_words)}')
        filtered_tokenizer = Vocab()
        for word in filtered_words:
            filtered_tokenizer.add_token(word)
        #note this handles reserved words automatically
        self.training_data.src_tokenizer.text = filtered_tokenizer
        self.validation_data.src_tokenizer.text = filtered_tokenizer
        self.testing_data.src_tokenizer.text = filtered_tokenizer
        self.src_tokenizer.text = filtered_tokenizer


    def load_word_vec(self):
        if self.embedding_path is None:
            return None
        dataset_vocab = self.src_tokenizer.text.stoi
        c=0
        with open(self.embedding_path,'r') as vec_f:
            n,vec_dim = map(int,vec_f.readline().rstrip().split())
            word_vectors = np.random.normal(0, 0.1,(len(dataset_vocab),vec_dim))
            for line in vec_f:
                word = line.strip().split()[0]
                if word in dataset_vocab:
                    vec = list(map(float,line.strip().split()[1:]))
                    word_vectors[dataset_vocab[word]] = vec
                    c+=1
        logger.info(f'Loaded {c} word vectors')
        return word_vectors


    def collate_fn(self,batch):
        inv_token_batch = []
        inv_len_batch = []
        ast_traversal_batch = []
        ast_traversal_len_batch = []
        utility_batch = []
        max_traversal_len = 0
        max_utility_len = 0
        for batch_item  in batch:
            (inv_token,inv_length), (ast_traversal,ast_traversal_len,utility_tensor) = batch_item
            inv_token_batch.append(inv_token)
            inv_len_batch.append(inv_length)
            ast_traversal_batch.append(ast_traversal)
            ast_traversal_len_batch.append(ast_traversal_len)
            utility_batch.append(utility_tensor)
            max_utility_len = max(max_utility_len,utility_tensor.shape[0])
            max_traversal_len = max(max_traversal_len,ast_traversal_len.item())
        
        #PAD ast traversal 
        pad_idx = self.trg_tokenizer.kind_id.pad
        for i,item in enumerate(ast_traversal_batch):
            item_len = len(item)
            pad_tensor = torch.LongTensor([[pad_idx,pad_idx,pad_idx,pad_idx]])
            pad_tensor = pad_tensor.repeat(max_traversal_len - item_len, 1)
            ast_traversal_batch[i] = torch.cat([item,pad_tensor])
        
        #PAD utility tensor
        for i,item in enumerate(utility_batch):
            item_len = item.shape[0]
            pad_tensor = torch.LongTensor([pad_idx])
            pad_tensor = pad_tensor.repeat(max_utility_len - item_len)
            utility_batch[i] = torch.cat([item,pad_tensor])

        inv_token_batch = torch.stack(inv_token_batch) #batch x maxlen X 2
        inv_len_batch = torch.stack(inv_len_batch) #batch x 1
        ast_traversal_batch = torch.stack(ast_traversal_batch) #batch x maxlen x 3
        ast_traversal_len_batch = torch.stack(ast_traversal_len_batch) #batch x 1
        utility_batch = torch.stack(utility_batch) #batch X max_util_len
        return (inv_token_batch,inv_len_batch), (ast_traversal_batch,ast_traversal_len_batch,utility_batch)


    def train_dataloader(self):
        dyn_sampler = DynamicSampler(self.training_data,self.batch_size,shuffle=False)
        return DataLoader(self.training_data, collate_fn= self.collate_fn, batch_sampler = dyn_sampler)

    def val_dataloader(self):
        dyn_sampler = DynamicSampler(self.validation_data,self.batch_size)
        return DataLoader(self.validation_data, collate_fn= self.collate_fn, batch_sampler = dyn_sampler)


    def test_dataloader(self):
        dyn_sampler = DynamicSampler(self.testing_data,self.batch_size)
        return DataLoader(self.testing_data, collate_fn= self.collate_fn, batch_sampler = dyn_sampler)


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch',type=int,default=10)
        parser.add_argument('--embedding_path',default = 'Data/WordVectors/mitten.2000')
        parser.add_argument('--data_base_address',default='Data/parallel_data')
        parser.add_argument('--run_base_address',default = 'run/')
        parser.add_argument('--split',default = '5')
        parser.add_argument('--documentation_path',default = 'Data/documentation/utility_descriptions.csv')
        parser.add_argument('--extra_data_path',nargs='*',default = [])
        parser.add_argument('--process_data', type = str2bool, default = False)
        parser.add_argument('--command_normalised',type = str2bool, default = True)
        parser.add_argument('--max_len',type=int,default=30)
        return parser
