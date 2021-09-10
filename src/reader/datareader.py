# DataReader
import torch
from torch.utils.data import Dataset
from reader.vocab import Vocab
import numpy as np
import nltk
from main_utils import get_logger

from bashlint import data_tools, nast
from collections import deque
logger = get_logger()


class TrgTokenizer:
    def __init__(self):
        self.value_id = Vocab(reserve_tokens=False)
        self.kind_id = Vocab(reserve_tokens=False)
        self.kind_id.add_token('EC')
        self.kind_id.add_token('ET')
        self.value_id.add_token('dummy')
        # self.utility_mask = {}
    
    def update(self,kind,value_set):
        self.kind_id.add_token(kind)
        for value in value_set:
            self.value_id.add_token(value)
    
    def get_value_id(self,value):
        return self.value_id.get_id(value)
    
    def get_kind_id(self,kind):
        return self.kind_id.get_id(kind)
    
    def get_kind_token(self,kindid):
        return self.kind_id.get_token(kindid)
    
    def get_value_token(self,valueid):
        return self.value_id.get_token(valueid)
    
    @property
    def pad(self):
        return self.kind_id.pad

class SrcTokenizer:
    def __init__(self):
        self.text = Vocab()
        self.pos = Vocab()

    def text_get_token(self,tokid):
        return self.text.get_token(tokid)

    def text_get_id(self,tok):
        return self.text.get_id(tok)
    
    def pos_get_token(self,tokid):
        return self.pos.get_token(tokid)
    
    def pos_get_id(self,tok):
        return self.pos.get_id(tok)
    
    @property    
    def sos(self):
        return self.text.sos
    
    @property
    def eos(self):
        return self.text.eos
    
    @property
    def pad(self):
        return self.text.pad
    
    def build_dic(self,x,x_tag):
        self.text.build_dic(x)
        self.pos.build_dic(x_tag)

class DataReader(Dataset):
    def __init__(self,\
                paths,\
                max_len,\
                src_tokenizer=None,\
                trg_tokenizer = None,\
                extra_data_paths = []):
        super(DataReader,self).__init__()
        self.src_path,self.trg_path = paths
        self.max_src_length = max_len
        self.max_trg_length = -1
        self.max_trg_level = -1

        if src_tokenizer is None:
            self.src_tokenizer = SrcTokenizer()
        else:
            self.src_tokenizer = src_tokenizer
        
        if trg_tokenizer is None:
            self.trg_tokenizer = TrgTokenizer()
        else:
            self.trg_tokenizer = trg_tokenizer

        self.read_raw_data(extra_data_paths)


    def replace_args(self,node):
        if node.is_argument():
            node.value = node.arg_type
        for i in range(len(node.children)):
            node.children[i] = self.replace_args(node.children[i])
        return node


    def get_util_flags(self,root):
        queue = deque()
        queue.append(root)
        util_flag = {}
        while len(queue)>0:
            top = queue.popleft()
            if top.is_utility():
                if top.value not in util_flag:
                    util_flag[top.value] = top.get_flags()
                else:
                    util_flag[top.value].extend(top.get_flags())
            for child in top.children:
                queue.append(child)
        return util_flag


    def get_type_value(self,root):
        queue = deque()
        queue.append(root)
        kind_value = {}
        while len(queue)>0:
            top = queue.popleft()
            if top.kind not in kind_value:
                kind_value[top.kind] = set([top.value])
            else:
                kind_value[top.kind].add(top.value)
            for child in top.children:
                queue.append(child)
        return kind_value

    def read_raw_data(self,extra_data_paths):
        self.x = []
        self.x_tag = []
        self.x_len = []

        self.y_ast_traversal = []
        self.y_len = []
        self.y_utils = []

        #Read nlc2cmd data
        self.read_raw_data_(self.src_path,self.trg_path)
        #Read extra data
        for src_path,trg_path in extra_data_paths:
            self.read_raw_data_(src_path,trg_path)

        #Sanity checks
        assert len(self.x) == len(self.y_ast_traversal)
        assert len(self.y_ast_traversal) == len(self.y_len)
        assert len(self.x) == len(self.x_len)
        assert len(self.x) == len(self.x_tag)
        for src_sentence,trg_sentence in zip(self.x,self.y_ast_traversal):
            assert len(src_sentence) > 0
            assert len(trg_sentence) > 0
        #Post processing read data
        self.x_len = np.array(self.x_len)
        self.y_len = np.array(self.y_len)
        self.instance_len = self.x_len + self.y_len

        self.src_tokenizer.build_dic(self.x,self.x_tag)
        #log num tokens read and max length
        logger.info(f"Total types {self.trg_tokenizer.kind_id.vocab_len}")
        logger.info(f"Total values {self.trg_tokenizer.value_id.vocab_len}")
        logger.info(f"Max target length{self.max_trg_length} Max source length {self.max_src_length}")

    def read_raw_data_(self,src_path,trg_path):
        logger.info(f'Reading from {src_path} and {trg_path}')
        #Read target data
        with open(trg_path,'r') as f:
            for line in f:
                root = self.replace_args(data_tools.bash_parser(line.strip()))
                kind_value = self.get_type_value(root)
                self.y_utils.append(list(kind_value['utility']))
                for kind,value_set in kind_value.items():
                    self.trg_tokenizer.update(kind,value_set)
                self.y_ast_traversal.append(self.construct_level_traversal(root))
                self.y_len.append(len(line.strip().split()))
        
        #Read source data
        with open(src_path,'r') as f:
            for line in f:
                line_tokens = line.strip().split()
                self.x.append(line_tokens)
                self.x_tag.append([tag[1] for tag in nltk.pos_tag(line_tokens)])
                self.x_len.append(len(line_tokens))        

    def _cut_and_pad(self, tok_list, length, pad):
        final_length = min(length,len(tok_list))
        tok_list = tok_list[:length] + [pad] * (max(0,length -len(tok_list)))
        return tok_list, final_length

    def __len__(self):
        return len(self.y_len) 

    def construct_level_traversal(self,root):
        tokens = []
        queue = deque()
        dummy = self.trg_tokenizer.value_id.get_id('dummy')
        root.level = 0
        root.parent_idx = 0
        root.value = "dummy"
        queue.append(root)
        while len(queue) > 0:
            top = queue.popleft()
            top_kind = self.trg_tokenizer.get_kind_id(top.kind)
            top_parent_idx = top.parent_idx
            top_level = top.level
            if top.kind == 'EC':
                tokens.append([top_kind,dummy,top_parent_idx,top_level])
                continue
            top_value = self.trg_tokenizer.get_value_id(top.value)
            tokens.append([top_kind,top_value,top_parent_idx,top_level])
            cur_idx = len(tokens)-1

            for child in top.children:                
                child.parent_idx = cur_idx
                child.level = top_level + 1
                queue.append(child)

            child_end = nast.Node(kind = 'EC',value = -1)
            child_end.parent_idx = cur_idx
            child_end.level = top_level+1
            queue.append(child_end)
        
        tokens.append([self.trg_tokenizer.get_kind_id('ET'),dummy,len(tokens)-1,tokens[-1][3]+1]) #EOTree node -> child of last EC node
        
        for t in tokens:
            self.max_trg_level = max(self.max_trg_level,t[-1])

        if len(tokens) > self.max_trg_length:
            self.max_trg_length = len(tokens)
        
        return tokens


    def __getitem__(self, idx):
        ast_traversal = self.y_ast_traversal[idx]
        utilities = self.y_utils[idx]

        inv_len = self.x_len[idx]
        inv = self.x[idx]
        inv_tag = self.x_tag[idx]
        
        
        x_tokens = []
        for word in inv:
            token_id = self.src_tokenizer.text_get_id(word.strip())
            x_tokens.append(token_id)
        
        x_tag_tok = []
        for word in inv_tag:
            token_id = self.src_tokenizer.pos_get_id(word.strip())
            x_tag_tok.append(token_id)
        
        utility_tok = []
        for util in utilities:
            utility_tok.append(self.trg_tokenizer.get_value_id(util))
        
        x_tokens, ilen = self._cut_and_pad(x_tokens,self.max_src_length,self.src_tokenizer.pad)
        x_tag_tok, _ = self._cut_and_pad(x_tag_tok, self.max_src_length,self.src_tokenizer.pad)
        assert _ == ilen

        ast_traversal_len = len(ast_traversal)

        inv_tok = torch.LongTensor(x_tokens)
        inv_tag_tok = torch.LongTensor(x_tag_tok)
        inv_tok = torch.stack([inv_tok,inv_tag_tok],dim=-1)
        inv_len = torch.LongTensor([inv_len])
        ast_traversal = torch.LongTensor(ast_traversal)
        ast_traversal_len = torch.LongTensor([ast_traversal_len])
        utility_tensor = torch.LongTensor(utility_tok)

        return (inv_tok, inv_len), (ast_traversal, ast_traversal_len, utility_tensor)


class DynamicSampler(torch.utils.data.Sampler):
    # handles dynamic batching wrt tokens
    def __init__(self,dataset,batch_size,shuffle=False):
        super(DynamicSampler,self).__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        if shuffle:
            self.sorted_indices = np.arange(dataset.instance_len.shape[0])
            np.random.shuffle(self.sorted_indices)
        else:
            self.sorted_indices = np.argsort(dataset.instance_len)#[249:]
    
    def __iter__(self):
        current_batch = []
        current_len = 0
        for i,idx in enumerate(self.sorted_indices):
            current_len += 1
            current_batch.append(idx)
            #if this is the last element in dataset or including next element exceeds batch size return current batch
            if current_len >= self.batch_size or i==len(self.sorted_indices)-1:
                yield current_batch
                current_batch = []
                current_len = 0

