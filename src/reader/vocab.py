#Vocab
from collections import Counter
class Vocab:
    # 0,1,2,3 reserved for pad,sos,eos,unk
    def __init__(self,tokenized_sentences=None,reserve_tokens = True):
        self.stoi = {}
        self.itos = {}
        self.token_counter = Counter()

        self.unknown_tok = '[UNK]'
        self.unknown_id = 0
        self.pad_tok = '[PAD]'
        self.pad = 1   
        self.vocab_len = 0
        self.add_token(self.unknown_tok,self.unknown_id)
        self.add_token(self.pad_tok,self.pad)
        

        if reserve_tokens:
            self.sos = 2
            self.eos = 3
            
            self.sos_tok = '[SOS]'
            self.eos_tok = '[EOS]'

            self.add_token(self.pad_tok,self.pad)
            self.add_token(self.sos_tok,self.sos)
            self.add_token(self.eos_tok,self.eos)

        if tokenized_sentences is not None:
            self.build_dic(tokenized_sentences)

    def get_id(self,token):
        return self.stoi.get(token,self.unknown_id)

    def get_token(self,id):
        return self.itos.get(id,'[UNK]')
    
    def token_exists(self,token):
        return token in self.stoi.keys()
    
    def id_exists(self,id):
        return id in self.itos.keys()
    
    def add_token(self,token,id=None):
        self.token_counter.update([token])
        if self.token_exists(token) or self.id_exists(id):
            return
        if id is None:
            id = self.vocab_len
        self.stoi[token] = id
        self.itos[id] = token
        while self.id_exists(self.vocab_len):
            self.vocab_len += 1
        assert len(self.stoi) == len(self.itos)
        assert self.vocab_len == len(self.stoi), str(self.vocab_len)+ '!!!' + str(len(self.stoi))
        assert self.vocab_len == len(self.itos)

    def build_dic(self, tokenized_sentences):
        for sentence in tokenized_sentences:
            for token in sentence:
                self.add_token(token.strip())
    
    def _build_inv_trg_dic(self):
        for k,v in self.stoi.items():
            self.itos[v]=k