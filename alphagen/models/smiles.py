import random
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from .layer import PositionalEmbedding, DecoderLayer
from .layer import pad_mask, tri_mask

dev = torch.device('cuda')
devices = [0]

class GPT2Decoder(nn.Module):
    def __init__(self, voc, d_emb=384, d_model=512, n_head=16, d_inner=512, n_layer=12, pad_idx=0):
        super(GPT2Decoder, self).__init__()
        self.n_layer = n_layer
        self.d_emb = d_emb
        self.d_model = d_model
        self.n_head = n_head
        self.voc = voc
        self.pad_idx = pad_idx

        self.token_emb = nn.Embedding(voc.size, self.d_emb, padding_idx=pad_idx)
        self.posit_emb = PositionalEmbedding(self.d_emb, max_len=voc.max_len) #  + voc.max_len
        self.blocks = nn.ModuleList([DecoderLayer(self.d_emb, self.n_head, d_inner=d_inner) for _ in range(self.n_layer)])
        self.layer_norm = nn.LayerNorm(self.d_emb)
        self.word_prj = nn.Linear(self.d_emb, self.voc.size)
        
        kaiming_normal_(self.word_prj.weight, nonlinearity="linear")
        self.x_logit_scale = (d_model ** -0.5)
        

    def forward(self, x, mem, trg_mask=None, atn_mask=None):
        x = self.posit_emb(x) + self.token_emb(x)
        for block in self.blocks:
            x = block(x, mem=mem, trg_mask=trg_mask, atn_mask=atn_mask)
        proj = self.word_prj(x) # * self.x_logit_scale
        return proj, x


class AF2SmilesTransformer(nn.Module):
    """
    Generative transformer encoder-gpt2 model that is conditioned for a specific target using 
    AlphaFold2 embeddings
    """
    def __init__(self, voc_trg, d_emb=384, d_model=384, n_head=16, d_inner=384, n_layer=6, pad_idx=0):
        super(AF2SmilesTransformer, self).__init__()
        self.voc_trg = voc_trg
        self.pad_idx = pad_idx
        self.gpt2 = GPT2Decoder(self.voc_trg, d_emb=d_emb, d_model=d_model,
                               n_head=n_head, d_inner=d_inner, n_layer=n_layer,
                               pad_idx=pad_idx)

        self.x_logit_scale = (d_model ** -0.5)
        self.pchembl_proj = nn.Linear(d_emb, 1)
        self.pchembl_proj_loss = torch.nn.MSELoss()
        self.loss_coefficients = {}
        self.init_states()
        
    def init_states(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            self.to(dev)

    def forward(self, x, mem, train=False, pchembl_targets=None, calc_pchembl=False):
        batch_size = len(x)
        z = torch.zeros(batch_size, 1).long().to(dev)
        # Generation
        if not train:
            loss = 0
            src = x
            seq_len = self.voc_trg.max_len # + self.voc_trg.max_len
            out = torch.zeros(len(src), seq_len).long().to(dev)
            out[:, :src.size(1)] = src
            is_end = torch.zeros(len(src)).bool().to(dev)
            for step in range(self.voc_trg.max_len-1): 
                input = out[:, :src.size(1)+step]
                key_mask = pad_mask(input, self.pad_idx)
                atn_mask = tri_mask(input)
                dec, z = self.gpt2(input.transpose(0, 1), mem=mem.transpose(0,1), trg_mask=key_mask, atn_mask=atn_mask)
                token = dec.softmax(dim=-1)[-1, :, :].multinomial(1).view(-1)
                token[is_end] = self.voc_trg.tk2ix['_']
                is_end |= token == self.voc_trg.tk2ix['EOS']
                out[:, src.size(1)+step] = token
                if is_end.all(): 
                    break
            trg = out
        
        # Training
        else:
            init_tokens = torch.LongTensor([[self.voc_trg.tk2ix['GO']]] * batch_size).to(dev)
            trg = x
            input = torch.cat([init_tokens, x[:, :-1]], dim=1)
            key_mask = pad_mask(input, self.pad_idx)
            atn_mask = tri_mask(input)
            proj, z = self.gpt2(input.transpose(0, 1), mem=mem.transpose(0,1), trg_mask=key_mask, atn_mask=atn_mask)
            out = proj.transpose(0, 1).log_softmax(dim=-1)
            loss = out.gather(2, trg.unsqueeze(2)).squeeze(2)
            loss = sum([-l.mean() for l in loss])

        outputs, losses = {}, {}
        
        ###################
        # Code for property prediction heads + losses
        if calc_pchembl:
            idx = (trg == self.voc_trg.tk2ix['EOS']).nonzero(as_tuple=False)
            z = z.transpose(0, 1)
            EOS_latent = z[idx[:, 0], idx[:, 1], :] # Pre-projections of all EOS tokens
            pchembl_pred = self.pchembl_proj(EOS_latent).squeeze()
            outputs['pchembl'] = pchembl_pred
            
            if pchembl_targets is not None:
                p_loss = self.pchembl_proj_loss(pchembl_pred, pchembl_targets) * .2
                loss += p_loss.item()
                losses['p_chembl'] = p_loss

        outputs['tokens'] = out
        # losses['total'] = loss
            
        return outputs, loss
