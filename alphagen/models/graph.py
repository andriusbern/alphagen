import sys
import torch
import torch.nn as nn
from alphagen.models.generator import Base
from alphagen.utils.utils import ScheduledOptim
from torch import optim
from torch.nn import functional as F
import utils
import time
import pandas as pd
from tqdm import tqdm
from layer import PositionalEmbedding, EncoderLayer, DecoderLayer, PositionalEncoding
from layer import tri_mask


class NaiveAlphaFoldEncoder(nn.Module):
    '''
    'Encoder' only consisting of a projection layer for controlling the dimensionality of
     AlphaFold's representations
    '''

    def __init__(self, d_emb=512):
        super().__init__()
        self.d_input = 384
        self.input_proj = nn.Linear(self.d_input, d_emb)

    def forward(self, x):
        return self.input_proj(x)


class AlphaFoldEncoder(NaiveAlphaFoldEncoder):
    '''
    A full transformer Encoder that uses AlphaFold's 'single' and 'structure_module' 
    representations as input
    '''

    def __init__(self, voc, dropout=0., d_emb=512, n_layers=6, n_head=8, d_inner=1024,
                 d_model=512, pad_idx=0, use_posit_emb=True):

        super().__init__()
        self.voc = voc
        self.d_emb = d_emb
        self.n_layers = 6
        self.n_head = n_head
        self.d_inner = d_inner
        self.d_model = d_model
        self.d_input = 384
        self.use_posit_emb = use_posit_emb
        
        if use_posit_emb:
            self.posit_emb = PositionalEmbedding(d_emb, max_len=voc.max_len)
        
        self.dropout = dropout
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_inner=d_inner, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, x, src_pad=None, src_atn=None):

        x = self.input_proj(x) 
        if self.use_posit_emb:
            x += self.posit_emb(x)

        for enc_layer in self.layer_stack:
            x = enc_layer(x, pad_mask=src_pad, attn_mask=src_atn)
        return x


class ProteinStringEncoder(nn.Module):
    '''
    Simple Protein string encoder
    '''

    def __init__(self, voc, dropout=0., d_emb=512, n_layers=6, n_head=8, d_inner=1024,
                 d_model=512, pad_idx=0, has_seg=False):

        super().__init__()
        self.voc = voc
        self.d_emb = d_emb
        self.n_layers = 6
        self.n_head = n_head
        self.d_inner = d_inner
        self.d_model = d_model

        self.token_emb = nn.Embedding(voc.size, d_emb, padding_idx=pad_idx)
        self.posit_emb = PositionalEmbedding(d_emb, max_len=voc.max_len)
        
        self.dropout = dropout
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_inner=d_inner, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, x, src_pad=None, src_atn=None):

        x = self.token_emb(x) + self.posit_emb(x)

        for enc_layer in self.layer_stack:
            x = enc_layer(x, pad_mask=src_pad, attn_mask=src_atn)
        return x


class MoleculeGraphDecoder(nn.Module):
    def __init__(self, d_model=512, n_head=8, d_inner=1024, n_layer=12):
        super(MoleculeGraphDecoder, self).__init__()
        self.n_layer = n_layer
        self.d_model = d_model
        self.n_head = n_head
        self.layer_stack = nn.ModuleList([DecoderLayer(self.d_model, self.n_head, d_inner=d_inner)
                                     for _ in range(self.n_layer)])

    def forward(self, x, encoder_out, trg_mask=None, src_mask=None, atn_mask=None):
        
        for decoder_layer in self.layer_stack:
            x = decoder_layer(x, encoder_out, trg_mask=trg_mask, src_mask=src_mask, atn_mask=atn_mask)
        return x



class TargetConditionedGraphGenerator(Base):
    def __init__(self, voc_trg, d_emb=512, d_model=512, n_head=8, d_inner=1024, n_layer=12, pad_idx=0):
        super(TargetConditionedGraphGenerator, self).__init__()

        self.voc_trg = voc_trg
        self.pad_idx = pad_idx
        self.d_model = d_model
        self.n_grows = voc_trg.max_len - voc_trg.n_frags - 1
        self.n_frags = voc_trg.n_frags + 1
        self.d_emb = d_emb

        # Encoder-Decoder
        self.encoder = ProteinStringEncoder(voc=voc_trg, d_emb=d_emb, d_model=d_model, n_head=n_head, d_inner=d_inner)
        self.decoder = MoleculeGraphDecoder(d_model=d_model, n_head=n_head, 
                                       d_inner=d_inner, n_layer=n_layer)

        # Recurrent
        self.rnn = nn.GRUCell(self.d_model, self.d_model)

        # Embeddings
        self.emb_word = nn.Embedding(voc_trg.size * 4, self.d_emb, padding_idx=pad_idx)   # quadruplet embedding ([40 * 4], is that because of [voc * bonds] or [voc * features])
        self.emb_atom = nn.Embedding(voc_trg.size, self.d_emb, padding_idx=pad_idx)       # Atom embedding 
        self.emb_loci = nn.Embedding(self.n_grows, self.d_emb)                            # Embedding of the position of quadruplet
        self.pos_encoding = PositionalEncoding(self.d_emb, max_len=self.n_grows*self.n_grows) # Pos encoding (why squared max_len?)


        # Vocabulary projection
        self.prj_atom = nn.Linear(d_emb, self.voc_trg.size) # Atom type head, 40 outs
        self.prj_bond = nn.Linear(d_model, 4)               # Bond type head (4 outs)
        self.prj_loci = nn.Linear(d_model, self.n_grows)    # Curr/prev index head, constrained by the max molecule size

        self.init_states()
        self.optim = ScheduledOptim(
            optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-9), 0.1, d_model)


    def forward(self, x, y, is_train=False):

        ## Encoder part goes here (might not need to transpose)
        encoder_out = self.encoder(x.transpose(0, 1))

        if is_train:
            """
            General notes:
                - word - bond + atom_type * 4 = 160 combinations 
                - features - [atom = 0, curr = 1, prev = 2, bond = 3]
                - Why is the RNN used at all? 
                    - Seems like it is used to decide the subsequent features: when you generate an atom, use that information to generate bond type, etc
                    - Still seems incredibly redundant and slow - will need tests
                - Why is the batch * seq reshaping happening? Probably fine
            """


            # Targets are offset by 1 index as in all Transformer Decoders
            y, y_hat = y[:, :-1, :], y[:, 1:, :]
            batch_size, seq_len, _ = y.shape
            decoder_mask = tri_mask(y[:, :, 0])

            # Generate embeddings for words and process through transformer layers
            embedded  = self.emb_word(y[:, :, 0] * 4 + y[:, :, 3]) # What does the '* 4' do? Scales _emb identifier
            embedded += self.pos_encoding(y[:, :, 1] * self.n_grows + y[:, :, 2]) # Positional encoding (seems to index curr+prev)

            z = self.decoder(x=embedded.transpose(0, 1),
                                       mem=encoder_out,
                                       attn_mask=decoder_mask)

            # Obtaining the atom type
            ##########################
            z = z.transpose(0, 1).reshape(batch_size * seq_len, -1)
            out_atom = self.prj_atom(z).log_softmax(dim=-1).view(batch_size, seq_len, -1) # then reshape back to [batch_size, len, probability] 

            # Gather along the [_, _, prob] dim
            # y_hat[:, :, 0].unsqueeze(2) means that only the first feature of all tokens is considered
            # Since this feature indicates the type of _emb, we are gathering the probabilities at those indices only (why not the whole tensor?)
            out_atom = out_atom.gather(2, y_hat[:, :, 0].unsqueeze(2))
            emb_atom = self.emb_atom(y_hat[:, :, 0]).reshape(batch_size * seq_len, -1) # Passing targets only through atom_embedding
            hidden = self.rnn(emb_atom, z)                                 # Creating a vector via the RNN (why?) using all the targets at feature 0

            # Bond
            out_bond = self.prj_bond(hidden).log_softmax(dim=-1).view(batch_size, seq_len, -1) # Project the generated vector, get logp
            out_bond = out_bond.gather(2, y_hat[:, :, 3].unsqueeze(2))                 # Again, gather the bond type probability at feature 3
            
            # Words
            emb_word = self.emb_word(y_hat[:, :, 3] + y_hat[:, :, 0] * 4)         # What does word mean exactly? Repeats in line 83
            emb_word = emb_word.reshape(batch_size * seq_len, -1)
            hidden = self.rnn(emb_word, hidden)                                     # Again, use RNN to embed both the embedded words and the _emb
            out_prev = self.prj_loci(hidden).log_softmax(dim=-1).view(batch_size, seq_len, -1) # Same procedure repeats for both prev and current
            out_prev = out_prev.gather(2, y_hat[:, :, 2].unsqueeze(2))

            curr = self.emb_loci(y_hat[:, :, 2]).reshape(batch_size * seq_len, -1)
            hidden = self.rnn(curr, hidden)
            out_curr = self.prj_loci(hidden).log_softmax(dim=-1).view(batch_size, seq_len, -1)
            out_curr = out_curr.gather(2, y_hat[:, :, 1].unsqueeze(2))

            out = [out_atom, out_curr, out_prev, out_bond]

        else:

            # Generation loop

            is_end = torch.zeros(len(y)).bool().to(y.device)
            exists = torch.zeros(len(y), self.n_grows, self.n_grows).long().to(y.device)
            vals_max = torch.zeros(len(y), self.n_grows).long().to(y.device)
            frg_ids = torch.zeros(len(y), self.n_grows).long().to(y.device)
            order = torch.LongTensor(range(len(y))).to(y.device)
            curr = torch.zeros(len(y)).long().to(y.device) - 1 # -1s
            blank = torch.LongTensor(len(y)).to(y.device).fill_(self.voc_trg.tk2ix['*'])
            single = torch.ones(len(y)).long().to(y.device)
            voc_mask = self.voc_trg.masks.to(y.device) # Need to figure out where voc_trg is declared

            # Iterative growing of the molecule (cryptic af)

            for step in range(1, self.n_grows):
                _emb = y[:, step, 0] # Get generated _emb
                num = (vals_max > 0).sum(dim=1)
                vals_max[order, num] = voc_mask[_emb]
                vals_rom = vals_max - exists.sum(dim=1)

                if is_end.all():
                    y[:, step, :] = 0
                    continue
                data = y[:, :step, :] # All tokens up to current step
                decoder_mask = tri_mask(data[:, :, 0])
                emb = self.emb_word(data[:, :, 0] * 4 + data[:, :, 3])
                emb += self.pos_encoding(data[:, :, 1] * self.n_grows + data[:, :, 2])
                dec = self.decoder(x=emb.transpose(0, 1), mem=encoder_out, attn_mask=decoder_mask) # Same as in the training loop, switch batch_size/n_tokens dims
                dec = dec[-1, :, :] # Last token only in all batches

                # Is batch_size dim treated differently during inference?
                grow = y[:, step, 4] == 0 # Index of the _emb to grow I guess (but this is the batch_size dim)
                mask = voc_mask.repeat(len(y), 1) < 0 # Mask out vocab?

                ## Everything here concerns masks and verification of validity
                if step <= 2: # First token after BOS I guess
                    mask[:, -1] = True 
                else:
                    judge = (vals_rom == 0) | (exists[order, curr, :] != 0) # vals_rom defined later as  # 
                    judge[order, curr] = True # Order is a indexing vector of len=N ([0,1,2,3,...,n]); curr is zeros of length N
                    judge = judge.all(dim=1) | (vals_rom[order, curr] == 0)
                    mask[judge, -1] = True
                mask[:, 1] = True
                mask[is_end, 1:] = True

                out_atom = self.prj_atom(dec).softmax(dim=-1) # Getting _emb probs
                _emb = out_atom.masked_fill(mask, 0).multinomial(1).view(-1) # Need to figure out what this does exactly
                y[grow, step, 0] = _emb[grow] # At _emb feature, either in batch_size 0 or 1 fill in the _emb value at index 0 or 1
                _emb = y[:, step, 0] # Get generated _emb
                

                # Equals to 1 if _emb is 0 (EOS) and grow is 1;;; 0 otherwise
                is_end |= (_emb == 0) & grow # Termination condition (in-place bitwise AND operation) - cool trick
                

                num = (vals_max > 0).sum(dim=1)
                vals_max[order, num] = voc_mask[_emb]
                vals_rom = vals_max - exists.sum(dim=1)
                

                bud = _emb != self.voc_trg.tk2ix['*'] # Check for the star which indicates a growing site
                curr += bud # Curr is -1s of len(batch_size), add 1 if _emb is not *
                curr[is_end] = 0 # curr at index 0/1
                y[:, step, 1] = curr # Index of current

                # Exists is [srclen, n_grows, n_grows], Order is a indexing vector of len=N ([0,1,2,3,...,n])
                exist = exists[order, curr, :] != 0 

                mask = torch.zeros(len(y), 4).bool().to(y.device)
                for i in range(1, 4):
                    judge = (vals_rom < i) | exist
                    judge[order, curr] = True
                    mask[:, i] = judge.all(dim=1) | (vals_rom[order, curr] < i)
                mask[:, 0] = False if step == 1 else True
                mask[is_end, 0] = False
                mask[is_end, 1:] = True

                # Same as in training
                atom_emb = self.emb_atom(_emb)
                dec = self.rnn(atom_emb, dec)
                out_bond = self.prj_bond(dec).softmax(dim=-1)
                bond = out_bond.masked_fill(mask, 0).multinomial(1).view(-1)
                y[grow, step, 3] = bond[grow]
                bond = y[:, step, 3]

                mask = (vals_max == 0) | exist | (vals_rom < bond.unsqueeze(-1))
                mask[order, curr] = True
                if step <= 2:
                    mask[:, 0] = False
                mask[is_end, 0] = False
                mask[is_end, 1:] = True
                word_emb = self.emb_word(_emb * 4 + bond)
                dec = self.rnn(word_emb, dec)
                prev_out = self.prj_loci(dec).softmax(dim=-1)
                prev = prev_out.masked_fill(mask, 0).multinomial(1).view(-1)
                y[grow, step, 2] = prev[grow]
                prev = y[:, step, 2]

                for i in range(len(y)):
                    if not grow[i]:
                        frg_ids[i, curr[i]] = y[i, step, -1]
                    elif bud[i]:
                        frg_ids[i, curr[i]] = frg_ids[i, prev[i]]
                    obj = frg_ids[i, curr[i]].clone()
                    ix = frg_ids[i, :] == frg_ids[i, prev[i]]
                    frg_ids[i, ix] = obj
                exists[order, curr, prev] = bond
                exists[order, prev, curr] = bond
                vals_rom = vals_max - exists.sum(dim=1)
                is_end |= (vals_rom == 0).all(dim=1)

            # The part of connecting
            y[:, -self.n_frags, 1:] = 0
            y[:, -self.n_frags, 0] = self.voc_trg.tk2ix['GO']
            is_end = torch.zeros(len(y)).bool().to(y.device)
            for step in range(self.n_grows + 1, self.voc_trg.max_len):
                data = y[:, :step, :]
                decoder_mask = tri_mask(data[:, :, 0])
                emb = self.emb_word(data[:, :, 3] + data[:, :, 0] * 4)
                emb += self.pos_encoding(data[:, :, 1] * self.n_grows + data[:, :, 2])
                dec = self.decoder(emb.transpose(0, 1), mem=encoder_out, attn_mask=decoder_mask)

                vals_rom = vals_max - exists.sum(dim=1)
                frgs_rom = torch.zeros(len(y), 8).long().to(y.device)
                for i in range(1, 8):
                    ix = frg_ids != i
                    rom = vals_rom.clone()
                    rom[ix] = 0
                    frgs_rom[:, i] = rom.sum(dim=1)
                is_end |= (vals_rom == 0).all(dim=1)
                is_end |= (frgs_rom != 0).sum(dim=1) <= 1
                mask = (vals_rom < 1) | (vals_max == 0)
                mask[is_end, 0] = False
                atom_emb = self.emb_word(blank * 4 + single)
                dec = self.rnn(atom_emb, dec[-1, :, :])
                out_prev = self.prj_loci(dec).softmax(dim=-1)
                prev = out_prev.masked_fill(mask, 0).multinomial(1).view(-1)

                same = frg_ids == frg_ids[order, prev].view(-1, 1)
                exist = exists[order, prev] != 0
                mask = (vals_rom < 1) | exist | (vals_max == 0) | same
                mask[is_end, 0] = False
                prev_emb = self.emb_loci(prev)
                dec = self.rnn(prev_emb, dec)
                out_curr = self.prj_loci(dec).softmax(dim=-1)
                curr = out_curr.masked_fill(mask, 0).multinomial(1).view(-1)

                y[:, step, 3] = single
                y[:, step, 2] = prev
                y[:, step, 1] = curr
                y[:, step, 0] = blank
                y[is_end, step, :] = 0

                for i in range(len(y)):
                    obj = frg_ids[i, curr[i]].clone()
                    ix = frg_ids[i, :] == frg_ids[i, prev[i]]
                    frg_ids[i, ix] = obj
                exists[order, y[:, step, 1], y[:, step, 2]] = y[:, step, 3]
                exists[order, y[:, step, 2], y[:, step, 1]] = y[:, step, 3]
            out = y
        return out

    def fit(self, train_loader, ind_loader, epochs=100, method=None, out=None):
        log = open(out + '.log', 'w')
        best = float('inf')
        net = nn.DataParallel(self, device_ids=utils.devices)
        t00 = time.time()
        for epoch in tqdm(range(epochs)):
            t0 = time.time()
            for i, y in enumerate(train_loader):
                y = y.to(utils.dev)
                self.optim.zero_grad()
                loss = net(y, is_train=True)
                loss_train = [round(-l.mean().item(), 3) for l in loss]
                loss = sum([-l.mean() for l in loss])
                loss.backward()
                self.optim.step()
                del loss
                
                if sum(loss_train) < best:
                    torch.save(self.state_dict(), out + '.pkg')
                    best = sum(loss_train)
            
            #t2 = time.time()
            #print('Epoch {} - Train time: {}'.format(epoch, int(t2-t0)))
            frags, smiles, scores = self.evaluate(ind_loader)
            loss_valid = [round(-l.mean().item(), 3) for l in net(y, is_train=True) for y in ind_loader] # temporary solution to get validation loss
            t1 = time.time()
            #print('Epoch {} - Eval time: {}'.format(epoch, int(t1-t2)))
            valid = scores.VALID.mean()
            dt = int(t1-t0)
            #print("Epoch: {} Train loss : {:.3f} Validation loss : {:.3f} Valid molecules: {:.3f} Time: {}\n" .format(epoch, sum(loss_train), sum(loss_valid), valid, dt))
            log.write("Epoch: {} Train loss : {:.3f} Validation loss : {:.3f} Valid molecules: {:.3f} Time: {}\n" .format(epoch, sum(loss_train), sum(loss_valid), valid, dt))
            for j, smile in enumerate(smiles):
                log.write('%s\t%s\n' % (frags[j], smile))
            log.flush()
            t0 = t1
            del loss_valid
        
        log.close()

    def evaluate(self, loader, repeat=1, method=None):
        net = nn.DataParallel(self, device_ids=utils.devices)
        frags, smiles = [], []
        #t0 = time.time()
        with torch.no_grad():
            for _ in range(repeat):
                for i, y in enumerate(loader):
                    y_hat = net(y.to(utils.dev)) 
                    f, s = self.voc_trg.decode(y_hat)
                    frags += f
                    smiles += s
        #print('Eval net time:', time.time()-t0)
        if method is None:
            scores = utils.Env.check_smiles(smiles, frags=frags)
            scores = pd.DataFrame(scores, columns=['VALID', 'DESIRE'])
        else:
            scores = method(smiles, frags=frags)
        #print('Eval env time:', time.time()-t0)
        return frags, smiles, scores


