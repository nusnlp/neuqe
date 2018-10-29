import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    """ Attention module based on Bahdanau et al. (2015)"""
    # code adapted from: https://github.com/AuCson/PyTorch-Batch-Attention-Seq2seq/blob/master/attentionRNN.py (modification of the code to support maskng)
    def __init__(self, hidden_size, encoder_state_size=None, debug=False):
        super(BahdanauAttention, self).__init__()
        self.debug = debug
        self.hidden_size = hidden_size
        if encoder_state_size == None:
            self.encoder_state_size = self.hidden_size
        else:
            self.encoder_state_size = encoder_state_size
        self.attn = nn.Linear(self.hidden_size + self.encoder_state_size, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)
        # end of update
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hidden, encoder_outputs, encoder_outputs_mask):
        """ Forward pass of the attention module.

        Args:
            hidden: previous hidden state of the decoder, in shape (layers*directions,B,H)
            encoder_outputs: encoder outputs from Encoder, in shape (T,B,H')

        Returns:
            attention energies in shape (B,T)
        """
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        H = hidden.expand(max_len,-1,-1).transpose(0,1)  # repeat uses up memory
        encoder_outputs = encoder_outputs.transpose(0,1) # [B*T*H]
        attn_energies = self.score(H,encoder_outputs) # compute attention scores [B*T]
        if self.debug==True: print("attn_energies:", attn_energies.size())
        if self.debug==True: print("encoder_outputs_mask:", encoder_outputs_mask.data.eq(0).size())

        # inverse mask 1 for 0, and 0 for 1, and fill with -inf.
        # fill the attn_energies with -inf where it was originally 0
        # this will make softmax components form those positions to be useless
        attn_energies.data.masked_fill_(encoder_outputs_mask.transpose(1,0).data.eq(0), -float('inf'))
        if self.debug==True: print("attn_energies", attn_energies.size())

        # perform softmax with the updated attention energies
        attn_scores = self.softmax(attn_energies).unsqueeze(1) # [B*T -> B*1*T]normalize with softmax
        if self.debug==True: print("soft_max:", attn_scores.size())

        return attn_scores

    def score(self, hidden, encoder_outputs):
        """ find attention energies """
        energy = self.attn(torch.cat([hidden, encoder_outputs], 2)) # [B*T*(H+H')]->[B*T*H]
        energy = F.tanh(energy.transpose(2,1)) # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) #[B*1*H]
        energy = torch.bmm(v,energy) # [B*1*T]
        return energy.squeeze(1) #[B*T]


class DotAttention(nn.Module):
    """ Dot-product based attention """
    def __init__(self, hidden_size, encoder_state_size=None, debug=False):
        super(DotAttention, self).__init__()
        self.debug = debug
        self.hidden_size = hidden_size
        if encoder_state_size == None:
            self.encoder_state_size = self.hidden_size
        else:
            self.encoder_state_size = encoder_state_size
        self.hidden_projection = nn.Linear(self.encoder_state_size, self.hidden_size)
        # end of update
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hidden, encoder_outputs, encoder_outputs_mask):
        """ forward pass for attention module

        Args:
            hidden: previous hidden state of the decoder, in shape (B,H)
            encoder_outputs: encoder outputs from Encoder, in shape (T,B,H')

        Returns:
            attention energies in shape (B,T)
        """

        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1)  # BxH => Bx1xH
        if self.debug==True: print("hidden(after unsqueeze)", hidden.size())

        encoder_outputs_proj = self.hidden_projection(encoder_outputs)  # BxTxH' => BxTxH
        encoder_outputs_proj = encoder_outputs_proj.permute(1,2,0) # TxBxH => BxHxT
        if self.debug==True: print("encoder_outputs_proj (for attn, after permute)", encoder_outputs_proj.size())

        attn_energies = hidden.bmm(encoder_outputs_proj).squeeze(1) # Bx1xH * BxHxT => Bx1xT => BxT
        if self.debug==True: print("attn_energies:", attn_energies.size())
        if self.debug==True: print("encoder_outputs_mask:", encoder_outputs_mask.data.eq(0).size())

        # inverse mask 1 for 0, and 0 for 1, and fill with -inf.
        # fill the attn_energies with -inf where it was originally 0
        # this will make softmax components form those positions to be useless
        attn_energies.data.masked_fill_(encoder_outputs_mask.transpose(1,0).data.eq(0), -float('inf'))
        if self.debug==True: print("attn_energies", attn_energies.size())

        # perform softmax with the updated attention energies
        attn_scores = self.softmax(attn_energies).unsqueeze(1) # [B*T -> B*1*T]normalize with softmax
        if self.debug==True: print("soft_max:", attn_scores.size())

        return attn_scores


class BatchDotAttention(nn.Module):
    """ Class for finding dot product based attention over a batch efficiently """
    def __init__(self, hidden_size, encoder_state_size=None, debug=False):
        # performs dot attention on T_e encoder outputs with T_d decoder outputs
        super(BatchDotAttention, self).__init__()
        self.debug = debug
        self.hidden_size = hidden_size
        if encoder_state_size == None:
            self.encoder_state_size = self.hidden_size
        else:
            self.encoder_state_size = encoder_state_size

        self.hidden_projection = None
        # project only if sizes mismatch
        if hidden_size != encoder_state_size:
            self.hidden_projection = nn.Linear(self.encoder_state_size, self.hidden_size)

        # end of update
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hidden, encoder_outputs, encoder_outputs_mask):
        """ Forward pass for the attenion module

        Args:
            hidden: previous hidden state of the decoder, in shape (T,B,H)
            encoder_outputs: encoder outputs from Encoder, in shape (T',B,H')
            encoder_outputs_mask: mask of encoder outputs in shape (T',B)

        Returns:
            attention energies in shape (B,T)
        """

        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        hidden = hidden.transpose(0,1)  # T,BxH => BxTxH
        if self.debug==True: print("hidden(after unsqueeze)", hidden.size())

        if self.hidden_projection:
            encoder_outputs = self.hidden_projection(encoder_outputs)  #T'xBxH' => T'xBxH

        encoder_outputs = encoder_outputs.permute(1,2,0) # T'xBxH => BxHxT'
        if self.debug==True: print("encoder_outputs_proj (for attn, after permute)", encoder_outputs.size())

        attn_energies = hidden.bmm(encoder_outputs) # BxTxH * BxHxT' => BxTxT'
        if self.debug==True: print("attn_energies:", attn_energies.size())
        if self.debug==True: print("encoder_outputs_mask:", encoder_outputs_mask.data.eq(0).size())

        # inverse mask 1 for 0, and 0 for 1, and fill with -inf.
        # fill the attn_energies with -inf where it was originally 0
        # this will make softmax components form those positions to be useless

        # T' x B => B x T' => B x 1 x T' => B x T x T'
        encoder_outputs_mask = encoder_outputs_mask.transpose(1,0).unsqueeze(1).expand(-1,hidden.size(1),-1)

        attn_energies.data.masked_fill_(encoder_outputs_mask.data.eq(0), -float('inf'))
        if self.debug==True: print("attn_energies", attn_energies.size())

        # perform softmax with the updated attention energies
        attn_scores = self.softmax(attn_energies) # [BxTxT']
        if self.debug==True: print("soft_max:", attn_scores.size())

        return attn_scores
