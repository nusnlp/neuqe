import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ...modules import attention

class Predictor(nn.Module):
    """ RNN-based predictor model """
    def __init__(self, args):

        super(Predictor,self).__init__()
        self.debug = args.debug
        self.source_vocab_size = args.source_vocab_size
        self.target_vocab_size = args.target_vocab_size
        self.source_embed_size = args.num_source_embed
        self.target_embed_size = args.num_target_embed
        self.hidden_size = args.num_hidden_units
        self.maxout_size = args.num_maxout_units
        self.output_embed_size = args.num_output_embed

        # Model params

        # source embedding layer
        self.source_embedding = nn.Embedding(self.source_vocab_size, self.source_embed_size)
        # source bidirectional RNN
        self.source_birnn = nn.GRU(self.source_embed_size, self.hidden_size, bidirectional=True)
        # source v target attention
        self.forward_attn = attention.BahdanauAttention(self.hidden_size, encoder_state_size=2*self.hidden_size, debug=self.debug) #here attention will work on the bidirectional outputs
        self.reverse_attn = attention.BahdanauAttention(self.hidden_size, encoder_state_size=2*self.hidden_size, debug=self.debug) #here attention will work on the bidirectional outputs


        # target eembedding
        self.target_embedding = nn.Embedding(self.target_vocab_size, self.target_embed_size)

        # target bidirectional RNN
        self.target_forward_rnn = nn.GRU(self.target_embed_size + 2*self.hidden_size, self.hidden_size)
        self.target_reverse_rnn = nn.GRU(self.target_embed_size + 2*self.hidden_size, self.hidden_size)

        # projections for activation (maxout) input
        self.decoder_states_proj = nn.Linear(2*self.hidden_size, 2*self.maxout_size)        # S_0
        self.target_words_proj = nn.Linear(2*self.target_embed_size, 2*self.maxout_size)   # V_0
        self.context_proj = nn.Linear(2*self.hidden_size, 2*self.maxout_size)             # S_0

        # projection after activation (maxout) output
        self.final_proj = nn.Linear(self.maxout_size, self.output_embed_size, bias=False)   #W_0 = M x O

        # final output vocabulary projection
        self.out_vocab_proj = nn.Linear(self.output_embed_size, self.target_vocab_size, bias=False) #W_1 = O x V

    def forward(self, model_input):
        """ Forward pass for the predictor model

        Args:
            model_input: is of the form (source,source_mask, source_left, source_right_reversed) where,
        """
        source, source_mask, target, *_ = model_input

        #with torch.autograd.profiler.profile() as prof:
        # compute lengths of source sentences
        source_lengths = list(source_mask.data.sum(0)) # dim = B
        if self.debug == True: print("lengths:", source_lengths)

        # source embedding
        source_embedded = self.source_embedding(source) # TxB => TxBxD
        if self.debug == True: print("source embedded:", source_embedded.size())

        # target embedding
        target_embedded = self.target_embedding(target) # TxB => T x B x D
        if self.debug == True: print("target embedded:", target_embedded.size())

        # packing and unpacking for using padding with RNNs
        source_embedded_packed = nn.utils.rnn.pack_padded_sequence(source_embedded, source_lengths)
        encoder_states_packed, _ = self.source_birnn(source_embedded_packed) # TxBx2*H (bidirectional)
        encoder_states, encoder_state_lengths = nn.utils.rnn.pad_packed_sequence(encoder_states_packed)
        if self.debug == True: print("encoder_states:",encoder_states.size())

        # find the max length of the batch
        max_target_length = target.size(0)

        # run a bidirectional RNN from left and right of the target embeddings (without padding)
        n_batches = encoder_states.size(1)
        init_context = Variable(torch.cuda.FloatTensor(1, n_batches, 2*self.hidden_size).fill_(0), requires_grad=False) # context from bidirectional RNN

        # forward rnn
        rnn_in = torch.cat([target_embedded[0].unsqueeze(0),init_context], dim=-1)
        #rnn_in = target_embedded[0].unsqueeze(0)
        rnn_out, hidden = self.target_forward_rnn(rnn_in)
        decoder_forward_states = [rnn_out]
        source_forward_contexts = [init_context]
        for ti in range(1, max_target_length):
            attn_weights = self.forward_attn(rnn_out.squeeze(0), encoder_states, source_mask)  #Bx1xT
            context = attn_weights.bmm(encoder_states.transpose(0, 1)).transpose(0,1)  #Bx1xT * BxT*2H => Bx1x2H  => 1xBx2H
            if self.debug == True: print("context:",context.size())
            rnn_in = torch.cat([target_embedded[ti].unsqueeze(0), context], dim=-1)
            #rnn_in = target_embedded[ti].unsqueeze(0)
            rnn_out, hidden = self.target_forward_rnn(rnn_in, hidden)  # dim(out) = 1xBxH
            decoder_forward_states.append(rnn_out)
            source_forward_contexts.append(context)
        decoder_forward_states = torch.cat(decoder_forward_states, dim=0)
        source_forward_contexts = torch.cat(source_forward_contexts, dim=0)
        if self.debug == True: print("decoder_forward_states:",decoder_forward_states.size())
        if self.debug == True: print("source_forward_contexts:", source_forward_contexts.size())

        # backward rnn
        init_context = Variable(torch.cuda.FloatTensor(1, n_batches, 2*self.hidden_size).fill_(0), requires_grad=False) # context from bidirectional RNN
        rnn_in = torch.cat([target_embedded[max_target_length-1].unsqueeze(0),init_context], dim=-1)

        rnn_out, hidden = self.target_reverse_rnn(rnn_in)
        decoder_reverse_states = [rnn_out]
        source_reverse_contexts = [init_context]
        for ti in reversed(range(0, max_target_length-1)):
            attn_weights = self.reverse_attn(rnn_out.squeeze(0), encoder_states, source_mask)  #Bx1xT
            context = attn_weights.bmm(encoder_states.transpose(0, 1)).transpose(0,1)  #Bx1xT * BxT*2H => Bx1x2H  => 1xBx2H
            if self.debug == True: print("context:",context.size())

            # reverse RNN shares context shared from forward RNN
            rnn_in = torch.cat([target_embedded[ti].unsqueeze(0), context], dim=-1)
            rnn_out, hidden = self.target_reverse_rnn(rnn_in, hidden)
            decoder_reverse_states = [rnn_out] + decoder_reverse_states
            source_reverse_contexts = [context] + source_reverse_contexts
        decoder_reverse_states = torch.cat(decoder_reverse_states, dim=0)
        source_reverse_contexts = torch.cat(source_reverse_contexts, dim=0)
        if self.debug == True: print("decoder_reverse_states:",decoder_reverse_states.size())
        if self.debug == True: print("source_reverse_contexts:", source_reverse_contexts.size())

        decoder_states = torch.cat([decoder_forward_states, decoder_reverse_states], dim=-1)
        if self.debug == True: print("decoder_states:",decoder_states.size())

        # list of vocab outputs
        vocab_outputs = []
        preqvs = []

        # loop trhough 1 to max_target_length-1
        for ti in range(1,max_target_length-1):  # target = BxL
            # computing decoder right and left state
            decoder_left_state = torch.split(decoder_states[ti-1], self.hidden_size, dim=-1)[0] # BxH : extract forward RNN output only
            decoder_right_state = torch.split(decoder_states[ti+1], self.hidden_size, dim=-1)[1] # BxH : extract reverse RNN output only
            if self.debug == True: print("decoder_left_state:",decoder_left_state.size())
            if self.debug == True: print("decoder_right_state:",decoder_right_state.size())
            # concatenating right and left decoder state into single vector
            decoder_state = torch.cat((decoder_left_state,decoder_right_state),dim=-1) # 1xBx2H
            if self.debug == True: print("decoder_state:",decoder_state.size())

            # computing attention context vector
            source_context = source_forward_contexts[ti].unsqueeze(0) + source_reverse_contexts[ti].unsqueeze(0) #1xBx2H

            if self.debug == True: print("source context", source_context.size())

            # target word projections of prev word and next words
            prev_word_proj = target_embedded[ti-1]  # BxD
            next_word_proj = target_embedded[ti+1]  # BxD

            # combining prev word and next word projections into single vector
            near_words_proj = torch.cat((prev_word_proj,next_word_proj),dim=-1) # 1xBx2D
            if self.debug == True: print("near_words_proj:",near_words_proj.size())

            # combining all to generate output which is input
            # to the activation function (maxout in this case!)
            #t_j = S_0[d_l;d_r] + V_0[E_y y_j-1; E_y y_j+1] + C0cj
            act_input = self.decoder_states_proj(decoder_state) + self.target_words_proj(near_words_proj) + self.context_proj(source_context) #     dim(act_input) = B x 2M   M for maxout input
            #act_input = self.decoder_states_proj(decoder_state) + self.target_words_proj(near_words_proj)  #     dim(act_input) = B x 2M   M for maxout input

            # non-linearty using maxout
            act_output = F.max_pool1d(act_input,kernel_size=2,stride=2)   # dim(act_output) = B x M
            if self.debug == True: print("maxout output:",act_output.size())

            output_embedded = self.final_proj(act_output) # 1xBxM -> 1xBxO
            output_embedded = output_embedded.squeeze(0) # 1 x B x O => B x O
            if self.debug == True: print("sqz output:",output_embedded.size())

            # projecting to final output
            vocab_output = self.out_vocab_proj(output_embedded) # B x D => B x V
            if self.debug == True: print("vocab output:",vocab_output.size())

            vocab_outputs.append(vocab_output)
            preqvs.append(self.out_vocab_proj.weight[target[ti]].unsqueeze(0) *  output_embedded)

            # for profiling
            #if debug==True: print(prof)

        final_output = torch.cat(vocab_outputs, dim=0)
        if self.debug == True: print("final_output:",final_output.size())
        preqv_output = torch.cat(preqvs, dim=0)
        postqv_output = decoder_states[1:max_target_length-1]
        return final_output, preqv_output, postqv_output
