import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

from ...modules import attention

class Predictor(nn.Module):
    """ Class for convolutional predictor model """

    def __init__(self, args):
        """ Constructor for the predictor model """
        super(Predictor,self).__init__()

        # setting hyper parameters
        self.debug = args.debug
        self.source_vocab_size = args.source_vocab_size
        self.target_vocab_size = args.target_vocab_size
        self.source_embed_size = args.num_source_embed
        self.target_embed_size = args.num_target_embed
        self.hidden_size = args.num_hidden_units
        self.maxout_size = args.num_maxout_units
        self.output_embed_size = args.num_output_embed
        self.source_kernel_width = args.source_kernel_width
        self.target_kernel_width = args.target_kernel_width
        self.num_source_layers = args.num_source_layers
        self.num_target_layers = args.num_target_layers
        self.max_positions = 1024

        # source embedding layer
        self.source_embedding = Embedding(self.source_vocab_size, self.source_embed_size)
        self.source_position_embedding = Embedding(self.max_positions, self.source_embed_size)

        # target embedding layer
        self.target_embedding = Embedding(self.target_vocab_size, self.target_embed_size)
        self.target_position_embedding = Embedding(self.max_positions, self.target_embed_size)

        # setting source projection and convolutions.
        pad = (self.source_kernel_width - 1 ) // 2
        self.source_conv_proj = nn.Linear(self.source_embed_size, self.hidden_size)
        self.source_convs = nn.ModuleList()
        for i in range(self.num_source_layers):
            self.source_convs.append(Conv1d(in_channels=self.hidden_size, out_channels=2*self.hidden_size,
                                          kernel_size=self.source_kernel_width, padding=pad))

        # linear layer to project encoder state to encoder embed size
        self.encoder_state_embed_proj = nn.Linear(self.hidden_size, self.source_embed_size)

        # setting forward convolutions and attention layers
        self.forward_target_conv_proj = nn.Linear(self.target_embed_size, self.hidden_size)
        self.forward_target_convs = nn.ModuleList()
        self.forward_target_attns = nn.ModuleList()
        for i in range(self.num_target_layers):
            pad = self.target_kernel_width - 1
            self.forward_target_convs.append(Conv1d(in_channels=self.hidden_size, out_channels=2*self.hidden_size,
                                          kernel_size=self.target_kernel_width, padding=pad))
            self.forward_target_attns.append(AttentionLayer(self.hidden_size, self.target_embed_size, self.source_embed_size, debug=self.debug))


        # setting backward convolutions and attention layers
        self.reverse_target_conv_proj = nn.Linear(self.target_embed_size, self.hidden_size)
        self.reverse_target_convs = nn.ModuleList()
        self.reverse_target_attns = nn.ModuleList()
        for i in range(self.num_target_layers):
            pad = self.target_kernel_width - 1
            self.reverse_target_convs.append(Conv1d(in_channels=self.hidden_size, out_channels=2*self.hidden_size,
                                          kernel_size=self.target_kernel_width, padding=pad))
            self.reverse_target_attns.append(AttentionLayer(self.hidden_size, self.target_embed_size, self.source_embed_size, debug=self.debug))


        # projections for activation (maxout) input
        self.decoder_states_proj = nn.Linear(2*self.hidden_size, 2*self.maxout_size)        # S_0
        self.target_words_proj = nn.Linear(2*self.target_embed_size, 2*self.maxout_size)   # V_0

        # projection after activation (maxout) output
        self.final_proj = nn.Linear(self.maxout_size, self.output_embed_size, bias=False)   #W_0 = M x O

        # final output vocabulary projection
        self.out_vocab_proj = nn.Linear(self.output_embed_size, self.target_vocab_size, bias=False) #W_1 = O x V


    def forward(self, model_input):
        """ Forward pass for the predictor model

        Args:
            model_input: is of the form (source,source_mask, source_left, target) where,
        """

        source, source_mask, target, target_mask = model_input

        # create matrix of source positions for position embeddings
        source_positions = Variable(torch.arange(1,self.max_positions)[:source.size(0)].view(-1,1).expand(-1,source.size(1)).long()).cuda()
        source_positions = source_positions * source_mask
        if self.debug == True: print("source_positions",type(source_positions))

        # create matrix of target positions for position embeddings
        target_positions = Variable(torch.arange(1,self.max_positions)[:target.size(0)].view(-1,1).expand(-1,target.size(1)).long()).cuda()
        target_positions = target_positions * target_mask
        if self.debug == True: print("target_positions",target_positions.size())

        # compute lengths of source sentences
        source_lengths = list(source_mask.data.sum(0)) # dim = B
        if self.debug == True: print("lengths:", source_lengths)

        # source embedding
        source_embedded = self.source_embedding(source) + self.source_position_embedding(source_positions)# TxB => TxBxD
        if self.debug == True: print("source embedded:", source_embedded.size())

        # target embedding
        target_embedded = self.target_embedding(target) + self.target_position_embedding(target_positions)# TxB => T x B x D
        if self.debug == True: print("target embedded:", target_embedded.size())

        # source convolutions
        source_proj = self.source_conv_proj(source_embedded)
        if self.debug == True: print("source_proj:", source_proj.size())
        conv_input = source_proj.permute(1,2,0) # B x H x T
        if self.debug == True: print("conv_input:", conv_input.size())
        for source_conv in self.source_convs:
            # permuting  TxBxD => BxDxT
            conv_output = source_conv(conv_input)  # B x 2H x T => B x 2H x T
            conv_output = F.glu(conv_output, dim=1)  # B x 2H x T = B x H x T
            # residual connection
            conv_output = ( conv_output + conv_input ) * math.sqrt(0.5)

        # apply masking on outputs
        encoder_states = conv_output.permute(2,0,1)
        source_mask_expanded = source_mask.unsqueeze(-1).expand(-1,-1,self.hidden_size)
        encoder_states.data.masked_fill_(source_mask_expanded.data.eq(0),0)
        if self.debug == True: print("encoder_states:",encoder_states.size())

        # for attention
        # compute attention score compuatation vectors by projecting encoder states (e)
        encoder_states_in = self.encoder_state_embed_proj(encoder_states) # T'xBxH' => T'xBxE'
        encoder_states_in = GradMultiply.apply(encoder_states_in, 1.0 / (2.0 * self.num_target_layers))

        # compute vectors on which attention weights are applied (e+s)
        encoder_vectors = ( encoder_states_in + source_embedded ) * math.sqrt(0.5) # T'xBxE' + T'xBxE' = T'xBxE'


        # find the max length of the batch
        max_target_length = target.size(0)

        # run CNNs from forward looking CNN and a backward looking CNN

        # decoder forward convolutions
        conv_input = self.forward_target_conv_proj(target_embedded)
        conv_input = conv_input.permute(1,2,0) # B x H x T
        for target_conv, target_attn in zip(self.forward_target_convs, self.forward_target_attns):
            # MISSING: dropout
            conv_output = target_conv(conv_input) #  B x H x T => B x 2H x T+(k-1)
            # removing future paddings (k-1)
            conv_output = conv_output[:,:,:-(self.target_kernel_width-1)] # B x 2H x T+(k-1) => B x H x T
            # applying non-linearty
            conv_output = F.glu(conv_output, dim=1) # B x 2H x T => B x H x T
            # attention
            context, attn_weights = target_attn(conv_output.permute(2,0,1), target_embedded, encoder_states_in, encoder_vectors, source_mask)
            # adding context vector and residual
            conv_output =  ( ( ( conv_output + context.permute(1,2,0) ) * math.sqrt(0.5) ) + conv_input ) * math.sqrt(0.5)

        # get back in original dimensions
        decoder_forward_states = conv_output.permute(2,0,1) # B x H x T => T x B x H
        if self.debug == True: print("decoder_forward_states:",decoder_forward_states.size())

        # decoder backward convolutions
        conv_input = self.reverse_target_conv_proj(target_embedded)
        conv_input = conv_input.permute(1,2,0) # B x H x T
        for target_conv, target_attn in zip(self.reverse_target_convs, self.reverse_target_attns):

            conv_output = target_conv(conv_input) #  B x H x T => B x 2H x T+(k-1)
            # removing first paddings (k-1)
            conv_output = conv_output[:,:,self.target_kernel_width-1:] # B x 2H x T+(k-1) => B x H x T
            # applying non-linearty
            conv_output = F.glu(conv_output, dim=1) # B x 2H x T => B x H x T
            # attention
            context, attn_weights = target_attn(conv_output.permute(2,0,1), target_embedded, encoder_states_in, encoder_vectors, source_mask)
            # adding context vector and residual
            conv_output =  ( ( ( conv_output + context.permute(1,2,0) ) * math.sqrt(0.5) ) + conv_input ) * math.sqrt(0.5)

        # projecting back to original dimensions
        decoder_reverse_states = conv_output.permute(2,0,1) # B x H x T => T x B x H
        if self.debug == True: print("decoder_reverse_states:",decoder_forward_states.size())

        decoder_states = torch.cat([decoder_forward_states, decoder_reverse_states], dim=-1)
        if self.debug == True: print("decoder_states:",decoder_states.size())

        # list of vocab outputs
        vocab_outputs = []
        preqvs = []

        # loop trhough 1 to max_target_length-1
        for ti in range(1,max_target_length-1):  # target = BxL

            # extract the forward decoder state (previous word)
            decoder_left_state = torch.split(decoder_states[ti-1], self.hidden_size, dim=-1)[0] # BxH : extract forward RNN output only
            if self.debug == True: print("decoder_left_state:",decoder_left_state.size())

            # extract the reverse decoder state (next word)
            decoder_right_state = torch.split(decoder_states[ti+1], self.hidden_size, dim=-1)[1] # BxH : extract reverse RNN output only
            if self.debug == True: print("decoder_right_state:",decoder_right_state.size())

            # concatenating right and left decoder state into single vector
            decoder_state = torch.cat((decoder_left_state,decoder_right_state),dim=-1).unsqueeze(0) # Bx2H
            if self.debug == True: print("decoder_state:",decoder_state.size())

            # target word projections of prev word and next words
            prev_word_proj = target_embedded[ti-1]  # BxD
            next_word_proj = target_embedded[ti+1]  # BxD

            # combining prev word and next word projections into single vector
            near_words_proj = torch.cat((prev_word_proj,next_word_proj),dim=-1).unsqueeze(0) # 1xBx2D
            if self.debug == True: print("near_words_proj:",near_words_proj.size())
            act_input = self.decoder_states_proj(decoder_state) + self.target_words_proj(near_words_proj) #     dim(act_input) = B x 2M   M for maxout input

            # non-linearty using maxout
            act_output = F.max_pool1d(act_input,kernel_size=2,stride=2)   # dim(act_output) = B x M
            if self.debug == True: print("act output:",act_output.size())

            # projection to output embedding space
            output_embedded = self.final_proj(act_output) # 1xBx2H -> 1xBxO
            output_embedded = output_embedded.squeeze(0) # 1 x B x O => B x O
            if self.debug == True: print("sqz output:",output_embedded.size())

            # projecting to final output
            vocab_output = self.out_vocab_proj(output_embedded) # B x D => B x V
            if self.debug == True: print("vocab output:",vocab_output.size())

            vocab_outputs.append(vocab_output)
            preqvs.append(self.out_vocab_proj.weight[target[ti]].unsqueeze(0) *  output_embedded)


        # concatenating all final output layers for each time step
        final_output = torch.cat(vocab_outputs, dim=0)
        if self.debug == True: print("final_output:",final_output.size())
        preqv_output = torch.cat(preqvs, dim=0)
        postqv_output = decoder_states[1:max_target_length-1]
        return final_output, preqv_output, postqv_output


class AttentionLayer(nn.Module):
    """ Attention layer """

    def __init__(self, hidden_size, target_embed_size, source_embed_size, debug=False):
        """computes attention at every conv layer between encoder outputs+source embeddings and the decoder outputs

        Args:
            hidden_size: size of decoder hidden layer
            target_embed_size: size of target embeddings
            source_embed_size: size of source embeddings
            context vectors and attention weights at every layer

        """
        super(AttentionLayer,self).__init__()

        self.hidden_size = hidden_size
        self.target_embed_size = target_embed_size
        self.source_embed_size = source_embed_size
        self.debug = debug

        # linear layer to map decoder state to decoder embed size
        self.decoder_state_embed_proj = nn.Linear(self.hidden_size, self.target_embed_size )

        # attention score computation
        self.attn = attention.BatchDotAttention(self.target_embed_size, encoder_state_size=self.source_embed_size, debug=self.debug)
        # linear layer to map context vector output to hidden size
        self.context_proj = nn.Linear(self.source_embed_size, self.hidden_size)

    def forward(self, decoder_states, target_embed, encoder_states_in, encoder_vectors, encoder_mask):
        """ Forward pass for the mutli step attention layer

        Args:
            decoder_states: decoder states (TxBxH)
            target_embed: target embeddings (size: TxBxE)
            encoder_states: final output vectors of the encoder (size: T'xBxH')
            source_embed: source embeddings (size: T'xBxE')
            encoder_mask: mask for encoder states (size: T'xB)
        """
        # compute keys by summing target hidden states and target embeddings (z)
        decoder_states_in = ( self.decoder_state_embed_proj(decoder_states) + target_embed ) * math.sqrt(0.5)# TxBxE + TxBxE = TxBxE

        # compute attention weights (no projection within attn if E==E')
        attn_weights = self.attn(decoder_states_in, encoder_states_in, encoder_mask)  # BxTxT'

        # multiply attention weights to compute context vector#
        context = attn_weights.bmm(encoder_vectors.transpose(0,1)).transpose(0,1) # BxTxT' * BxT'xE' => TxBxE'

        encoder_time_steps = encoder_states_in.size(0)
        context = context * (encoder_time_steps * math.sqrt(1.0/encoder_time_steps) )

        # project context vector to decoder size
        context = self.context_proj(context) # TxBxE' => TxBxH

        # return context vectors and attention weights
        return context, attn_weights

def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    """ Weight normalized embedding layer """
    embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    embedding.weight.data.normal_(0, 0.1)
    return embedding

def Conv1d(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
    """Weight-normalized Conv1d layer"""
    conv = nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (conv.kernel_size[0] * in_channels))
    conv.weight.data.normal_(mean=0, std=std)
    conv.bias.data.zero_()
    return nn.utils.weight_norm(conv)


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        ctx.mark_shared_storage((x, res))
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None