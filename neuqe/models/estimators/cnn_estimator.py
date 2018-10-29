import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CNNEstimator(nn.Module):
    """ Class for convolutional estimator model """

    def __init__(self, args, pred_model=None):
        """ Constructor for setting arguments and the predictor model"""
        super(CNNEstimator,self).__init__()
        self.debug = args.debug

        # to not train the predictor
        if args.update_predictor == False:
            for param in pred_model.parameters():
                param.requires_grad = False

        self.update_predictor_flag = args.update_predictor

        # setting arguments
        self.pred_model = pred_model
        self.qvectype = args.quality_vector_type
        self.kernel_width = 3

        # computing input size based on the input fromt the predictor
        # DIM(I) = O(noutembed) for pre, 2H for post, and O+2H for pre/post
        if self.qvectype == 'pre':
            self.input_size = self.pred_model.output_embed_size
        elif self.qvectype == 'post':
            self.input_size = self.pred_model.hidden_size * 2
        elif self.qvectype == 'prepost':
            self.input_size = self.pred_model.output_embed_size + self.pred_model.hidden_size * 2
        else:
            raise NotImplementedError

        self.hidden_size = args.num_hidden_units
        self.dropout_rate = args.dropout
        self.num_layers = 1

        #components
        self.convs = nn.ModuleList()
        self.attnpools = nn.ModuleList()
        self.input_proj = nn.Linear(self.input_size, self.hidden_size)
        pad = self.kernel_width // 2

        # setting convolutional layers
        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(Conv1d(in_channels=self.hidden_size, out_channels=self.hidden_size, dropout=self.dropout_rate, kernel_size= self.kernel_width,  padding=pad))

        # attention-based pooling layer
        self.attention_pool = AttentionPool(self.hidden_size)

        # dense layer
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)

        # single conv
        self.out_proj = nn.Linear(self.hidden_size, 1) # TxBxQ => TxBx1

    def forward(self, model_input):
        """ forward pass for the convolutional estimator model """

        if self.update_predictor_flag == False:
            self.pred_model.eval()

        # getting input
        source, source_mask, hyp, hyp_mask = model_input

        # getting output
        pred_model_input = (source, source_mask, hyp, hyp_mask)
        log_probs, preqvecs, postqvecs, *_ = self.pred_model(pred_model_input)

        if self.qvectype == 'pre':
            qvecs = preqvecs
        elif self.qvectype == 'post':
            qvecs = postqvecs
        elif self.qvectype == 'prepost':
            qvecs = torch.cat([preqvecs,postqvecs],dim=-1)
        else:
            raise NotImplementedError

        # removing first <s/> and last <pad> from hyp_mask
        hyp_mask = hyp_mask[1:hyp_mask.size(0)-1]

        # expanding hyp maxis across the last axis to 0-out all input dimensions which are to be masked
        hyp_mask_expanded = hyp_mask.unsqueeze(-1).expand(-1,-1,self.input_size)

        # applying mask on input
        qvecs.data.masked_fill_(hyp_mask_expanded.data.eq(0),0)

        if self.debug==True: print("Qvecs before conv:{}".format(qvecs.size()))
        F.dropout(qvecs, p=self.dropout_rate, training=self.training, inplace=True)

        # passing through convolutions
        convin = self.input_proj(qvecs)

        for conv in self.convs:
            # apply dropout
            F.dropout(convin, p=self.dropout_rate, training=self.training, inplace=True)
            # convolution operation
            convout =  conv(convin.permute(1,2,0))   # BxQxT => BxHxT (due to appropriate paddings)
            #convout = self.bn(convout)
            convout = convout.permute(2,0,1) # BxHxT => TxBxH
            if self.debug==True: print("Summary after conv:{}".format(convout.size()))
            # non linearty
            convout = ( F.relu(convout) + convin ) * math.sqrt(0.5)
            convin = convout

        # attention-based pooling
        summary = self.attention_pool(convout, hyp_mask)

        F.dropout(summary, p=self.dropout_rate, training=self.training, inplace=True)
        summary = F.relu(self.dense(summary))

        # final projection
        unnormalized_score = self.out_proj(summary) #Bx2H => Bx1

        final_score = F.sigmoid(unnormalized_score)
        return final_score, log_probs

class AttentionPool(nn.Module):
    """Attention-based pooling on set of input vectors. attention is computed by using a learnt vector"""
    def __init__(self, hidden_size):
        super(AttentionPool,self).__init__()
        self.hidden_size = hidden_size
        self.attention_v = nn.Linear(self.hidden_size, 1)

    def forward(self, input_vecs, input_mask):
        """ Function to compute self context vector from a set of input vectors

        Args:
            input_vecs: the input vectors used to compute attention (dim: TxBxH)
            input_mask: the input mask used for the input vectors (dim: TxB)

        Returns:
            summary vector which is the weighted sum of input vectors.
        """
        # compute unnormalized attention weights
        attn_energies = self.attention_v(input_vecs).squeeze(-1) # TxBxQ => TxBxH => TxBx1 => TxB

        # masking with -inf so that softmax will become 0
        attn_energies.data.masked_fill_(input_mask.data.eq(0), -float('inf')) # T x B

        # compute softmax of attention in the time dimension
        attn_weights = F.softmax(attn_energies, dim=0).transpose(0,1).unsqueeze(1)  # TxB => BxT => Bx1xT

        # multiply attention weights to the input vectors and sum it up
        summary = attn_weights.bmm(input_vecs.transpose(0, 1)).squeeze(1)

        # return the summary
        return summary


def Conv1d(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
    """Weight-normalized Conv1d layer"""

    conv = nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt(( (1.0 - dropout)) / (conv.kernel_size[0] * in_channels))
    conv.weight.data.normal_(mean=0, std=std)
    conv.bias.data.zero_()
    return nn.utils.weight_norm(conv)

