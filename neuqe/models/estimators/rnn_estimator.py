import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNEstimator(nn.Module):
    """ Class for recurrent estimator model """

    def __init__(self, args, pred_model=None):
        """ Constructor for setting arguments and the predictor model"""

        super(RNNEstimator,self).__init__()
        self.debug = args.debug
        # to not train the predictor
        if args.update_predictor == False:
            for param in pred_model.parameters():
                param.requires_grad = False

        self.update_predictor_flag = args.update_predictor
        self.pred_model = pred_model

        self.qvectype = args.quality_vector_type

        # computing input size based on the input from the predictor
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

        #components
        self.birnn = nn.GRU(self.input_size, self.hidden_size, bidirectional=True)
        self.out_proj = nn.Linear(2*self.hidden_size, 1) # TxBx2Q => TxBx1

    def forward(self, model_input):
        """ forward pass for the estimator model """

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

        # apply dropout
        F.dropout(qvecs, p=self.dropout_rate, training=self.training, inplace=True)

        # running birnn
        hidden_outs, final_hidden = self.birnn(qvecs)  # TxBxI => TxBx2Q (Q=hid. dim)

        # finding lengths of hypotheses within the batch
        hyp_lengths = hyp_mask.sum(0) # dim = B

       # getting backward summary
        backward_summary = torch.split(hidden_outs[0], self.hidden_size, dim=-1)[1] # dim = BxQ

        # getting forward summary
        last_idx_expanded = (hyp_lengths-1).view(1,-1,1).expand(-1,-1,2*self.hidden_size) # dim = 1xBx2Q
        forward_summary = hidden_outs.gather(dim=0,index=last_idx_expanded).split(self.hidden_size, dim=-1)[0].squeeze(0) # BxQ

        # combining summaries
        summary = torch.cat([backward_summary,forward_summary], dim=-1) #dim(summary) = Bx2Q

        # final projection
        unnormalized_score = self.out_proj(summary) #Bx2Q => Bx1
        final_score = F.sigmoid(unnormalized_score)

        return final_score, log_probs
