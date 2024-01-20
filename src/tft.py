import math
from typing import Dict, List, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeDistributed(nn.Module):
    def __init__(self, module: nn.Module, batch_first: bool = False):
        super().__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        return y


class TimeDistributedInterpolation(nn.Module):
    def __init__(self, output_size: int, batch_first: bool = False, trainable: bool = False):
        super().__init__()
        self.output_size = output_size
        self.batch_first = batch_first
        self.trainable = trainable
        if self.trainable:
            self.mask = nn.Parameter(torch.zeros(self.output_size, dtype=torch.float32))
            self.gate = nn.Sigmoid()

    def interpolate(self, x):
        upsampled = F.interpolate(x.unsqueeze(1), self.output_size, mode="linear", align_corners=True).squeeze(1)
        if self.trainable:
            upsampled = upsampled * self.gate(self.mask.unsqueeze(0)) * 2.0
        return upsampled

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.interpolate(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.interpolate(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit"""

    def __init__(self, input_size: int, hidden_size: int = None, dropout: float = None):
        super().__init__()

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = dropout
        self.hidden_size = hidden_size or input_size
        self.fc = nn.Linear(input_size, self.hidden_size * 2)

        self.init_weights()

        
    def init_weights(self):
        for n, p in self.named_parameters():
            if "bias" in n:
                torch.nn.init.zeros_(p)
            elif "fc" in n:
                torch.nn.init.xavier_uniform_(p)

                
    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)
        x = F.glu(x, dim=-1)
        return x    
    

class ResampleNorm(nn.Module):
    def __init__(self, input_size: int, output_size: int = None, trainable_add: bool = True):
        super().__init__()

        self.input_size = input_size
        self.trainable_add = trainable_add
        self.output_size = output_size or input_size

        if self.input_size != self.output_size:
            self.resample = TimeDistributedInterpolation(self.output_size, batch_first=True, trainable=False)

        if self.trainable_add:
            self.mask = nn.Parameter(torch.zeros(self.output_size, dtype=torch.float))
            self.gate = nn.Sigmoid()
        self.norm = nn.LayerNorm(self.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_size != self.output_size:
            x = self.resample(x)

        if self.trainable_add:
            x = x * self.gate(self.mask) * 2.0
        output = self.norm(x)
        return output
    
    
class AddNorm(nn.Module):
    def __init__(self, input_size: int, skip_size: int = None, trainable_add: bool = True):
        super().__init__()

        self.input_size = input_size
        self.trainable_add = trainable_add
        self.skip_size = skip_size or input_size
        
        assert self.input_size == self.skip_size
        
        if self.input_size != self.skip_size:
            self.resample = TimeDistributedInterpolation(self.input_size, batch_first=True, trainable=False)

        if self.trainable_add:
            self.mask = nn.Parameter(torch.zeros(self.input_size, dtype=torch.float))
            self.gate = nn.Sigmoid()
        self.norm = nn.LayerNorm(self.input_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        if self.input_size != self.skip_size:
            skip = self.resample(skip)

        if self.trainable_add:
            skip = skip * self.gate(self.mask) * 2.0

        output = self.norm(x + skip)
        return output
    

class GateAddNorm(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = None,
        skip_size: int = None,
        trainable_add: bool = False,
        dropout: float = None,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size or input_size
        self.skip_size = skip_size or self.hidden_size
        self.dropout = dropout

        self.glu = GatedLinearUnit(self.input_size, hidden_size=self.hidden_size, dropout=self.dropout)
        self.add_norm = AddNorm(self.hidden_size, skip_size=self.skip_size, trainable_add=trainable_add)

    def forward(self, x, skip):
        output = self.glu(x)
        output = self.add_norm(output, skip)
        return output
    
    
class GatedResidualNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.1,
        context_size: int = None,
        residual: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.residual = residual

        if self.input_size != self.output_size and not self.residual:
            residual_size = self.input_size
        else:
            residual_size = self.output_size

        if self.output_size != residual_size:
            self.resample_norm = ResampleNorm(residual_size, self.output_size)

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.elu = nn.ELU()

        if self.context_size is not None:
            self.context = nn.Linear(self.context_size, self.hidden_size, bias=False)

        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.init_weights()

        self.gate_norm = GateAddNorm(
            input_size=self.hidden_size,
            skip_size=self.output_size,
            hidden_size=self.output_size,
            dropout=self.dropout,
            trainable_add=False,
        )

    def init_weights(self):
        for name, p in self.named_parameters():
            if "bias" in name:
                torch.nn.init.zeros_(p)
            elif "fc1" in name or "fc2" in name:
                torch.nn.init.kaiming_normal_(p, a=0, mode="fan_in", nonlinearity="leaky_relu")
            elif "context" in name:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x, context=None, residual=None):
        if residual is None:
            residual = x

        if self.input_size != self.output_size and not self.residual:
            residual = self.resample_norm(residual)

        x = self.fc1(x)
        if context is not None:
            context = self.context(context)
            x = x + context
        x = self.elu(x)
        x = self.fc2(x)
        x = self.gate_norm(x, residual)
        return x




class VariableSelectionNetwork(nn.Module):
    def __init__(
        self,
        cont_input_sizes: List[int],
        cat_input_sizes: List[int],
        hidden_size: int,
        dropout: float = 0.1,
        
    ):
        """
        Calcualte weights for ``num_inputs`` variables  which are each of size ``input_size``
        """
        super().__init__()
        
        
        self.hidden_size = hidden_size
        self.cont_input_sizes = cont_input_sizes
        self.cat_input_sizes = cat_input_sizes
        
        self.input_sizes = self.cont_input_sizes + self.cat_input_sizes
        
        self.dropout = dropout
        
        input_size_total = sum(cont_input_sizes+cat_input_sizes)
        
        num_inputs = len(cont_input_sizes+cat_input_sizes)
        
        self.flattened_grn = GatedResidualNetwork(
                    input_size_total,
                    min(self.hidden_size, num_inputs),
                    num_inputs,
                    self.dropout,
                    residual=False,
                )
                

        self.single_variable_grns = nn.ModuleList()
        
        for input_size in self.input_sizes:
            
            self.single_variable_grns.append(GatedResidualNetwork(
                input_size,
                min(input_size, self.hidden_size),
                output_size=self.hidden_size,
                dropout=self.dropout,
            ))
            


        self.softmax = nn.Softmax(dim=-1)
        

    
    def forward(self, x: List[torch.Tensor],
                x_cat: List[torch.Tensor]):
        
        # transform single variables
        var_outputs = []
        weight_inputs = []
        for j, elem in enumerate(x+x_cat):
            # select embedding belonging to a single input
            
            weight_inputs.append(elem)
            var_outputs.append(self.single_variable_grns[j](elem))
        var_outputs = torch.stack(var_outputs, dim=-1)

        # calculate variable weights
        flat_embedding = torch.cat(weight_inputs, dim=-1)
        sparse_weights = self.flattened_grn(flat_embedding, context=None)
        sparse_weights = self.softmax(sparse_weights).unsqueeze(-2)

        outputs = var_outputs * sparse_weights
        outputs = outputs.sum(dim=-1)
        
        return outputs, sparse_weights


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = None, scale: bool = True):
        super(ScaledDotProductAttention, self).__init__()
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = dropout
        self.softmax = nn.Softmax(dim=2)
        self.scale = scale

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.permute(0, 2, 1))  # query-key overlap

        if self.scale:
            dimension = torch.as_tensor(k.size(-1), dtype=attn.dtype, device=attn.device).sqrt()
            attn = attn / dimension

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)
        attn = self.softmax(attn)

        if self.dropout is not None:
            attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn

    
class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, d_model: int, dropout: float = 0.0):
        super(InterpretableMultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = self.d_q = self.d_v = d_model // n_head
        self.dropout = nn.Dropout(p=dropout)

        self.v_layer = nn.Linear(self.d_model, self.d_v)
        self.q_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_q) for _ in range(self.n_head)])
        self.k_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_k) for _ in range(self.n_head)])
        self.attention = ScaledDotProductAttention()
        self.w_h = nn.Linear(self.d_v, self.d_model, bias=False)

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if "bias" not in name:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.zeros_(p)

    def forward(self, q, k, v, mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        heads = []
        attns = []
        vs = self.v_layer(v)
        
        for i in range(self.n_head):
            qs = self.q_layers[i](q)
            ks = self.k_layers[i](k)
            head, attn = self.attention(qs, ks, vs, mask)
            head_dropout = self.dropout(head)
            heads.append(head_dropout)
            attns.append(attn)

        head = torch.stack(heads, dim=2) if self.n_head > 1 else heads[0]
        attn = torch.stack(attns, dim=2)

        outputs = torch.mean(head, dim=2) if self.n_head > 1 else head
        outputs = self.w_h(outputs)
        outputs = self.dropout(outputs)

        return outputs, attn    
    

class TFT(nn.Module):
    def __init__(self, input_dim, output_dim, timesteps, cat_info=None, emb_dim=16, prescale_dim=16, 
                 hidden_dim=32, n_heads=4, dropout=0.1, device='cuda'):
        super().__init__()
        
        cat_info = {} if cat_info is None else cat_info
        
        if cat_info:
            self.embeddings_enc = nn.ModuleList([nn.Embedding(nunique, emb_dim) for nunique in cat_info.values()])
            self.embeddings_dec = nn.ModuleList([nn.Embedding(nunique, emb_dim) for nunique in cat_info.values()])
        else:
            self.embeddings_enc = []
            self.embeddings_dec = []
            
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.n_heads = n_heads
        self.output_dim = output_dim

        self.prescalers_enc = nn.ModuleList([nn.Linear(1, prescale_dim) for _ in range(input_dim)])
        self.prescalers_dec = nn.ModuleList([nn.Linear(1, prescale_dim) for _ in range(input_dim)])
        
        self.vsn_enc = VariableSelectionNetwork([prescale_dim for _ in range(input_dim)], 
                                                [emb_dim for _ in range(len(cat_info.keys()))],
                                                hidden_dim,
                                                dropout=dropout,
                                                )
        self.vsn_dec = VariableSelectionNetwork([prescale_dim for _ in range(input_dim)], 
                                                [emb_dim for _ in range(len(cat_info.keys()))],
                                                hidden_dim,
                                                dropout=dropout,
                                                )
        
        
        self.lstm_encoder = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            
            batch_first=True,
        )

        self.lstm_decoder = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            
            batch_first=True,
        )
        
        self.post_lstm_gate_encoder = GatedLinearUnit(self.hidden_dim, dropout=self.dropout)
        self.post_lstm_gate_decoder = self.post_lstm_gate_encoder #shared
        
        self.post_lstm_add_norm_encoder = AddNorm(self.hidden_dim, trainable_add=False)
        self.post_lstm_add_norm_decoder = self.post_lstm_add_norm_encoder #shared
        
        
        self.multihead_attn = InterpretableMultiHeadAttention(
            d_model=self.hidden_dim, n_head=self.n_heads, dropout=self.dropout
        )
        
        self.post_attn_gate_norm = GateAddNorm(
            self.hidden_dim, dropout=self.dropout, trainable_add=False
        )
        
        self.pos_wise_ff = GatedResidualNetwork(
            self.hidden_dim, self.hidden_dim, self.hidden_dim, dropout=self.dropout
        )

        self.pre_output_gate_norm = GateAddNorm(self.hidden_dim, dropout=None, trainable_add=False)

        
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)
        
        self.device = device
        
    
    def get_attention_mask(self, encoder_lengths: torch.LongTensor, decoder_lengths: torch.LongTensor):

        decoder_length = decoder_lengths.max()

        attend_step = torch.arange(decoder_length, device=self.device)
        predict_step = torch.arange(0, decoder_length, device=self.device)[:, None]
        decoder_mask = (attend_step >= predict_step).unsqueeze(0).expand(encoder_lengths.size(0), -1, -1)
       
        encoder_mask = self.create_mask(encoder_lengths.max(), encoder_lengths).unsqueeze(1).expand(-1, decoder_length, -1)
        
        mask = torch.cat([encoder_mask, decoder_mask], dim=2)

        return mask
    
    
    def create_mask(self, size: int, lengths: torch.LongTensor, inverse: bool = False) -> torch.BoolTensor:

        if inverse:  
            return torch.arange(size, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(-1)
        else:  
            return torch.arange(size, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(-1)
        
        
    def forward(self, x_enc, x_cat_enc, x_dec, x_cat_dec, enc_length, dec_length):
    
        
        cat_embs_enc = []
        cat_embs_dec = []
        if self.embeddings_enc:
            
            for i in range(x_cat_enc.shape[2]):
                y_enc = self.embeddings_enc[i](x_cat_enc[..., i])
                y_dec = self.embeddings_dec[i](x_cat_dec[..., i])
                cat_embs_enc.append(y_enc)
                cat_embs_dec.append(y_dec)
            
            
        scaled_x_enc = []
        scaled_x_dec = []

        for i in range(self.input_dim):
            x_sc_enc = self.prescalers_enc[i](x_enc[..., i:i+1])
            scaled_x_enc.append(x_sc_enc)

            x_sc_dec = self.prescalers_dec[i](x_dec[..., i:i+1])
            scaled_x_dec.append(x_sc_dec)
    
        vsn_enc, weight_enc = self.vsn_enc(scaled_x_enc, cat_embs_enc)
        vsn_dec, weight_dec = self.vsn_dec(scaled_x_dec, cat_embs_dec)
        
        encoder_output, (hidden, cell) = self.lstm_encoder(vsn_enc)

        decoder_output, _ = self.lstm_decoder(vsn_dec, (hidden, cell))
        
        lstm_output_encoder = self.post_lstm_gate_encoder(encoder_output)
        lstm_output_encoder = self.post_lstm_add_norm_encoder(lstm_output_encoder, vsn_enc)

        lstm_output_decoder = self.post_lstm_gate_decoder(decoder_output)
        lstm_output_decoder = self.post_lstm_add_norm_decoder(lstm_output_decoder, vsn_dec)

        lstm_output = torch.cat([lstm_output_encoder, lstm_output_decoder], dim=1)
        
        attn_output, attn_output_weights = self.multihead_attn(
            q=lstm_output[:, x_enc.shape[1]:],  
            k=lstm_output,
            v=lstm_output,
            mask=self.get_attention_mask(encoder_lengths=enc_length, decoder_lengths=dec_length),
        )

        attn_output = self.post_attn_gate_norm(attn_output, lstm_output[:, x_enc.shape[1]:])

        output = self.pos_wise_ff(attn_output)

        output = self.pre_output_gate_norm(output, lstm_output[:, x_enc.shape[1]:])
        output = self.output_layer(output)
        
            
        return output