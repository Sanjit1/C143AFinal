import torch
from torch import nn
import math

from .augmentations import GaussianSmoothing


class GRUDecoder(nn.Module):
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        layer_dim,
        nDays=24,
        dropout=0,
        device="cuda",
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        bidirectional=False,
    ):
        super(GRUDecoder, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.nDays = nDays
        self.device = device
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.bidirectional = bidirectional
        self.inputLayerNonlinearity = torch.nn.Softsign()
        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )
        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, self.gaussianSmoothWidth, dim=1
        )
        self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))

        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(neural_dim)
            
        
        self.layer_norm = nn.LayerNorm(self.neural_dim * self.kernelLen)

        # GRU layers
        self.gru_decoder = nn.GRU(
            (neural_dim) * self.kernelLen,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        for name, param in self.gru_decoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # Input layers
        for x in range(nDays):
            setattr(self, "inpLayer" + str(x), nn.Linear(neural_dim, neural_dim))

        for x in range(nDays):
            thisLayer = getattr(self, "inpLayer" + str(x))
            thisLayer.weight = torch.nn.Parameter(
                thisLayer.weight + torch.eye(neural_dim)
            )

        # rnn outputs
        if self.bidirectional:
            self.fc_decoder_out = nn.Linear(
                hidden_dim * 2, n_classes + 1
            )  # +1 for CTC blank
        else:
            self.fc_decoder_out = nn.Linear(hidden_dim, n_classes + 1)  # +1 for CTC blank
            

    def forward(self, neuralInput, dayIdx):
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))

        # apply day layer
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        transformedNeural = torch.einsum(
            "btd,bdk->btk", neuralInput, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)

        # stride/kernel
        stridedInputs = torch.permute(
            self.unfolder(
                torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)
            ),
            (0, 2, 1),
        )
        
        # stridedInputs = self.layer_norm(stridedInputs)

        # apply RNN layer
        if self.bidirectional:
            h0 = torch.zeros(
                self.layer_dim * 2,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()
        else:
            h0 = torch.zeros(
                self.layer_dim,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()

        hid, _ = self.gru_decoder(stridedInputs, h0.detach())
        

        # get seq
        seq_out = self.fc_decoder_out(hid)
        return seq_out




class GRUDecoderWithHead(nn.Module):
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        layer_dim,
        nDays=24,
        dropout=0,
        device="cuda",
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        bidirectional=False,
    ):
        super(GRUDecoderWithHead, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.nDays = nDays
        self.device = device
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.bidirectional = bidirectional
        self.inputLayerNonlinearity = torch.nn.Softsign()
        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )
        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, self.gaussianSmoothWidth, dim=1
        )
        self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))

        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(neural_dim)
            
        
        self.layer_norm = nn.LayerNorm(self.neural_dim * self.kernelLen)

        # GRU layers
        self.gru_decoder = nn.GRU(
            (neural_dim) * self.kernelLen,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        for name, param in self.gru_decoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # Input layers
        for x in range(nDays):
            setattr(self, "inpLayer" + str(x), nn.Linear(neural_dim, neural_dim))

        for x in range(nDays):
            thisLayer = getattr(self, "inpLayer" + str(x))
            thisLayer.weight = torch.nn.Parameter(
                thisLayer.weight + torch.eye(neural_dim)
            )
            
            
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

        # rnn outputs
        if self.bidirectional:
            self.fc_decoder_out = nn.Linear(
                hidden_dim * 2, n_classes + 1
            )  # +1 for CTC blank
        else:
            self.fc_decoder_out = nn.Linear(hidden_dim, n_classes + 1)  # +1 for CTC blank
            

    def forward(self, neuralInput, dayIdx):
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))

        # apply day layer
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        transformedNeural = torch.einsum(
            "btd,bdk->btk", neuralInput, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)

        # stride/kernel
        stridedInputs = torch.permute(
            self.unfolder(
                torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)
            ),
            (0, 2, 1),
        )
        
        # stridedInputs = self.layer_norm(stridedInputs)

        # apply RNN layer
        if self.bidirectional:
            h0 = torch.zeros(
                self.layer_dim * 2,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()
        else:
            h0 = torch.zeros(
                self.layer_dim,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()

        hid, _ = self.gru_decoder(stridedInputs, h0.detach())
        hid = self.head(hid)

        # get seq
        seq_out = self.fc_decoder_out(hid)
        return seq_out


class SinusoidEncoding(nn.Module):
    def __init__(self, d_model, max_len=1200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # pe: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (batch, time, d_model)
        """
        T = x.size(1)
        x = x + self.pe[:, :T, :]
        return self.dropout(x)
    
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embed = nn.Embedding(max_len, d_model)

    def forward(self, x):
        """
        x: (batch, time, d_model)
        """
        B, T, D = x.size()
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        pos = self.pos_embed(positions)  # (1, T, d_model)
        x = x + pos
        return self.dropout(x)

class TransformerDecoder(nn.Module):
    def __init__(self, neural_dim, n_classes, n_layers=6, d_model=512, n_heads=8, max_len=1200, dropout=0.1,
        nDays=24,
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0):
        super().__init__()
        self.nDays = nDays
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.inputLayerNonlinearity = torch.nn.Softsign()
        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )
        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, self.gaussianSmoothWidth, dim=1
        )
        self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))

        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(neural_dim)

        # Optional: remove day-specific linear layers entirely
        self.input_projection = nn.Linear(neural_dim*self.kernelLen, d_model)

        # Positional Encoding
        self.pos_encoding = SinusoidEncoding(d_model, max_len=max_len, dropout=dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=1024,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Phoneme classifier (+1 fo r CTC blank)
        self.fc_out = nn.Linear(d_model, n_classes + 1)

    def forward(self, neuralInput, dayIdx):
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))

        # apply day layer
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        transformedNeural = torch.einsum(
            "btd,bdk->btk", neuralInput, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)

        # stride/kernel
        stridedInputs = torch.permute(
            self.unfolder(
                torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)
            ),
            (0, 2, 1),
        )
        X = self.input_projection(stridedInputs)
        X = self.pos_encoding(X)
        h = self.transformer(X)
        return self.fc_out(h)
    
    
# Transformer + GRU Hybrid Decoder
class Hybrid(nn.Module):
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        n_heads,
        n_layers,
        layer_dim,
        nDays=24,
        dropout=0,
        device="cuda",
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        bidirectional=False,
        max_len=1200,
    ):
        super(Hybrid, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.nDays = nDays
        self.device = device
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.bidirectional = bidirectional
        self.inputLayerNonlinearity = torch.nn.Softsign()
        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )
        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, self.gaussianSmoothWidth, dim=1
        )
        self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))

        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(neural_dim)
            

        # Transformer Encoder

        # Optional: remove day-specific linear layers entirely
        self.input_projection = nn.Linear(neural_dim*self.kernelLen, hidden_dim)

        # Positional Encoding
        self.pos_encoding = SinusoidEncoding(hidden_dim, max_len=max_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=1024,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # GRU stuff
        self.layer_norm = nn.LayerNorm(self.neural_dim * self.kernelLen)

        # GRU layers
        self.gru_decoder = nn.GRU(
            (neural_dim) * self.kernelLen,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        for name, param in self.gru_decoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # Input layers
        for x in range(nDays):
            setattr(self, "inpLayer" + str(x), nn.Linear(neural_dim, neural_dim))

        for x in range(nDays):
            thisLayer = getattr(self, "inpLayer" + str(x))
            thisLayer.weight = torch.nn.Parameter(
                thisLayer.weight + torch.eye(neural_dim)
            )

        # rnn outputs
        if self.bidirectional:
            self.fc_decoder_out = nn.Linear(
                hidden_dim * 3, n_classes + 1
            )  # +1 for CTC blank
        else:
            self.fc_decoder_out = nn.Linear(hidden_dim * 2, n_classes + 1)  # +1 for CTC blank
            
        
            

    def forward(self, neuralInput, dayIdx):
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))

        # apply day layer
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        transformedNeural = torch.einsum(
            "btd,bdk->btk", neuralInput, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)

        # stride/kernel
        stridedInputs = torch.permute(
            self.unfolder(
                torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)
            ),
            (0, 2, 1),
        )
        
        X = self.input_projection(stridedInputs)
        X = self.pos_encoding(X)
        h = self.transformer(X)

        # apply RNN layer
        if self.bidirectional:
            h0 = torch.zeros(
                self.layer_dim * 2,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()
        else:
            h0 = torch.zeros(
                self.layer_dim,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()

        hid, _ = self.gru_decoder(stridedInputs, h0.detach())
        

        # get seq
        seq_out = self.fc_decoder_out(torch.cat([h, hid], dim=2))
        return seq_out
    
    
    
class GRUTransitionDecoder(nn.Module):
    """
    GRU + explicit phoneme-transition head.

    - Phoneme head: same as GRUDecoder (CTC over phonemes).
    - Transition head: CTC over bigram IDs: (p_t, p_{t+1}) -> id.
    """

    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        layer_dim,
        nDays=24,
        dropout=0,
        device="cuda",
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        bidirectional=False,
        # optional: explicit number of transition classes
        n_transition_classes=None,
    ):
        super().__init__()

        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.nDays = nDays
        self.device = device
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.bidirectional = bidirectional

        # === shared frontend (copied from GRUDecoder) ===
        self.inputLayerNonlinearity = torch.nn.Softsign()
        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )
        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, self.gaussianSmoothWidth, dim=1
        )

        self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))
        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(neural_dim)

        self.layer_norm = nn.LayerNorm(self.neural_dim * self.kernelLen)

        # GRU core
        self.gru_decoder = nn.GRU(
            (neural_dim) * self.kernelLen,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        for name, param in self.gru_decoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # input layers per day (copied)
        for x in range(nDays):
            setattr(self, "inpLayer" + str(x), nn.Linear(neural_dim, neural_dim))

        for x in range(nDays):
            thisLayer = getattr(self, "inpLayer" + str(x))
            thisLayer.weight = torch.nn.Parameter(
                thisLayer.weight + torch.eye(neural_dim)
            )

        # === heads ===
        # phoneme head (same as GRUDecoder)
        if self.bidirectional:
            head_in_dim = hidden_dim * 2
        else:
            head_in_dim = hidden_dim

        self.fc_decoder_out = nn.Linear(
            head_in_dim, n_classes + 1  # +1 for CTC blank
        )

        # transition head: bigram classes
        if n_transition_classes is None:
            # simple choice: all ordered pairs (including self): n_classes^2
            n_transition_classes = n_classes * n_classes

        self.n_transition_classes = n_transition_classes
        self.fc_transition_out = nn.Linear(
            head_in_dim, self.n_transition_classes + 1  # +1 for CTC blank
        )

    def forward(self, neuralInput, dayIdx):
        """
        Returns:
          phoneme_logits: [B, T, n_classes+1]
          transition_logits: [B, T, n_transition_classes+1]
        """
        # [B, T, D] -> smooth
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))

        # day-specific transform
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        transformedNeural = torch.einsum(
            "btd,bdk->btk", neuralInput, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)

        # stride/kernel via unfold
        stridedInputs = torch.permute(
            self.unfolder(
                torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)
            ),
            (0, 2, 1),
        )

        # layer norm
        stridedInputs = self.layer_norm(stridedInputs)

        # GRU
        if self.bidirectional:
            h0 = torch.zeros(
                self.layer_dim * 2,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()
        else:
            h0 = torch.zeros(
                self.layer_dim,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()

        hid, _ = self.gru_decoder(stridedInputs, h0.detach())

        # heads
        phoneme_logits = self.fc_decoder_out(hid)
        transition_logits = self.fc_transition_out(hid)

        return phoneme_logits, transition_logits



class TransitionMLPWrapper(nn.Module):
    def __init__(self, base_model, num_classes, hidden_dim=64):
        super().__init__()
        self.base_model = base_model
        log_T_init = torch.full((num_classes, num_classes), 1.0 / num_classes, dtype=torch.float32).log()
        self.log_T = nn.Parameter(log_T_init.clone())  # transition params
        self.mlp = nn.Sequential(
            nn.Linear(2 * num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        self.kernelLen = base_model.kernelLen
        self.strideLen = base_model.strideLen

    def forward(self, x, *args, **kwargs):
        logits = self.base_model(x, *args, **kwargs)  # [B, T, C]
        probs = torch.softmax(logits, dim=-1)        # [B, T, C]

        # Build a Markov prior from probabilities without stepping an RNN:
        # e.g., expected next-phone prior: m_t = probs_t @ softmax(log_T)
        T_probs = torch.softmax(self.log_T, dim=-1)  # [C, C]
        markov_prior = probs @ T_probs               # [B, T, C]

        # Combine raw logits + Markov prior via MLP
        combined = torch.cat([logits, markov_prior], dim=-1)   # [B, T, 2C]
        delta = self.mlp(combined)                             # [B, T, C]

        adjusted_logits = logits + delta
        return adjusted_logits
    
    
class EnsembledTransitionWrapper(nn.Module):
    def __init__(self, models: nn.ModuleList, num_classes, hidden_dim=64):
        super().__init__()
        self.models = models
        log_T_init = torch.full((num_classes, num_classes), 1.0 / num_classes, dtype=torch.float32).log()
        self.log_T = nn.Parameter(log_T_init.clone())  # transition params
        self.mlp = nn.Sequential(
            nn.Linear((len(models)+1) * num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        self.kernelLen = models[0].kernelLen
        self.strideLen = models[0].strideLen

    def forward(self, x, *args, **kwargs):
        T_probs = torch.softmax(self.log_T, dim=-1)  # [C, C]
        
        logits_list = []
        priors = []
        for model in self.models:
            logits = model(x, *args, **kwargs)  # [B, T, C]
            prior = torch.softmax(logits, dim=-1) @ T_probs  # [B, T, C]
            priors.append(prior)
        

        combined = torch.cat(priors, dim=-1)   # [B, T, (len(models)*C)]
        combined = torch.cat([logits, combined], dim=-1)   # [B, T, (len(models)+1)*C]
        delta = self.mlp(combined)                             # [B, T, C]

        adjusted_logits = logits + delta
        return adjusted_logits, logits_list
    
    
    
    
# This is not a model, this is just a custom cosine annealing thing to schedule the ensemble weight transition or something
class CosineAnnealing:
    def __init__(self, weight_start: float, weight_end: float, total_len: int):
        """
        Cosine annealing from weight_start to weight_end over total_len steps.

        Args:
            weight_start: initial value (at step 0)
            weight_end: final value (at step >= total_len)
            total_len: number of steps over which to anneal
        """
        if total_len <= 0:
            raise ValueError("total_len must be positive")
        self.weight_start = weight_start
        self.weight_end = weight_end
        self.total_len = total_len

    def __call__(self, current_step: int) -> float:
        """
        Get the annealed value at the given step.

        Assumes current_step starts at 0 and goes up.
        Values are clamped so that:
          - step <= 0   → weight_start
          - step >= T   → weight_end
        """
        if current_step <= 0:
            return self.weight_start
        if current_step >= self.total_len:
            return self.weight_end

        # Normalize step to [0, 1]
        t = current_step / self.total_len

        # Standard cosine annealing from start to end
        cos_term = 0.5 * (1 + math.cos(math.pi * t))  # goes from 1 → 0
        return self.weight_end + (self.weight_start - self.weight_end) * cos_term