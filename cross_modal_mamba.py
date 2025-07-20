import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
# (Assuming FcClassifier is defined elsewhere and imported.)
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from collections import OrderedDict
from models.base_model import BaseModel
from models.networks.lstm import LSTMEncoder
from models.networks.classifier import FcClassifier
from models.utils.config import OptConfig



class CrossModalMambaModel(nn.Module):
    """
    Multimodal emotion recognition model with Mamba-based fusion.
    Supports 'cross_attention_mamba' and 'direct_mamba' modes.
    """
    def __init__(self, opt):
        super(CrossModalMambaModel, self).__init__()
        self.fusion_type = opt.fusion_type  # 'cross_attention_mamba' or 'direct_mamba'
        self.hidden_dim = opt.hidden_dim   # common hidden dimension for fusion
        self.personal_dim = getattr(opt, 'personal_dim', 0)  # dim of personal embeddings
        
        # Linear projections to common hidden dimension
        self.audio_proj = nn.Linear(opt.audio_dim, self.hidden_dim)
        self.visual_proj = nn.Linear(opt.visual_dim, self.hidden_dim)
        
        # Cross-attention setup (only for cross_attention_mamba mode)
        if self.fusion_type == 'cross_attention_mamba':
            self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
            mamba_d_model = self.hidden_dim
        elif self.fusion_type == 'direct_mamba':
            mamba_d_model = self.hidden_dim * 2
        else:
            raise ValueError(f"Unsupported fusion_type: {self.fusion_type}")
        
        # Initialize Mamba (state-space) sequence model
        self.mamba = Mamba(
            d_model=mamba_d_model,
            d_state=opt.mamba_d_state,
            d_conv=opt.mamba_d_conv,
            expand=opt.mamba_expand
        )
        # Load pretrained Mamba weights if provided
        if hasattr(opt, 'pretrained_mamba') and opt.pretrained_mamba:
            self.mamba.load_state_dict(torch.load(opt.pretrained_mamba))
        
        # Fully-connected classifier: input dim = Mamba output dim + personal_dim
        classifier_input_dim = mamba_d_model + self.personal_dim
        self.classifier = FcClassifier(classifier_input_dim, opt.num_classes)
    
    def forward(self, audio_feats, visual_feats, personal_emb=None):
        """
        Forward pass computing logits and softmax predictions.
        Args:
          audio_feats: Tensor(batch, seq_len, audio_dim) from audio LSTM
          visual_feats: Tensor(batch, seq_len, visual_dim) from visual LSTM
          personal_emb: optional Tensor(batch, personal_dim)
        Returns:
          logits: Tensor(batch, num_classes)
          preds: Tensor(batch, num_classes) after softmax
        """
        # Project features to hidden dimension
        audio_hidden = self.audio_proj(audio_feats)  # (batch, T, hidden_dim)
        visual_hidden = self.visual_proj(visual_feats)  # (batch, T, hidden_dim)
        
        # Fusion stage
        if self.fusion_type == 'cross_attention_mamba':
            # Cross-modal attention: audio queries visual
            Q = self.q_proj(audio_hidden)               # (batch, T, hidden_dim)
            K = self.k_proj(visual_hidden)              # (batch, T, hidden_dim)
            V = self.v_proj(visual_hidden)              # (batch, T, hidden_dim)
            # Scaled dot-product attention
            scores = torch.bmm(Q, K.transpose(1, 2)) / (self.hidden_dim ** 0.5)  # (batch, T, T)
            attn_weights = torch.softmax(scores, dim=-1)
            fused_seq = torch.bmm(attn_weights, V)      # (batch, T, hidden_dim)
            mamba_input = fused_seq
        else:  # 'direct_mamba'
            # Concatenate audio and visual features
            mamba_input = torch.cat([audio_hidden, visual_hidden], dim=2)  # (batch, T, 2*hidden_dim)
        
        # Sequence modeling with Mamba
        mamba_out = self.mamba(mamba_input)  # (batch, T, d_model)
        pooled = mamba_out.mean(dim=1)       # Global average pooling (batch, d_model)
        
        # Append personal embedding if given
        if personal_emb is not None:
            pooled = torch.cat([pooled, personal_emb], dim=1)  # (batch, d_model + personal_dim)
        
        # Classification
        logits = self.classifier(pooled)      # (batch, num_classes)
        preds = torch.softmax(logits, dim=1)
        return logits, preds