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


class CrossModalAttention(nn.Module):
    def __init__(self, d_a, d_v, d_model, nhead):
        super().__init__()
        self.q_a = nn.Linear(d_a, d_model)
        self.k_v = nn.Linear(d_v, d_model)
        self.v_v = nn.Linear(d_v, d_model)
        self.q_v = nn.Linear(d_v, d_model)
        self.k_a = nn.Linear(d_a, d_model)
        self.v_a = nn.Linear(d_a, d_model)

        self.nhead = nhead
        self.scale = 1.0 / math.sqrt(d_model // nhead)

    def _reshape_to_heads(self, x):
        B, T, D = x.size()
        h = self.nhead
        d_head = D // h
        return x.view(B, T, h, d_head).transpose(1, 2)  # (B, h, T, d_head)

    def _combine_from_heads(self, xh):
        B, h, T, dh = xh.size()
        return xh.transpose(1, 2).contiguous().view(B, T, h * dh)

    def forward(self, feat_A, feat_V):
        Q_A = self.q_a(feat_A)
        K_V = self.k_v(feat_V)
        V_V = self.v_v(feat_V)

        Q_V = self.q_v(feat_V)
        K_A = self.k_a(feat_A)
        V_A = self.v_a(feat_A)

        Q_Ah, K_Vh, V_Vh = map(self._reshape_to_heads, (Q_A, K_V, V_V))
        Q_Vh, K_Ah, V_Ah = map(self._reshape_to_heads, (Q_V, K_A, V_A))

        attn_A2V = (Q_Ah @ K_Vh.transpose(-2, -1)) * self.scale
        attn_A2V = F.softmax(attn_A2V, dim=-1)
        A2Vh = attn_A2V @ V_Vh

        attn_V2A = (Q_Vh @ K_Ah.transpose(-2, -1)) * self.scale
        attn_V2A = F.softmax(attn_V2A, dim=-1)
        V2Ah = attn_V2A @ V_Ah

        A2V = self._combine_from_heads(A2Vh)
        V2A = self._combine_from_heads(V2Ah)

        return A2V, V2A, attn_A2V, attn_V2A


class CrossModalModel(BaseModel, nn.Module):
    def __init__(self, opt):
        # Patch opt fields if missing to avoid attribute errors
        if not hasattr(opt, 'init_type'):
            opt.init_type = 'normal'
        if not hasattr(opt, 'init_gain'):
            opt.init_gain = 0.02
        if not hasattr(opt, 'lr_policy'):
            opt.lr_policy = 'linear'
        if not hasattr(opt, 'gpu_ids'):
            opt.gpu_ids = []
        if not hasattr(opt, 'niter'):
            opt.niter = 100
        if not hasattr(opt, 'niter_decay'):
            opt.niter_decay = 100
        if not hasattr(opt, 'epoch_count'):
            opt.epoch_count = 1
        if not hasattr(opt, 'lr_decay_iters'):
            opt.lr_decay_iters = 50
        if not hasattr(opt, 'device'):
            opt.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not hasattr(opt, 'personalized_dim'):
            opt.personalized_dim = 0    

        BaseModel.__init__(self, opt)
        nn.Module.__init__(self)

        self.model_names = ['EmoA', 'EmoV', 'CrossAttn', 'Classifier']
        self.loss_names = ['emo_CE']

        self.device = torch.device(opt.device)

        # LSTM Encoders for audio and video
        self.netEmoA = LSTMEncoder(opt.input_dim_a, opt.embd_size_a, embd_method=opt.embd_method_a)
        self.netEmoV = LSTMEncoder(opt.input_dim_v, opt.embd_size_v, embd_method=opt.embd_method_v)

        # Cross modal attention layer
        self.netCrossAttn = CrossModalAttention(
            d_a=opt.embd_size_a,
            d_v=opt.embd_size_v,
            d_model=opt.cross_d_model,
            nhead=int(opt.cross_nhead)
        )

        # Classifier input size: concatenated attended features + optional personalized features
        print("\n[DEBUG] ----------- opt fields check -----------")
        print(f"opt.cross_d_model = {getattr(opt, 'cross_d_model', 'MISSING')}")
        print(f"opt.personalized_dim = {getattr(opt, 'personalized_dim', 'MISSING')}")
        print(f"opt.use_personalized_feat = {getattr(opt, 'use_personalized_feat', 'MISSING')}")
        print(f"opt.cls_layers = {getattr(opt, 'cls_layers', 'MISSING')}")
        print(f"Final expected classifier input dim: {opt.cross_d_model * 2 + (opt.personalized_dim if getattr(opt, 'use_personalized_feat', False) else 0)}")
        print("[DEBUG] ----------------------------------------\n")
        cls_in = opt.cross_d_model * 2
        if hasattr(opt, 'personalized_dim') and opt.personalized_dim > 0:
            cls_in += opt.personalized_dim

        cls_layers = list(map(int, opt.cls_layers.split(',')))
        self.netClassifier = FcClassifier(cls_in, cls_layers, output_dim=opt.emo_output_dim, dropout=opt.dropout_rate)

        # Loss initialization with possible class weights (default None)
        self.criterion_ce = None
        if self.isTrain:
            self.criterion_ce = nn.CrossEntropyLoss(weight=None)
            # collect parameters
            params = list(self.netEmoA.parameters()) + \
                     list(self.netEmoV.parameters()) + \
                     list(self.netCrossAttn.parameters()) + \
                     list(self.netClassifier.parameters())
            self.optimizer = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)

        # Create save directory
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(self.save_dir, exist_ok=True)

        self.setup(opt)

    def set_input(self, input):
        self.acoustic = input['A_feat'].float().to(self.device)
        self.visual = input['V_feat'].float().to(self.device)
        self.emo_label = input['emo_label'].to(self.device)

        self.personalized = input.get('personalized_feat', None)
        if self.personalized is not None:
            self.personalized = self.personalized.float().to(self.device)

    def forward(self):
        A = self.netEmoA(self.acoustic)  # Audio embedding [B, T, d]
        V = self.netEmoV(self.visual)    # Video embedding [B, T, d]

        # Cross-modal attention outputs και attention weights
        A2V, V2A, self.attn_A2V, self.attn_V2A = self.netCrossAttn(A, V)

        # Pool over time (mean pooling)
        A2V_p = A2V.mean(dim=1)  # [B, d]
        V2A_p = V2A.mean(dim=1)  # [B, d]

        # Fuse attended embeddings
        fused = torch.cat([A2V_p, V2A_p], dim=-1)  # [B, 2*d]

        # Αν υπάρχει personalized feature, πρόσθεσέ το
        if self.personalized is not None:
            fused = torch.cat([fused, self.personalized], dim=-1)

        # Classifier
        self.emo_logits, _ = self.netClassifier(fused)

        # Softmax για πρόβλεψη κατηγορίας
        self.emo_pred = torch.softmax(self.emo_logits, dim=-1)

    def backward(self):
        assert self.criterion_ce is not None, "criterion_ce not initialized"
        self.loss_emo_CE = self.criterion_ce(self.emo_logits, self.emo_label)
        self.loss_emo_CE.backward()

    def optimize_parameters(self, epoch=None):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def get_current_losses(self):
        return {'emo_CE': self.loss_emo_CE.item() if hasattr(self, 'loss_emo_CE') else 0.0}

    def update_class_weights(self, class_weights):
        """Με αυτή τη μέθοδο μπορείς να περάσεις δυναμικά τα βάρη κλάσεων (torch.tensor) στο loss"""
        if class_weights is not None:
            self.criterion_ce = nn.CrossEntropyLoss(weight=class_weights.to(self.device))

    def load_from_opt_record(self, path):
        opt_content = json.load(open(path, 'r'))
        opt = OptConfig()
        opt.load(opt_content)
        return opt
