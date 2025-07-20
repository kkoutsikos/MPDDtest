import os
import torch
import math
import json
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from models.base_model import BaseModel
from models.networks.lstm import LSTMEncoder
from models.networks.classifier import FcClassifier
from models.utils.config import OptConfig


class FuzeModel(BaseModel, nn.Module):
    def __init__(self, opt):
        super().__init__(opt)
        nn.Module.__init__(self)

        self.loss_names = []
        self.model_names = []

        # Acoustic & Visual LSTM Encoders
        self.netEmoA = LSTMEncoder(opt.input_dim_a, opt.embd_size_a, embd_method=opt.embd_method_a)
        self.netEmoV = LSTMEncoder(opt.input_dim_v, opt.embd_size_v, embd_method=opt.embd_method_v)
        self.model_names += ['EmoA', 'EmoV']

        # Fusion Transformer
        fusion_layer = nn.TransformerEncoderLayer(d_model=opt.hidden_size, nhead=int(opt.Transformer_head), batch_first=True)
        self.netEmoFusion = nn.TransformerEncoder(fusion_layer, num_layers=opt.Transformer_layers)
        self.model_names.append('EmoFusion')

        # Classifiers
        cls_layers = list(map(int, opt.cls_layers.split(',')))
        
        if getattr(opt, "fusion_type", "joint") == "late":
            cls_input_size_C  = opt.feature_max_len * opt.embd_size_a + 1024  # For audio path
            cls_input_size_CF = opt.feature_max_len * opt.embd_size_v + 1024  # For video path
        else:  # Joint fusion uses Transformer output
            cls_input_size_C = cls_input_size_CF = opt.feature_max_len * opt.hidden_size + 1024

        self.netEmoC  = FcClassifier(cls_input_size_C,  cls_layers, output_dim=opt.emo_output_dim, dropout=opt.dropout_rate)
        self.netEmoCF = FcClassifier(cls_input_size_CF, cls_layers, output_dim=opt.emo_output_dim, dropout=opt.dropout_rate)

        
        self.model_names += ['EmoC', 'EmoCF']
        self.loss_names += ['emo_CE', 'EmoF_CE']

        self.temperature = opt.temperature
        self.criterion_ce = nn.CrossEntropyLoss()

        if self.isTrain:
            if not opt.use_ICL:
                self.criterion_focal = nn.CrossEntropyLoss()
            else:
                self.criterion_focal = Focal_Loss()
            parameters = [{'params': getattr(self, 'net' + n).parameters()} for n in self.model_names]
            self.optimizer = torch.optim.Adam(parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            self.ce_weight = opt.ce_weight
            self.focal_weight = opt.focal_weight

        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(self.save_dir, exist_ok=True)

    
    
    def post_process(self):
        # called after model.setup()
        def transform_key_for_parallel(state_dict):
            return OrderedDict([('module.' + key, value) for key, value in state_dict.items()])

        if self.isTrain:
            print('[ Init ] Load parameters from pretrained encoder network')
            f = lambda x: transform_key_for_parallel(x)
            self.netEmoA.load_state_dict(f(self.pretrained_encoder.netEmoA.state_dict()))
            self.netEmoV.load_state_dict(f(self.pretrained_encoder.netEmoV.state_dict()))
            self.netEmoFusion.load_state_dict(f(self.pretrained_encoder.netEmoFusion.state_dict()))
            
            
    def set_input(self, input):
        self.acoustic = input['A_feat'].float().to(self.device)
        self.visual = input['V_feat'].float().to(self.device)
        self.emo_label = input['emo_label'].to(self.device)
        self.personalized = input.get('personalized_feat', None)
        if self.personalized is not None:
            self.personalized = self.personalized.float().to(self.device)

    def forward(self, acoustic_feat=None, visual_feat=None):
        if acoustic_feat is not None:
            self.acoustic = acoustic_feat.float().to(self.device)
            self.visual = visual_feat.float().to(self.device)

        feat_A = self.netEmoA(self.acoustic)
        feat_V = self.netEmoV(self.visual)
        fusion_type = getattr(self.opt, 'fusion_type', 'joint')
        batch_size = feat_A.size(0)

        if fusion_type == 'joint':
            fused_feat = torch.cat((feat_V, feat_A), dim=-1)
            fused_feat = self.netEmoFusion(fused_feat)
            fused_feat = fused_feat.permute(1, 0, 2).reshape(batch_size, -1)
            if self.personalized is not None:
                fused_feat = torch.cat((fused_feat, self.personalized), dim=-1)
            self.emo_logits_fusion, _ = self.netEmoCF(fused_feat)
            self.emo_logits = self.emo_logits_fusion
            self.emo_pred = F.softmax(self.emo_logits, dim=-1)

        elif fusion_type == 'late':
            flat_A = feat_A.permute(1, 0, 2).reshape(batch_size, -1)
            flat_V = feat_V.permute(1, 0, 2).reshape(batch_size, -1)
            if self.personalized is not None:
                flat_A = torch.cat((flat_A, self.personalized), dim=-1)
                flat_V = torch.cat((flat_V, self.personalized), dim=-1)
            logits_A, _ = self.netEmoC(flat_A)
            logits_V, _ = self.netEmoCF(flat_V)
            probs_A = F.softmax(logits_A, dim=-1)
            probs_V = F.softmax(logits_V, dim=-1)
            self.emo_logits = (logits_A + logits_V) / 2
            self.emo_pred = (probs_A + probs_V) / 2

        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

    def backward(self):
        self.loss_emo_CE = self.criterion_ce(self.emo_logits, self.emo_label)
        self.loss_EmoF_CE = self.focal_weight * self.criterion_focal(self.emo_logits, self.emo_label)
        loss = self.loss_emo_CE + self.loss_EmoF_CE
        loss.backward()
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net' + model).parameters(), 1.0)

    def optimize_parameters(self, epoch):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def post_process(self):
        def transform_key(state):
            return OrderedDict([('module.' + k, v) for k, v in state.items()])
        if self.isTrain:
            print('[ Init ] Loading pre-trained encoders')
            f = lambda x: transform_key(x)
            self.netEmoA.load_state_dict(f(self.pretrained_encoder.netEmoA.state_dict()))
            self.netEmoV.load_state_dict(f(self.pretrained_encoder.netEmoV.state_dict()))
            self.netEmoFusion.load_state_dict(f(self.pretrained_encoder.netEmoFusion.state_dict()))

    def load_from_opt_record(self, path):
        opt_content = json.load(open(path, 'r'))
        opt = OptConfig()
        opt.load(opt_content)
        return opt


class ActivateFun(torch.nn.Module):
    def __init__(self, opt):
        super(ActivateFun, self).__init__()
        self.activate_fun = opt.activate_fun

    def _gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x):
        if self.activate_fun == 'relu':
            return torch.relu(x)
        elif self.activate_fun == 'gelu':
            return self._gelu(x)
        

class Focal_Loss(nn.Module):
    def __init__(self, weight=0.5, gamma=3, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = weight
        self.reduction = reduction

    def forward(self, preds, targets):
        ce_loss = F.cross_entropy(preds, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        return focal_loss
