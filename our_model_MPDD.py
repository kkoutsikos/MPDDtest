import torch
import os
import json
from collections import OrderedDict
import torch.nn.functional as F
import torch.nn as nn

# Υποθέτουμε ότι αυτές οι κλάσεις υπάρχουν στους φακέλους models/
# Βεβαιωθείτε ότι η BaseModel, LSTMEncoder και FcClassifier είναι σωστά εισαγμένα
from models.base_model import BaseModel
from models.networks.lstm import LSTMEncoder
from models.networks.classifier import FcClassifier
# Αν το OptConfig είναι πραγματικά απαραίτητο για το μοντέλο, βεβαιωθείτε ότι το path είναι σωστό
# from models.utils.config import OptConfig 


# Η κλάση Focal_Loss παραμένει η ίδια
class Focal_Loss(torch.nn.Module):
    # Changed gamma default to 4 as per your latest code, but it will be overridden by opt.focal_gamma
    def __init__(self, weight=0.5, gamma=4, reduction='mean', class_weights=None):
        super(Focal_Loss, self).__init__()
        self.gamma = gamma
        self.alpha = weight 
        self.reduction = reduction
        self.class_weights = class_weights # Store class_weights

    def forward(self, preds, targets):
        targets = targets.long() 
        # Ensure class_weights are passed to F.cross_entropy
        ce_loss = F.cross_entropy(preds, targets, reduction='none', weight=self.class_weights) 

        if isinstance(self.alpha, (float, int)):
            alpha_tensor = torch.full_like(targets, self.alpha, dtype=torch.float)
        elif torch.is_tensor(self.alpha):
            alpha_tensor = torch.full_like(targets, self.alpha.item(), dtype=torch.float)
        else:
            raise TypeError("Alpha must be a scalar (float/int) or a 0-dim tensor.")
        
        pt = torch.exp(-ce_loss) 
        
        focal_loss = alpha_tensor * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'none':
            return focal_loss
        elif self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            raise NotImplementedError("Invalid reduction mode. Please choose 'none', 'mean', or 'sum'.")


class ourModelMPDD(BaseModel, nn.Module):
    def __init__(self, opt):
        nn.Module.__init__(self)
        super().__init__(opt) 

        self.loss_names = []
        self.model_names = []

        # --- Acoustic Encoder ---
        self.netEmoA = LSTMEncoder(opt.input_dim_a, opt.embd_size_a, embd_method='last')
        self.model_names.append('EmoA')

        # --- Visual Encoder ---
        self.netEmoV = LSTMEncoder(opt.input_dim_v, opt.embd_size_v, embd_method='last')
        self.model_names.append('EmoV')

        # --- Transformer Fusion model ---
        if opt.hidden_size != (opt.embd_size_a + opt.embd_size_v):
            print(f"Warning: opt.hidden_size ({opt.hidden_size}) does not match sum of embd_sizes ({opt.embd_size_a + opt.embd_size_v}). "
                  f"Using sum of embd_sizes as d_model for Transformer.")
            opt.hidden_size = opt.embd_size_a + opt.embd_size_v
            
        emo_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=opt.hidden_size, 
                                                             nhead=opt.Transformer_head, 
                                                             batch_first=True,
                                                             dropout=opt.dropout_rate) 
        self.netEmoFusion = torch.nn.TransformerEncoder(emo_encoder_layer, num_layers=opt.Transformer_layers)
        self.model_names.append('EmoFusion')

        # --- Classifier ---
        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))

        # Calculate classifier input size and store it for later validation
        self.cls_input_size = opt.hidden_size 
        if opt.use_personalized_feat:
            self.cls_input_size += opt.personalized_dim

        self.netEmoC = FcClassifier(self.cls_input_size, cls_layers, output_dim=opt.emo_output_dim, dropout=opt.dropout_rate)
        self.model_names.append('EmoC')
        self.loss_names.append('emo_CE')

        self.netEmoCF = FcClassifier(self.cls_input_size, cls_layers, output_dim=opt.emo_output_dim, dropout=opt.dropout_rate)
        self.model_names.append('EmoCF')
        self.loss_names.append('EmoF_CE') 

        self.temperature = opt.temperature
        
        if self.isTrain:
            # --- CRITICAL CORRECTION HERE ---
            # Use opt.class_weights for CrossEntropyLoss
            self.criterion_ce = torch.nn.CrossEntropyLoss(weight=opt.class_weights) # <--- CORRECTED!

            if opt.use_ICL: 
                # Use opt.focal_weight, opt.focal_gamma, and opt.class_weights for Focal_Loss
                self.criterion_focal = Focal_Loss(weight=opt.focal_weight, gamma=opt.focal_gamma, class_weights=opt.class_weights) # <--- CORRECTED!
            else:
                # If not using ICL, but still want class weights for CE
                self.criterion_focal = torch.nn.CrossEntropyLoss(weight=opt.class_weights) # <--- CORRECTED!

            parameters_to_optimize = []
            for net_name in self.model_names:
                net = getattr(self, 'net' + net_name, None)
                if net is not None:
                    parameters_to_optimize.append({'params': net.parameters()})
                else:
                    print(f"Warning: Model component 'net{net_name}' is None, skipping its parameters from optimizer.")

            self.optimizer = torch.optim.Adam(parameters_to_optimize, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            self.ce_weight = opt.ce_weight
            self.focal_weight_opt = opt.focal_weight 

        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name) 
        os.makedirs(self.save_dir, exist_ok=True) 


    def post_process(self):
        def transform_key_for_parallel(state_dict):
            return OrderedDict([('module.' + key if not key.startswith('module.') else key, value) for key, value in state_dict.items()])

        if self.isTrain: 
            if hasattr(self, 'pretrained_encoder') and self.pretrained_encoder is not None:
                print('[ Init ] Load parameters from pretrained encoder network (if available)')
                try:
                    if hasattr(self.pretrained_encoder, 'netEmoA') and self.pretrained_encoder.netEmoA is not None:
                        self.netEmoA.load_state_dict(transform_key_for_parallel(self.pretrained_encoder.netEmoA.state_dict()))
                    if hasattr(self.pretrained_encoder, 'netEmoV') and self.pretrained_encoder.netEmoV is not None:
                        self.netEmoV.load_state_dict(transform_key_for_parallel(self.pretrained_encoder.netEmoV.state_dict()))
                    if hasattr(self.pretrained_encoder, 'netEmoFusion') and self.pretrained_encoder.netEmoFusion is not None:
                        self.netEmoFusion.load_state_dict(transform_key_for_parallel(self.pretrained_encoder.netEmoFusion.state_dict()))
                    print('[ Info ] Pretrained encoders loaded successfully.')
                except Exception as e:
                    print(f'[ Warning ] Failed to load pretrained encoders: {e}. Initializing from scratch.')
            else:
                print('[ Info ] No pretrained encoders found or configured. Initializing from scratch.')


    def load_from_opt_record(self, file_path):
        # Assuming OptConfig is defined or imported somewhere accessible
        # from models.utils.config import OptConfig 
        opt_content = json.load(open(file_path, 'r'))
        opt = OptConfig() # Make sure OptConfig is correctly imported or defined
        opt.load(opt_content)
        return opt

    def set_input(self, input):
        self.acoustic = input['A_feat'].float().to(self.device)
        self.visual = input['V_feat'].float().to(self.device)
        self.emo_label = input['emo_label'].to(self.device)

        if self.opt.use_personalized_feat and 'personalized_feat' in input:
            self.personalized = input['personalized_feat'].float().to(self.device)
        else:
            self.personalized = None
            
    def forward(self, acoustic_feat=None, visual_feat=None):
        if acoustic_feat is not None and visual_feat is not None:
            self.acoustic = acoustic_feat.float().to(self.device)
            self.visual = visual_feat.float().to(self.device)
        
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        # 1. Encoders: LSTMEncoder returns (batch_size, seq_len, hidden_size)
        emo_feat_A = self.netEmoA(self.acoustic) # -> [batch_size, seq_len, embd_size_a]
        emo_feat_V = self.netEmoV(self.visual)   # -> [batch_size, seq_len, embd_size_v]

        # 2. Pooling: Apply max-pooling over the sequence length dimension (dim=1)
        # This converts (batch_size, seq_len, hidden_size) to (batch_size, hidden_size)
        emo_feat_A_pooled = torch.max(emo_feat_A, dim=1)[0] # [0] to get the values, not indices
        emo_feat_V_pooled = torch.max(emo_feat_V, dim=1)[0] # [0] to get the values

        # 3. Unsqueeze for Transformer: Add a sequence dimension of 1
        # This converts (batch_size, hidden_size) to (batch_size, 1, hidden_size)
        emo_feat_A_unsqueeze = emo_feat_A_pooled.unsqueeze(1)
        emo_feat_V_unsqueeze = emo_feat_V_pooled.unsqueeze(1)

        # 4. Concatenate acoustic and visual features along the feature dimension (dim=-1)
        # Result: [batch_size, 1, embd_size_a + embd_size_v]
        emo_fusion_feat_input = torch.cat((emo_feat_V_unsqueeze, emo_feat_A_unsqueeze), dim=-1) 
        
        # 5. Transformer Fusion
        # Input: [batch_size, 1, hidden_size] -> Output: [batch_size, 1, hidden_size]
        emo_fusion_feat = self.netEmoFusion(emo_fusion_feat_input) 

        # 6. Squeeze Transformer output: Flatten from [batch_size, 1, hidden_size] to [batch_size, hidden_size]
        final_fusion_feat = emo_fusion_feat.squeeze(1) # Remove the sequence dimension of 1

        # 7. Concatenate personalized features if they are used
        if self.personalized is not None:
            final_fusion_feat = torch.cat((final_fusion_feat, self.personalized), dim=-1) 
        
        # 8. Pass through classifiers
        if final_fusion_feat.size(1) != self.cls_input_size:
            raise ValueError(f"Classifier input dimension mismatch. Expected {self.cls_input_size}, but got {final_fusion_feat.size(1)}. "
                             f"Ensure opt.hidden_size ({self.opt.hidden_size}) and opt.personalized_dim ({self.opt.personalized_dim if self.opt.use_personalized_feat else 0}) are correctly set and match the model's structure.")
        
        self.emo_logits_fusion, _ = self.netEmoCF(final_fusion_feat)
        self.emo_logits, _ = self.netEmoC(final_fusion_feat)
        self.emo_pred = F.softmax(self.emo_logits, dim=-1)


    def backward(self):
        self.loss_emo_CE = self.ce_weight * self.criterion_ce(self.emo_logits, self.emo_label) 
        self.loss_EmoF_CE = self.focal_weight_opt * self.criterion_focal(self.emo_logits_fusion, self.emo_label)
        loss = self.loss_emo_CE + self.loss_EmoF_CE

        loss.backward()

        for model_name in self.model_names:
            current_net = getattr(self, 'net' + model_name, None)
            if current_net is not None:
                torch.nn.utils.clip_grad_norm_(current_net.parameters(), 1.0)

    def optimize_parameters(self, epoch):
        self.forward()
        for optimizer in self.optimizers: # Iterate through optimizers
            optimizer.zero_grad()
        self.backward()
        for optimizer in self.optimizers: # Iterate through optimizers
            optimizer.step()