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

# Import your custom CrossAttentionEncoder
# Make sure the path is correct based on where you saved cross_attention_encoder.py
from models.networks.cross_attention_encoder import CrossAttentionEncoder 

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


class ourModelMPDDCross(BaseModel, nn.Module):
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

        # --- NEW: Cross-Modal Attention Layers using custom CrossAttentionEncoder ---
        # The k_dim and v_dim for CrossAttention should be opt.cross_d_model // opt.cross_nhead
        # as per your custom CrossAttention implementation.
        assert opt.cross_d_model % opt.cross_nhead == 0, "cross_d_model must be divisible by cross_nhead"
        cross_k_dim_per_head = opt.cross_d_model // opt.cross_nhead
        cross_v_dim_per_head = opt.cross_d_model // opt.cross_nhead # Typically k_dim and v_dim are the same per head

        # A2V: Acoustic (Query) attends to Visual (Key/Value)
        self.netCrossA2V = CrossAttentionEncoder(
            in_dim1=opt.embd_size_a,    # Query input dimension (output dim of netEmoA)
            in_dim2=opt.embd_size_v,    # Key/Value input dimension (output dim of netEmoV)
            k_dim=cross_k_dim_per_head,
            v_dim=cross_v_dim_per_head,
            num_heads=opt.cross_nhead,
            num_layers=opt.cross_num_layers, # Uses the specified number of cross-attention layers
            dropout=opt.cross_dropout
        )
        self.model_names.append('CrossA2V')

        # V2A: Visual (Query) attends to Acoustic (Key/Value)
        self.netCrossV2A = CrossAttentionEncoder(
            in_dim1=opt.embd_size_v,    # Query input dimension (output dim of netEmoV)
            in_dim2=opt.embd_size_a,    # Key/Value input dimension (output dim of netEmoA)
            k_dim=cross_k_dim_per_head,
            v_dim=cross_v_dim_per_head,
            num_heads=opt.cross_nhead,
            num_layers=opt.cross_num_layers, # Uses the specified number of cross-attention layers
            dropout=opt.cross_dropout
        )
        self.model_names.append('CrossV2A')

        # --- Transformer Fusion model ---
        # The input to this self-attention Transformer will be the concatenation
        # of the outputs from the CrossAttentionEncoder blocks.
        # The output of CrossAttentionEncoder (and thus CrossAttention) matches its `in_dim1`.
        # So, cross_attn_A2V_output will be (batch_size, seq_len, opt.embd_size_a)
        # And cross_attn_V2A_output will be (batch_size, seq_len, opt.embd_size_v)
        # Concatenating them will result in (batch_size, seq_len, opt.embd_size_a + opt.embd_size_v).
        
        fusion_transformer_input_dim = opt.embd_size_a + opt.embd_size_v
        
        # Ensure that the d_model for the self-attention Transformer matches the input dim
        emo_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=fusion_transformer_input_dim, 
                                                             nhead=opt.Transformer_head, 
                                                             batch_first=True,
                                                             dropout=opt.dropout_rate) 
        self.netEmoFusion = torch.nn.TransformerEncoder(emo_encoder_layer, num_layers=opt.Transformer_layers)
        self.model_names.append('EmoFusion')

        # --- Classifier ---
        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))

        # Calculate classifier input size based on the output of the final fusion Transformer after pooling.
        # The output of netEmoFusion will be (batch_size, seq_len, fusion_transformer_input_dim)
        # After pooling (torch.max(..., dim=1)[0]), it will be (batch_size, fusion_transformer_input_dim)
        self.cls_input_size = fusion_transformer_input_dim
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
            self.criterion_ce = torch.nn.CrossEntropyLoss(weight=opt.class_weights) 

            if opt.use_ICL: 
                self.criterion_focal = Focal_Loss(weight=opt.focal_weight, gamma=opt.focal_gamma, class_weights=opt.class_weights) 
            else:
                self.criterion_focal = torch.nn.CrossEntropyLoss(weight=opt.class_weights) 

            parameters_to_optimize = []
            for net_name in self.model_names:
                net = getattr(self, 'net' + net_name, None)
                if net is not None:
                    parameters_to_optimize.append({'params': net.parameters()})
                else:
                    print(f"Warning: Model component 'net{net_name}' is None, skipping its parameters from optimizer.")

            self.optimizer = torch.optim.Adam(parameters_to_optimize, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
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
                    # Load weights for existing encoders
                    if hasattr(self.pretrained_encoder, 'netEmoA') and self.pretrained_encoder.netEmoA is not None:
                        self.netEmoA.load_state_dict(transform_key_for_parallel(self.pretrained_encoder.netEmoA.state_dict()))
                    if hasattr(self.pretrained_encoder, 'netEmoV') and self.pretrained_encoder.netEmoV is not None:
                        self.netEmoV.load_state_dict(transform_key_for_parallel(self.pretrained_encoder.netEmoV.state_dict()))
                    
                    # Load weights for NEW cross-attention encoders if they exist in pretrained model
                    if hasattr(self.pretrained_encoder, 'netCrossA2V') and self.pretrained_encoder.netCrossA2V is not None:
                        self.netCrossA2V.load_state_dict(transform_key_for_parallel(self.pretrained_encoder.netCrossA2V.state_dict()))
                    if hasattr(self.pretrained_encoder, 'netCrossV2A') and self.pretrained_encoder.netCrossV2A is not None:
                        self.netCrossV2A.load_state_dict(transform_key_for_parallel(self.pretrained_encoder.netCrossV2A.state_dict()))

                    # Load weights for the final fusion transformer
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
        # You might need to define OptConfig or ensure it's imported correctly
        # For now, let's assume it's a simple class like the one in train.py
        class OptConfig:
            def __init__(self):
                pass
            def load(self, content):
                self.__dict__.update(content)

        opt = OptConfig()
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

        # --- NEW: Cross-Attention operations using CrossAttentionEncoder ---
        # Acoustic (Query) attends to Visual (Key/Value)
        # query_features (x) is emo_feat_A, kv_features (x2 in CrossAttention) is emo_feat_V
        cross_attn_A2V_output = self.netCrossA2V(query_features=emo_feat_A, kv_features=emo_feat_V)
        # Output: (batch_size, seq_len, opt.embd_size_a) (since proj_o in CrossAttention maps to in_dim1)

        # Visual (Query) attends to Acoustic (Key/Value)
        # query_features (x) is emo_feat_V, kv_features (x2 in CrossAttention) is emo_feat_A
        cross_attn_V2A_output = self.netCrossV2A(query_features=emo_feat_V, kv_features=emo_feat_A)
        # Output: (batch_size, seq_len, opt.embd_size_v) (since proj_o in CrossAttention maps to in_dim1)

        # 2. Concatenate cross-attended features along the feature dimension (dim=-1)
        # Result: [batch_size, seq_len, opt.embd_size_a + opt.embd_size_v]
        # This correctly maintains the sequence length for the subsequent TransformerEncoder.
        emo_fusion_feat_input = torch.cat((cross_attn_A2V_output, cross_attn_V2A_output), dim=-1)
        
        # 3. Transformer Fusion (Self-attention on the combined cross-attention features)
        # Input: [batch_size, seq_len, fusion_transformer_input_dim]
        # Output: [batch_size, seq_len, fusion_transformer_input_dim]
        emo_fusion_feat = self.netEmoFusion(emo_fusion_feat_input) 

        # 4. Pooling: Apply max-pooling over the sequence length dimension (dim=1)
        # This converts (batch_size, seq_len, hidden_size) to (batch_size, hidden_size)
        final_fusion_feat = torch.max(emo_fusion_feat, dim=1)[0] # [0] to get the values, not indices

        # 5. Concatenate personalized features if they are used
        if self.personalized is not None:
            final_fusion_feat = torch.cat((final_fusion_feat, self.personalized), dim=-1) 
            
        # 6. Pass through classifiers
        if final_fusion_feat.size(1) != self.cls_input_size:
            raise ValueError(f"Classifier input dimension mismatch. Expected {self.cls_input_size}, but got {final_fusion_feat.size(1)}. "
                             f"Ensure opt.embd_size_a, opt.embd_size_v and opt.personalized_dim are correctly set and match the model's structure.")
        
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