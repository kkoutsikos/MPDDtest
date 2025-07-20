from datetime import datetime
import os
import json
import time
import argparse
import torch
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import logging
import sys

# Import for ReduceLROnPlateau
import torch.optim.lr_scheduler as lr_scheduler # <--- ADD THIS LINE

from train_val_split import train_val_split1, train_val_split2
from dataset import AudioVisualDataset
from utils.logger import get_logger

# Import your model
from models.our.our_model_MPDD import ourModelMPDD
from models.our.our_model_MPDD_cross import ourModelMPDDCross


class Opt:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

def load_config(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)

def eval(model, val_loader, device):
    model.eval()
    total_preds, total_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for data in val_loader:
            for k, v in data.items():
                data[k] = v.to(device)
            model.set_input(data)
            model.test()

            if hasattr(model, 'criterion_focal') and hasattr(model, 'emo_logits_fusion'):
                loss_val = model.criterion_focal(model.emo_logits_fusion, data['emo_label'])
                total_loss += loss_val.item()
            elif hasattr(model, 'criterion_ce') and hasattr(model, 'emo_logits_fusion'):
                loss_val = model.criterion_ce(model.emo_logits_fusion, data['emo_label'])
                total_loss += loss_val.item()
            elif hasattr(model, 'criterion_ce') and hasattr(model, 'emo_logits'):
                loss_val = model.criterion_ce(model.emo_logits, data['emo_label'])
                total_loss += loss_val.item()
            else:
                pass
            
            total_preds.append(model.emo_pred.argmax(dim=1).cpu().numpy())
            total_labels.append(data['emo_label'].cpu().numpy())

    preds = np.concatenate(total_preds)
    labels = np.concatenate(total_labels)

    acc_unweighted = accuracy_score(labels, preds)
    weights = 1 / (np.bincount(labels)[labels] + 1e-6) # Added 1e-6 for stability
    acc_weighted = accuracy_score(labels, preds, sample_weight=weights)
    f1_weighted = f1_score(labels, preds, average='weighted', zero_division=0)
    f1_unweighted = f1_score(labels, preds, average='macro', zero_division=0)
    cm = confusion_matrix(labels, preds)

    avg_val_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0

    return avg_val_loss, f1_weighted, f1_unweighted, acc_weighted, acc_unweighted, cm

# Pass the optimizer to train_model, as it's defined outside in ourModelMPDDCross
# Also pass the scheduler
def train_model(train_json, model, optimizer, scheduler, audio_path, video_path, max_len, best_model_name_prefix, seed): # <--- MODIFIED FUNCTION SIGNATURE
    global logger
    logger.info(f'Using personalized features: {args.personalized_features_file}')
    device = args.device # Read device from args, as it's passed from bash
    model.to(device)

    if args.track_option == 'Track1':
        train_data, val_data, _, _ = train_val_split1(train_json, val_ratio=0.1, random_seed=seed)
    else:
        train_data, val_data, _, _ = train_val_split2(train_json, val_percentage=0.1, seed=seed)

    train_loader = DataLoader(
        AudioVisualDataset(train_data, args.labelcount, args.personalized_features_file, max_len,
                           batch_size=args.batch_size, audio_path=audio_path, video_path=video_path),
        batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(
        AudioVisualDataset(val_data, args.labelcount, args.personalized_features_file, max_len,
                           batch_size=args.batch_size, audio_path=audio_path, video_path=video_path),
        batch_size=args.batch_size, shuffle=False)

    logger.info(f'Training samples: {len(train_loader.dataset)}, Validation samples: {len(val_loader.dataset)}')

    best_f1_val = -1.0
    epochs_no_improve = 0
    patience = 20 # This patience is for early stopping

    overall_best_f1 = -1.0
    overall_best_acc = -1.0
    overall_best_epoch = -1
    overall_best_cm = None

    for epoch in range(args.num_epochs): # Read num_epochs from args
        model.train()
        total_loss_train = 0

        for batch in train_loader:
            for k, v in batch.items():
                batch[k] = v.to(device)
            model.set_input(batch)
            model.optimize_parameters(epoch) # This calls optimizer.step() inside the model

            current_losses = model.get_current_losses()
            if 'emo_CE' in current_losses:
                total_loss_train += current_losses['emo_CE']

        avg_loss_train = total_loss_train / len(train_loader)

        val_loss, f1_w, f1_u, acc_w, acc_u, cm = eval(model, val_loader, device)

        # Step the learning rate scheduler based on validation loss
        scheduler.step(val_loss) # <--- ADD THIS LINE (or f1_w if you chose 'max' mode)

        current_lr = optimizer.param_groups[0]['lr'] # <--- ADD THIS LINE TO LOG CURRENT LR
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}, Train Avg Loss: {avg_loss_train:.4f}, Val Avg Loss: {val_loss:.4f}, "
                             f"Weighted F1: {f1_w:.4f}, Unweighted F1: {f1_u:.4f}, "
                             f"Weighted Acc: {acc_w:.4f}, Unweighted Acc: {acc_u:.4f}, Current LR: {current_lr:.8f}") # <--- MODIFIED LOGGING
        logger.info(f"Confusion Matrix:\n{cm}")

        if f1_w > best_f1_val:
            best_f1_val = f1_w
            epochs_no_improve = 0
            
            # Use opt.checkpoints_dir for model save path
            model_save_path = os.path.join(opt.checkpoints_dir, opt.name, f"{best_model_name_prefix}_best_f1.pth")
            # Ensure the directory exists before saving
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path) 
            logger.info(f"Saved best model with Weighted F1: {best_f1_val:.4f} at epoch {epoch + 1}.")
            
            overall_best_f1 = f1_w
            overall_best_acc = acc_w
            overall_best_epoch = epoch + 1
            overall_best_cm = cm
        else:
            epochs_no_improve += 1
            logger.info(f"No improvement in Weighted F1. Patience: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs due to no improvement for {patience} epochs.")
            break

    logger.info(f"Training complete. Overall Best Epoch: {overall_best_epoch}, Best Weighted F1: {overall_best_f1:.4f}, Best Weighted Acc: {overall_best_acc:.4f}")
    if overall_best_cm is not None:
        logger.info(f"Overall Best Confusion Matrix:\n{overall_best_cm}")

    return overall_best_f1, overall_best_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--labelcount', type=int, default=3)
    parser.add_argument('--track_option', type=str, required=True)
    parser.add_argument('--feature_max_len', type=int, required=True)
    parser.add_argument('--data_rootpath', type=str, required=True)
    parser.add_argument('--personalized_features_file', type=str)
    parser.add_argument('--audiofeature_method', type=str, choices=['mfccs', 'opensmile', 'wav2vec'], default='wav2vec')
    parser.add_argument('--videofeature_method', type=str, choices=['openface', 'resnet', 'densenet'], default='resnet')
    parser.add_argument('--splitwindow_time', type=str, default='5s')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    args.train_json = os.path.join(args.data_rootpath, 'Training', 'labels', 'Training_Validation_files.json')
    args.personalized_features_file = os.path.join(args.data_rootpath, 'Training', 'individualEmbedding', 'descriptions_embeddings_with_ids.npy')

    config = load_config('config.json')
    opt = Opt(config) # opt now contains all values from config.json

    # Transfer arguments from argparse to opt object for parameters controlled by CLI/bash script
    opt.emo_output_dim = args.labelcount
    opt.feature_max_len = args.feature_max_len
    opt.lr = args.lr
    opt.batch_size = args.batch_size # Ensure batch_size from args is used
    opt.num_epochs = args.num_epochs # Ensure num_epochs from args is used
    opt.device = args.device # Ensure device from args is used

    audio_path = os.path.join(args.data_rootpath, 'Training', args.splitwindow_time, 'Audio', args.audiofeature_method)
    video_path = os.path.join(args.data_rootpath, 'Training', args.splitwindow_time, 'Visual', args.videofeature_method)

    audio_files = [f for f in os.listdir(audio_path) if f.endswith('.npy')]
    if audio_files:
        opt.input_dim_a = np.load(os.path.join(audio_path, audio_files[0])).shape[1]
    else:
        raise FileNotFoundError(f"No .npy files found in {audio_path} to determine input_dim_a.")

    video_files = [f for f in os.listdir(video_path) if f.endswith('.npy')]
    if video_files:
        opt.input_dim_v = np.load(os.path.join(video_path, video_files[0])).shape[1]
    else:
        raise FileNotFoundError(f"No .npy files found in {video_path} to determine input_dim_v.")

    # Set up logging directory and logger
    cur_time = datetime.now().strftime('%Y-%m-%d-%H.%M.%S')
    opt.name = f"{args.splitwindow_time}_{args.labelcount}labels_{args.audiofeature_method}+{args.videofeature_method}_{cur_time}"
    logger_path = os.path.join(opt.log_dir, opt.name) # opt.log_dir now directly from config
    os.makedirs(logger_path, exist_ok=True)
    logger = get_logger(logger_path, 'result')

    # Print exact log file path
    log_file_timestamp = time.strftime('%Y-%m-%d-%H.%M.%S', time.localtime(time.time()))
    actual_log_file_name = f"result_{log_file_timestamp}.log"
    actual_log_file_path = os.path.join(logger_path, actual_log_file_name)
    print(f"INFO: Log file is being written to: {actual_log_file_path}")
    sys.stdout.flush()

    logger.info("Logger successfully initialized in train.py. This message should appear in the log file.")

    seed = 3407
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if opt.cuda_benchmark: # opt.cuda_benchmark now directly from config
            torch.backends.cudnn.benchmark = True

    if args.track_option == 'Track1':
        train_data_for_weights, _, _, _ = train_val_split1(args.train_json, val_ratio=0.1, random_seed=seed)
    else:
        train_data_for_weights, _, _, _ = train_val_split2(args.train_json, val_percentage=0.1, seed=seed)

    label_key = {2: "bin_category", 3: "tri_category", 5: "pen_category"}[args.labelcount]
    train_labels_for_weights = [sample[label_key] for sample in train_data_for_weights]

    class_weights_np = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels_for_weights), y=train_labels_for_weights)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float).to(args.device) # device still from args
    opt.class_weights = class_weights

    model = ourModelMPDDCross(opt) # Pass opt object, which now fully contains config params
    
    # Initialize optimizer (assuming your model.py has self.optimizer = ... set up,
    # or you'll need to define it here and pass to model for optimization)
    # Based on your train_model function calling model.optimize_parameters(epoch),
    # it seems the optimizer is handled inside the model.
    # However, for ReduceLROnPlateau, you need access to the optimizer object.
    # Let's assume model.optimizer exists after model instantiation or is set via a method.

    # If the optimizer is created inside ourModelMPDDCross's __init__ or setup method,
    # you'll need to expose it, e.g., model.optimizer.
    # Assuming 'model' object has an 'optimizer' attribute
    optimizer = model.optimizer # <--- This line is critical. Make sure your model exposes its optimizer.
                                # If not, you might need to change how optimizer is handled.

    # Initialize the learning rate scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',         # Monitor validation loss (minimize)
        factor=0.2,         # Halve the learning rate
        patience=10,        # Wait for 10 epochs of no improvement
        threshold=0.0001,   # Minimum change to count as improvement
        threshold_mode='rel',
        cooldown=0,
        min_lr=1e-7,        # Don't let LR go below this
        verbose=True        # Print messages when LR changes
    ) # <--- ADD THESE LINES

    logger.info(f"Training model")
    logger.info(f"Settings: labels={opt.emo_output_dim}, feature_len={opt.feature_max_len}, lr={opt.lr}")
    # Log these directly from opt, as they come from config
    logger.info(f"Hyperparameters: Transformer_head={opt.Transformer_head}, Transformer_layers={opt.Transformer_layers}, cls_layers={opt.cls_layers}, dropout_rate={opt.dropout_rate}, focal_weight={opt.focal_weight}, focal_gamma={opt.focal_gamma}, use_ICL={opt.use_ICL}, bn={opt.bn}, cuda_benchmark={opt.cuda_benchmark}") 

    best_model_name_prefix = f"best_model_run_{cur_time}" 

    try:
        # Pass optimizer and scheduler to train_model
        train_model(args.train_json, model, optimizer, scheduler, audio_path, video_path, opt.feature_max_len, best_model_name_prefix, seed) # <--- MODIFIED CALL
    except Exception as e:
        logger.exception("An unhandled exception occurred during training:")
        raise
    finally:
        if logger:
            for handler in logger.handlers[:]:
                try:
                    handler.flush()
                    handler.close()
                    logger.removeHandler(handler)
                except Exception as close_e:
                    print(f"Warning: Error while closing logger handler: {close_e}", file=sys.stderr)
        logging.shutdown()
        print(f"Training script finished. Check logs in: {logger_path}")
        sys.stdout.flush()