
import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.getcwd())

from src.config import get_default_config
from src.models.model import PrototypeNetwork
from src.training.losses import PTaRLLoss

def check_decoder_gradients():
    # 1. Setup options
    config = get_default_config()
    device = 'cpu'
    
    print(f"Config Reconstruction Weight: {config.ptarl_weights['reconstruction_weight']}")
    
    # 2. Create model
    model = PrototypeNetwork(
        n_numerical=config.n_numerical,
        categorical_cardinalities=config.get_cardinality_list(),
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ffn=config.d_ffn,
        n_global_prototypes=config.n_global_prototypes,
        decoder_hidden_dim=config.decoder_hidden_dim
    ).to(device)
    
    # 3. Set Phase 2
    model.set_second_phase()
    embeddings = torch.randn(10, config.d_model)
    model.initialize_global_prototypes(embeddings)
    
    # 4. Create dummy data
    batch_size = 4
    x_num = torch.randn(batch_size, config.n_numerical).to(device)
    x_cat = torch.randint(0, 1, (batch_size, config.n_categorical)).to(device)
    targets = torch.randint(0, 2, (batch_size,)).float().to(device)
    
    # 5. Zero grad
    model.zero_grad()
    
    # 6. Forward pass (Simulate Trainer Phase 2: return_all=False)
    print("Running forward pass (return_all=False)...")
    outputs = model(x_num, x_cat, return_all=False)
    
    if 'num_recon' in outputs:
        print("[WARNING] num_recon is present in outputs!")
    else:
        print("[OK] num_recon is NOT in outputs.")
    
    # 7. Compute Loss
    print("Computing loss...")
    # Manually construct loss with current config weights (reconstruction_weight should be 0)
    ptarl_args = config.ptarl_weights.copy()
    ptarl_args['n_classes'] = config.n_classes
    criterion = PTaRLLoss(**ptarl_args)
    
    losses = criterion(
        outputs, targets,
        model.global_prototype_layer,
        x_num=x_num,
        x_cat=x_cat
    )
    
    print(f"Total Loss: {losses['total'].item()}")
    print(f"Reconstruction Loss: {losses['reconstruction'].item()}")
    
    # 8. Backward
    print("Running backward pass...")
    losses['total'].backward()
    
    # 9. Check gradients
    decoder_grad = False
    for name, param in model.decoder.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 0:
                print(f"Decoder param {name} has gradient norm: {grad_norm}")
                decoder_grad = True
                break
        
    if not decoder_grad:
        print("\n[SUCCESS] Decoder has NO gradients in Phase 2.")
    else:
        print("\n[FAILURE] Decoder still receives gradients!")

if __name__ == "__main__":
    check_decoder_gradients()
