import argparse
import time
import json
from pathlib import Path
from transformer import *
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def main():
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--vocab_size', type=int, default=50257, help='Vocabulary size')
    parser.add_argument('--context_length', type=int, default=30, help='Context length')
    parser.add_argument('--d_model', type=int, default=48, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=307, help='Feed-forward dimension')
    parser.add_argument('--rope_theta', type=float, default=10000.0, help='RoPE theta parameter')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--max_iters', type=int, default=2000, help='Maximum training iterations')
    parser.add_argument('--eval_interval', type=int, default=100, help='Evaluation interval')
    parser.add_argument('--learning_rate', type=float, default=6e-4, help='Maximum learning rate')
    parser.add_argument('--min_learning_rate', type=float, default=6e-5, help='Minimum learning rate')
    parser.add_argument('--warmup_iters', type=int, default=2000, help='Warmup iterations')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping norm')
    
    # Data and logging
    parser.add_argument('--train_data', type=str, required=True, help='Path to training data')
    parser.add_argument('--val_data', type=str, required=True, help='Path to validation data')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--checkpoint_interval', type=int, default=50, help='Evaluation interval')
    parser.add_argument('--resume_from', type=str, default=None, help='Checkpoint to resume from')
    parser.add_argument('--log_interval', type=int, default=1, help='Logging interval')
    
    args = parser.parse_args()
    
    # Set device
    device = 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Save training configuration
    with open(checkpoint_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
        
    # Load data with memory mapping
    train_data = np.memmap(args.train_data, dtype=np.uint16, mode='r')
    print("Loading validation data...")
    val_data = np.memmap(args.val_data, dtype=np.uint16, mode='r')
    
    print(f"Training tokens: {len(train_data):,}")
    print(f"Validation tokens: {len(val_data):,}")
    
    # Initialize model
    print("Initializing model...")
    
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
        dtype=torch.float32
    )
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )
    
    # Resume from checkpoint if specified
    start_iter = 0
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        start_iter = load_checkpoint(args.resume_from, model, optimizer)
        print(f"Resumed from iteration {start_iter}")
        
    # Training loop
    model.train()
    train_losses = []
    val_losses = []
    
    print("Starting training...")
    for iteration in range(start_iter, args.max_iters):
        # Set learning rate
        lr = cosine_learning_rate_schedule(
            iteration,
            max_learning_rate=args.learning_rate,
            min_learning_rate=args.min_learning_rate,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.max_iters
        )
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        # Get batch
        inputs, targets = get_batch(
            train_data,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=device
        )
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(inputs)
        
        # Compute loss (reshape for cross entropy)
        loss = cross_entropy_loss(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if args.grad_clip > 0:
            gradient_clipping(model.parameters(), args.grad_clip)
            
        # Optimizer step
        optimizer.step()
        
        # Logging
        if iteration % args.log_interval == 0:
            train_losses.append(loss.item())
            print(f"Iter {iteration:6d} | Loss: {loss.item():.4f} | LR: {lr:.2e}")
            
        # Evaluation
        if iteration % args.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                val_inputs, val_targets = get_batch(
                    val_data,
                    batch_size=args.batch_size,
                    context_length=args.context_length,
                    device=device
                )
                val_logits = model(val_inputs)
                val_loss = cross_entropy_loss(
                    val_logits.view(-1, val_logits.size(-1)),
                    val_targets.view(-1)
                )
                val_losses.append(val_loss.item())
                print(f"Iter {iteration:6d} | Val Loss: {val_loss.item():.4f}")
            model.train()
            
        # Checkpointing
        if iteration % args.checkpoint_interval == 0 and iteration > start_iter:
            checkpoint_path = checkpoint_dir / f'checkpoint_{iteration:06d}.pt'
            save_checkpoint(model, optimizer, iteration, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
            
    # Save final checkpoint
    final_checkpoint_path = checkpoint_dir / 'final_checkpoint.pt'
    save_checkpoint(model, optimizer, args.max_iters, final_checkpoint_path)
    print(f"Saved final checkpoint: {final_checkpoint_path}")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_losses)
    plt.title('Validation Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(checkpoint_dir / 'training_curves.png')
    plt.show()

if __name__ == "__main__":
    main()
    