import argparse
import time
import json
from pathlib import Path
import sys
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from transformer import *
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class ExperimentLogger:
    """Lightweight experiment tracking for transformer training on GPU."""
    def __init__(self, experiment_name: str, log_dir: str = "experiments"):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.experiment_dir = self.log_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.start_time = time.time()
        self.metrics = {}
        self.config = {}
        
    def log_config(self, config: dict):
        """Log hyper-parameters and model configuration"""
        self.config.update(config)
        with open(self.experiment_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=2)
            
    def log_metrics(self, metrics: dict, step: int):
        """Log training/validation metrics"""
        wallclock_time = time.time() - self.start_time
        
        for metric_name, value in metrics.items():
            if metric_name not in self.metrics:
                self.metrics[metric_name] = []
            
            self.metrics[metric_name].append({
                'step': step,
                'value': float(value),
                'wallclock_time': wallclock_time
            })
        
        # Save after each log
        with open(self.experiment_dir / 'metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)    
            
    
    def plot_metrics(self):
        """Generate and save training curve plots"""
        import matplotlib.pyplot as plt
        
        metrics_to_plot = ['train_loss', 'val_loss', 'val_perplexity']
        available_metrics = [m for m in metrics_to_plot if m in self.metrics]
        
        if not available_metrics:
            return

        # Plot by steps
        fig, axes = plt.subplots(
            len(available_metrics),
            1,
            figsize=(10, 4 * len(available_metrics)),
        )
        if len(available_metrics) == 1:
            axes = [axes]
        for ax, metric_name in zip(axes, available_metrics):
            data = self.metrics[metric_name]
            steps = [d['step'] for d in data]
            values = [d['value'] for d in data]
            
            ax.plot(steps, values, marker='o', markersize=3, linewidth=1.5)
            ax.set_xlabel('Training Steps')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} vs Steps')
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(self.experiment_dir / 'metrics_vs_steps.png', dpi=150)
        plt.close()
        
        # Plot by wallclock time
        fig, axes = plt.subplots(len(available_metrics), 1,
                                 figsize=(10, 4 * len(available_metrics)))
        if len(available_metrics) == 1:
            axes = [axes]
            
        for ax, metric_name in zip(axes, available_metrics):
            data = self.metrics[metric_name]
            times = [d['wallclock_time'] / 3600 for d in data] # Convert to hours
            values = [d['value'] for d in data]
            
            ax.plot(times, values, marker='o', markersize=3, linewidth=1.5)
            ax.set_xlabel('Wallclock Time (hours)')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} vs Wallclock Time')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / 'metrics_vs_time.png', dpi=150)
        plt.close()
        
    def print_summary(self):
        """Print experiment summary"""
        total_time = time.time() - self.start_time
        print(f"\n{'='*60}")
        print(f"Experiment: {self.experiment_name}")
        print(f"{'='*60}")
        print(f"Total Time : {total_time/3600:.2f} hours ({total_time/60:.2f} minutes)")
        
        if 'train_loss' in self.metrics and self.metrics['train_loss']:
            final_train_loss = self.metrics['train_loss'][-1]['value']
            print(f"Final Training Loss: {final_train_loss:.4f}")
            
        if 'val_loss' in self.metrics and self.metrics['val_loss']:
            final_val_loss = self.metrics['val_loss'][-1]['value']
            best_val = min(d['value'] for d in self.metrics['val_loss'])
            print(f"Final Validation Loss: {final_val_loss:.4f}")
            print(f"Best Val Loss: {best_val:.4f}")
            
        if 'val_perplexity' in self.metrics and self.metrics['val_perplexity']:
            final_ppl = self.metrics['val_perplexity'][-1]['value']
            best_ppl = min(d['value'] for d in self.metrics['val_perplexity'])
            print(f"Final Perplexity: {final_ppl:.2f}")
            print(f"Best Perplexity: {best_ppl:.2f}")
            
        print(f"{'='*60}\n")


def get_experiment_name(args):
    """Generate a descriptive experiment name."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    name = f"d{args.d_model}_l{args.num_layers}_h{args.num_heads}_bs{args.batch_size}_{timestamp}"
    return name


def main():
    parser = argparse.ArgumentParser(description='Train Transformer Language Model')
    
    # Model hyperparameters
    parser.add_argument('--vocab_size', type=int, default=10000, help='Vocabulary size')
    parser.add_argument('--context_length', type=int, default=256, help='Context length')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=1344, help='Feed-forward dimension')
    parser.add_argument('--rope_theta', type=float, default=10000.0, help='RoPE theta parameter')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--max_iters', type=int, default=20000, help='Maximum training iterations')
    parser.add_argument('--eval_interval', type=int, default=1000, help='Evaluation interval')
    parser.add_argument('--learning_rate', type=float, default=6e-4, help='Maximum learning rate')
    parser.add_argument('--min_learning_rate', type=float, default=6e-5, help='Minimum learning rate')
    parser.add_argument('--warmup_iters', type=int, default=2000, help='Warmup iterations')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.95, help='Adam beta2')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping norm')
    
    # Data and logging
    parser.add_argument('--train_data', type=str, required=True, help='Path to training data')
    parser.add_argument('--val_data', type=str, required=True, help='Path to validation data')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--checkpoint_interval', type=int, default=1000, help='Evaluation interval')
    parser.add_argument('--resume_from', type=str, default=None, help='Checkpoint to resume from')
    parser.add_argument('--log_interval', type=int, default=1000, help='Logging interval')
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name (auto-generated if not provided)')
    
    # Performance optimization
    parser.add_argument('--compile', action='store_true', help='Use torch.compile for faster training (PyTorch 2.0+)')
    parser.add_argument('--dtype', type=str, default='float32', choices=['float32', 'float16', 'bfloat16'], 
                       help='Model dtype (use bfloat16 on modern GPUs for speed)')
    
    
    
    args = parser.parse_args()
    
    # Set device
    if torch.cuda.is_available():
        device = 'cuda'
        # Enable TF32 for faster training on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("Using MPS (Apple Silicon)")
    else:
        device = 'cpu'
        print("Using CPU (this will be slow!)")
    
    
    # Set dtype
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }
    dtype = dtype_map[args.dtype]
    if args.dtype == 'bfloat16' and not torch.cuda.is_available():
        print("Warning: bfloat16 not well supported on CPU/MPS, using float32")
        dtype = torch.float32
        
    # Generate experiment name
    if args.experiment_name is None:
        args.experiment_name = get_experiment_name(args)
        
    # Initialize experiment logger
    logger = ExperimentLogger(args.experiment_name, log_dir="experiments")
    logger.log_config(vars(args))
    print(f"\nExperiment: {args.experiment_name}")
    print(f"Logs will be saved to: experiments/{args.experiment_name}/\n")
        
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir) / args.experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    # Load data with memory mapping
    print("Loading training data...")
    train_data = np.load(args.train_data, mmap_mode='r')
    print("Loading validation data...")
    val_data = np.load(args.val_data, mmap_mode='r')
    
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
        dtype=dtype
    )
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    logger.config['num_parameters'] = num_params
    
    if args.compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2)
    )
    
    # Resume from checkpoint if specified
    start_iter = 0
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        start_iter = load_checkpoint(args.resume_from, model, optimizer)
        print(f"Resumed from iteration {start_iter}")
        
    # Training loop
    model.train()
    best_val_loss = float('inf')
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    print("Starting training...")
    
    training_start_time = time.time()
    
    for iteration in range(start_iter, args.max_iters):
        iter_start_time = time.time()
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
        
        with torch.amp.autocast(device_type=device.split(':')[0], dtype=dtype, enabled=(dtype != torch.float32)):
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
        
        iter_time = time.time() - iter_start_time
        # Logging
        if iteration % args.log_interval == 0:
            tokens_per_sec = (args.batch_size * args.context_length) / iter_time
            elapsed = time.time() - training_start_time
            metrics = {
                'train_loss': loss.item(),
                'learning_rate': lr
            }
            logger.log_metrics(metrics, iteration)
            print(f"Step {iteration:6d}/{args.max_iters} | "
                  f"Loss: {loss.item():.4f} | "
                  f"LR: {lr:.2e} | "
                  f"Tok/s: {tokens_per_sec:,.0f} | "
                  f"Time: {elapsed:.1f}s")
            
        # Evaluation
        if (iteration+1) % args.eval_interval == 0 or iteration == args.max_iters - 1:
            model.eval()
            val_losses = []
            with torch.no_grad():
                # Evaluate on multiple batches for more stable estimate
                num_val_batches = min(10, max(1, len(val_data) // (args.batch_size * args.context_length)))
                for _ in range(num_val_batches):
                    val_inputs, val_targets = get_batch(
                        val_data,
                        batch_size=args.batch_size,
                        context_length=args.context_length,
                        device=device
                    )
                    
                    with torch.amp.autocast(device_type=device.split(':')[0], dtype=dtype, enabled=(dtype != torch.float32)):
                        val_logits = model(val_inputs)
                        val_loss = cross_entropy_loss(
                            val_logits.view(-1, val_logits.size(-1)),
                            val_targets.view(-1)
                        )
                    val_losses.append(val_loss.item())
                    
            avg_val_loss = sum(val_losses) / len(val_losses)
            val_perplexity = np.exp(avg_val_loss)
            logger.log_metrics({
                'val_loss': avg_val_loss,
                'val_perplexity': val_perplexity
            }, iteration)
            print(f"{'='*60}")
            print(f"Step {iteration:6d} | Val Loss: {avg_val_loss:.4f} | Perplexity: {val_perplexity:.2f}")
            print(f"{'='*60}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_checkpoint_path = checkpoint_dir / 'best_model.pt'
                save_checkpoint(model, optimizer, iteration, best_checkpoint_path)
                print(f"New best model saved! (val_loss: {best_val_loss:.4f})")
            
            model.train()
            
        # Checkpointing
        if iteration % args.checkpoint_interval == 0 and iteration > 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_{iteration:06d}.pt'
            save_checkpoint(model, optimizer, iteration, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
            
    # Save final checkpoint
    final_checkpoint_path = checkpoint_dir / 'final_checkpoint.pt'
    save_checkpoint(model, optimizer, args.max_iters, final_checkpoint_path)
    print(f"Saved final checkpoint: {final_checkpoint_path}")
    
    # Generate plots and summary
    print("\nGenerating plots...")
    logger.plot_metrics()
    logger.print_summary()
    
    print(f"\nTraining complete! Results saved to experiments/{args.experiment_name}/")
    print(f"View plots: experiments/{args.experiment_name}/metrics_*.png")
    print(f"View metrics: experiments/{args.experiment_name}/metrics.json")

if __name__ == "__main__":
    main()
    