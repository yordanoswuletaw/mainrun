# Comprehensive Optimization Report for Mainrun

## Executive Summary
Based on analysis of 11 training runs, the best validation loss achieved was **1.283065** (vs baseline 1.754). The optimal configuration used dropout=0.05, lr=1e-4, weight_decay=0.01. Current run is underperforming with validation loss of 2.091732.

## Current Issues & Optimization Opportunities

### 1. **Immediate Critical Fix: OneCycleLR Configuration**
The current OneCycleLR settings are too aggressive:
```python
# Current (problematic):
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=1e-3, total_steps=max_steps,
    pct_start=0.5, anneal_strategy='cos', div_factor=10.0
)

# Recommended:
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=5e-4, total_steps=max_steps,
    pct_start=0.2, anneal_strategy='cos', 
    div_factor=5.0, final_div_factor=100.0
)
```
- **max_lr=1e-3** is 10x higher than the base lr (1e-4), causing instability
- **pct_start=0.5** spends too long in warmup
- Missing **final_div_factor** for proper cooldown

### 2. **Model Architecture Optimizations**

#### A. Implement Scaled Dot-Product Attention (SDPA)
Replace manual attention computation with PyTorch's optimized SDPA:
```python
def forward(self, x: torch.Tensor):
    B, T, C = x.size()
    qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).transpose(1, 3)
    q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]
    
    # Use SDPA instead of manual computation
    y = F.scaled_dot_product_attention(
        q, k, v, attn_mask=None,
        dropout_p=self.attn_drop.p if self.training else 0.0,
        is_causal=True
    )
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    return self.resid_drop(self.proj(y))
```

#### B. Implement SwiGLU MLP
Replace standard MLP with SwiGLU activation:
```python
class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = int(8/3 * cfg.d_model)  # SwiGLU hidden dim
        self.w1 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w2 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.w3 = nn.Linear(hidden, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        
    def forward(self, x):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))
```

#### C. RMSNorm Instead of LayerNorm
```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * x / norm
```

### 3. **Training Loop Optimizations**

#### A. Parameter-Grouped Weight Decay
```python
# Separate parameters that should/shouldn't have weight decay
def configure_optimizers(model, weight_decay, learning_rate, betas=(0.9, 0.95)):
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if param.ndim == 1 or name.endswith('.bias') or 'ln' in name or 'norm' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    return torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=learning_rate, betas=betas)
```

#### B. Gradient Accumulation (if memory permits)
```python
gradient_accumulation_steps = 2
for i, (xb, yb) in enumerate(batch_iterator):
    _, loss = model(xb, yb)
    loss = loss / gradient_accumulation_steps
    loss.backward()
    
    if (i + 1) % gradient_accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        opt.zero_grad(set_to_none=True)
```

#### C. Mixed Precision Training
```python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    _, loss = model(xb, yb)
scaler.scale(loss).backward()
scaler.unscale_(opt)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(opt)
scaler.update()
```

### 4. **Tokenization Improvements**

#### A. Increase Vocabulary Size
```python
vocab_size: int = 32_000  # From 16_000
```
- Reduces out-of-vocabulary tokens
- Better representation of technical terms in Hacker News

#### B. Add Pre-tokenization Rules
```python
# Add custom pre-tokenization for URLs, mentions, etc.
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.Split(pattern=r'(?i)https?://\S+', behavior='isolated'),
    pre_tokenizers.ByteLevel(add_prefix_space=False)
])
```

### 5. **Hyperparameter Recommendations**

Based on log analysis:
```python
@dataclass
class Hyperparameters:
    # Architecture
    block_size: int = 128      # Increase from 64 for better context
    batch_size: int = 64       # Keep as is
    vocab_size: int = 32_000   # Increase from 16_000
    n_layer: int = 6           # Keep as is
    n_head: int = 8            # Keep as is
    d_model: int = 512         # Keep as is
    
    # Regularization
    dropout: float = 0.05      # Optimal based on analysis
    weight_decay: float = 0.01 # Optimal based on analysis
    
    # Optimization
    lr: float = 1e-4           # Keep as is
    betas: tuple = (0.9, 0.95) # More aggressive than default
    
    # Training
    evals_per_epoch: int = 5   # Increase from 3 for better monitoring
    gradient_clip: float = 1.0 # Already implemented
```

### 6. **Advanced Techniques**

#### A. Exponential Moving Average (EMA)
```python
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1 - self.decay
                )
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
```

#### B. Learning Rate Warmup with Cosine Decay
```python
def get_lr(step, warmup_steps=500, max_steps=1883, max_lr=5e-4, min_lr=1e-5):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    elif step < max_steps:
        decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)
    else:
        return min_lr
```

### 7. **Data Augmentation**

#### A. Token Masking
```python
def mask_tokens(input_ids, mask_prob=0.15, vocab_size=32000):
    """Randomly mask tokens for better generalization"""
    mask = torch.rand(input_ids.shape) < mask_prob
    masked_ids = input_ids.clone()
    masked_ids[mask] = torch.randint(0, vocab_size, (mask.sum(),))
    return masked_ids
```

#### B. Sequence Augmentation
```python
def augment_sequence(titles, augment_prob=0.1):
    """Randomly shuffle/drop titles during training"""
    if random.random() < augment_prob:
        # Shuffle order
        indices = torch.randperm(len(titles))
        titles = [titles[i] for i in indices[:int(len(titles)*0.95)]]
    return titles
```

## Implementation Priority

1. **Critical (Do First)**:
   - Fix OneCycleLR configuration
   - Implement parameter-grouped weight decay
   - Adjust dropout to 0.05

2. **High Impact**:
   - Implement SDPA
   - Add EMA
   - Implement SwiGLU MLP

3. **Medium Impact**:
   - Increase block_size to 128
   - Increase vocab_size to 32000
   - Add RMSNorm

4. **Nice to Have**:
   - Data augmentation
   - Mixed precision (if GPU available)
   - Custom learning rate schedule

## Expected Results
With these optimizations, you should expect:
- Validation loss: 1.25-1.28 (improvement from current 2.09)
- Training stability: Significant improvement
- Training speed: 10-20% faster with SDPA
- Better generalization with proper regularization

## Monitoring Recommendations
1. Log learning rate at each step
2. Track gradient norms
3. Monitor attention pattern entropy
4. Save checkpoints at each validation
