"""Both models classify the final query marker; no outcome tokens are predicted."""
import math
import re
import torch
from torch import nn
from torch.nn import functional as F


def soft_ce(logits, targets):
    if targets.shape != logits.shape or not torch.allclose(targets.sum(-1), torch.ones_like(targets[:, 0])):
        raise ValueError("Soft targets must sum to one")
    if not torch.isfinite(targets).all() or torch.any(targets < 0):
        raise ValueError("Invalid soft targets")
    return -(targets * F.log_softmax(logits.float(), dim=-1)).sum(-1).mean()


class RotaryAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        if width % heads or (width // heads) % 2:
            raise ValueError("RoPE head width must be even")
        self.heads, self.dim = heads, width // heads
        self.qkv = nn.Linear(width, width * 3, bias=False)
        self.out = nn.Linear(width, width, bias=False)

    def rotate(self, x):
        length, dim = x.shape[-2:]
        theta = torch.arange(length, device=x.device).float()[:, None] / (
            10000 ** (torch.arange(0, dim, 2, device=x.device).float() / dim))
        a, b = x[..., ::2], x[..., 1::2]
        return torch.stack((a * theta.cos() - b * theta.sin(),
                            a * theta.sin() + b * theta.cos()), dim=-1).flatten(-2)

    def forward(self, x, mask):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).view(batch, length, 3, self.heads, self.dim).permute(2, 0, 3, 1, 4)
        causal = torch.ones(length, length, device=x.device, dtype=torch.bool).tril()
        allowed = causal[None, None, :, :] & mask[:, None, None, :].bool()
        y = F.scaled_dot_product_attention(self.rotate(q), self.rotate(k), v, attn_mask=allowed)
        return self.out(y.transpose(1, 2).reshape(batch, length, width))


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attn = RotaryAttention(width, heads)
        self.mlp = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))

    def forward(self, x, mask):
        x = x + self.attn(self.n1(x), mask)
        return x + self.mlp(self.n2(x))


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size=257, width=64, layers=2, heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, width)
        self.blocks = nn.ModuleList([Block(width, heads) for _ in range(layers)])
        self.norm, self.head = nn.LayerNorm(width), nn.Linear(width, 3)

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x, attention_mask)
        # Right padding only, checked by the common collator.
        indices = attention_mask.sum(-1).long() - 1
        return self.head(self.norm(x[torch.arange(len(x), device=x.device), indices]))


def lora_config(num_layers, rank=8, alpha=16, last_layers=4):
    from peft import LoraConfig
    if not 1 <= last_layers <= num_layers:
        raise ValueError("Invalid LoRA layer count")
    # AutoModel exposes layers.N directly; PEFT's layers_pattern matcher expects an extra prefix.
    targets = [f"layers.{i}.self_attn.{projection}" for i in range(num_layers - last_layers, num_layers)
               for projection in ("q_proj", "k_proj", "v_proj", "o_proj")]
    return LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0., bias="none", target_modules=targets)


class QwenProbabilityModel(nn.Module):
    def __init__(self, model_id, revision, rank=8, alpha=16, last_layers=4,
                 attention="sdpa", gradient_checkpointing=True):
        super().__init__()
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Pin the model to a 40-character immutable revision")
        from transformers import AutoModel
        from peft import LoraConfig, get_peft_model
        base = AutoModel.from_pretrained(model_id, revision=revision, torch_dtype=torch.bfloat16,
                                         attn_implementation=attention, trust_remote_code=False)
        base.config.use_cache = False
        n = base.config.num_hidden_layers
        if not 1 <= last_layers <= n:
            raise ValueError("Invalid LoRA layer count")
        config = lora_config(n, rank, alpha, last_layers)
        self.backbone = get_peft_model(base, config)
        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.backbone.enable_input_require_grads()
        self.head = nn.Linear(base.config.hidden_size, 3, dtype=torch.float32)
        self.context_limit = base.config.max_position_embeddings

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        indices = attention_mask.sum(-1).long() - 1
        return self.head(hidden[torch.arange(len(hidden), device=hidden.device), indices].float())


def collate(episodes, pad_id=0, device="cpu"):
    if not episodes or any(not ep.input_ids for ep in episodes):
        raise ValueError("Empty episodes")
    length = max(len(ep.input_ids) for ep in episodes)
    ids = torch.full((len(episodes), length), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, ep in enumerate(episodes):
        ids[i, :len(ep.input_ids)] = torch.tensor(ep.input_ids, device=device)
        mask[i, :len(ep.input_ids)] = 1
    return {"input_ids": ids, "attention_mask": mask}


class Predictor:
    def __init__(self, model, device="cpu", pad_id=0):
        self.model, self.device, self.pad_id = model.to(device).eval(), device, pad_id
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.inference_mode()
    def predict_proba(self, episode):
        return self.model(**collate([episode], self.pad_id, self.device)).float().softmax(-1)[0].cpu().tolist()


    @torch.inference_mode()
    def predict_many(self, episodes, batch_size=8):
        order=sorted(range(len(episodes)),key=lambda i:len(episodes[i].input_ids))
        result=[None]*len(episodes)
        for start in range(0,len(order),batch_size):
            indices=order[start:start+batch_size]
            batch=collate([episodes[i] for i in indices],self.pad_id,self.device)
            probabilities=self.model(**batch).float().softmax(-1).cpu().tolist()
            for i,p in zip(indices,probabilities):result[i]=p
        return result
