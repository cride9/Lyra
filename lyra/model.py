"""V1-style encoder/decoder, jointly optimized, with raw recent event context."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .config import ModelConfig


def positions(length, width, device, offset=0):
    t = torch.arange(offset, offset + length, device=device, dtype=torch.float32)[:, None]
    scale = torch.exp(torch.arange(0, width, 2, device=device).float() * (-math.log(10000) / width))
    p = torch.zeros(length, width, device=device)
    p[:, 0::2], p[:, 1::2] = (t * scale).sin(), (t * scale).cos()
    return p


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads, self.dim, self.dropout = c.heads, c.width // c.heads, c.dropout
        self.q, self.k, self.v, self.out = [nn.Linear(c.width, c.width) for _ in range(4)]

    def split(self, x):
        return x.view(x.shape[0], x.shape[1], self.heads, self.dim).transpose(1, 2)

    def forward(self, x, source=None, valid=None, causal=False, cache=None, offset=0):
        q = self.split(self.q(x))
        static = source is not None
        if static and cache is not None:
            k, v = cache
        else:
            src = x if source is None else source
            k, v = self.split(self.k(src)), self.split(self.v(src))
            if cache is not None:
                k, v = torch.cat((cache[0], k), 2), torch.cat((cache[1], v), 2)
        mask = valid[:, None, None, :] if valid is not None else None
        if causal and (mask is not None or (offset and x.shape[1] > 1)):
            cm = torch.arange(k.shape[2], device=x.device)[None] <= torch.arange(offset, offset + x.shape[1], device=x.device)[:, None]
            mask = cm[None, None] if mask is None else mask & cm[None, None]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                is_causal=causal and mask is None and offset == 0,
                dropout_p=self.dropout if self.training else 0.)
        return self.out(y.transpose(1, 2).contiguous().view(x.shape)), (k, v)


class EncoderBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attn = Attention(c)
        self.n1, self.n2 = nn.LayerNorm(c.width), nn.LayerNorm(c.width)
        self.ff = nn.Sequential(nn.Linear(c.width, c.feedforward), nn.GELU(),
                                nn.Dropout(c.dropout), nn.Linear(c.feedforward, c.width))
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.n1(x))[0])
        return x + self.drop(self.ff(self.n2(x)))


class DecoderBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.self_attn, self.cross_attn = Attention(c), Attention(c)
        self.n1, self.n2, self.n3 = [nn.LayerNorm(c.width) for _ in range(3)]
        self.ff = nn.Sequential(nn.Linear(c.width, c.feedforward), nn.GELU(),
                                nn.Dropout(c.dropout), nn.Linear(c.feedforward, c.width))
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, memory, memory_valid, caches=None, offset=0):
        a, sa = self.self_attn(self.n1(x), causal=True,
                             cache=None if caches is None else caches[0], offset=offset)
        x = x + self.drop(a)
        a, ca = self.cross_attn(self.n2(x), memory, valid=memory_valid,
                              cache=None if caches is None else caches[1])
        x = x + self.drop(a)
        return x + self.drop(self.ff(self.n3(x))), (sa, ca)


class Lyra(nn.Module):
    def __init__(self, vocab_size, config=None):
        super().__init__()
        self.config = c = config or ModelConfig()
        self.frontend = nn.Sequential(nn.Conv1d(128, c.width, 5, padding=2), nn.GELU(),
                nn.Conv1d(c.width, c.width, 3, stride=2, padding=1), nn.GELU(),
                nn.Conv1d(c.width, c.width, 3, stride=2, padding=1), nn.GELU())
        self.encoder = nn.ModuleList(EncoderBlock(c) for _ in range(c.encoder_layers))
        self.decoder = nn.ModuleList(DecoderBlock(c) for _ in range(c.decoder_layers))
        self.encoder_norm, self.decoder_norm = nn.LayerNorm(c.width), nn.LayerNorm(c.width)
        self.embedding = nn.Embedding(vocab_size, c.width, padding_idx=0)
        self.timing_projection, self.beat_head = nn.Linear(3, c.width), nn.Linear(c.width, 2)
        self.summary_projection = nn.Linear(68, c.width)
        self.modality = nn.Embedding(2, c.width)
        self.apply(self.initialize)

    @staticmethod
    def initialize(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=.02)
            if getattr(m, 'bias', None) is not None:
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.Embedding) and m.padding_idx is not None:
                with torch.no_grad():
                    m.weight[m.padding_idx].zero_()

    def encode(self, mel, phase=None, summary=None, summary_valid=None):
        x = self.frontend(mel).transpose(1, 2)
        x = x + positions(x.shape[1], x.shape[2], x.device).to(x.dtype)
        for layer in self.encoder:
            x = checkpoint(layer, x, use_reentrant=False) if self.training and self.config.checkpointing else layer(x)
        x = self.encoder_norm(x)
        # Timing never sees reference phase, past-map summaries, or target tokens.
        beats = F.interpolate(self.beat_head(x).transpose(1, 2), size=mel.shape[-1],
                              mode='linear', align_corners=False).transpose(1, 2)
        if phase is not None:
            pooled = F.interpolate(phase.transpose(1, 2), size=x.shape[1], mode='linear', align_corners=False).transpose(1, 2)
            x = x + self.timing_projection(pooled)
        x = x + self.modality.weight[0]
        valid = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        if summary is not None:
            s = self.summary_projection(summary) + self.modality.weight[1]
            x = torch.cat((x, s), 1)
            valid = torch.cat((valid, summary_valid), 1)
        return x, valid, beats

    def hidden(self, tokens, memory, memory_valid):
        if tokens.shape[1] > self.config.max_tokens:
            raise ValueError('Lyra kontextuskeret túllépve.')
        x = self.embedding(tokens)
        x = x + positions(x.shape[1], x.shape[2], x.device).to(x.dtype)
        for layer in self.decoder:
            def block(a, m, mask, module=layer):
                return module(a, m, mask)[0]
            x = checkpoint(block, x, memory, memory_valid, use_reentrant=False) if self.training and self.config.checkpointing else block(x, memory, memory_valid)
        return self.decoder_norm(x)

    def project(self, hidden):
        return F.linear(hidden, self.embedding.weight)

    def forward(self, b):
        memory, valid, beats = self.encode(b['mel'], b.get('phase'), b.get('summary'), b.get('summary_valid'))
        return self.hidden(b['tokens'], memory, valid), beats

    @torch.no_grad()
    def step(self, tokens, memory, memory_valid, caches=None, offset=0):
        if offset + tokens.shape[1] > self.config.max_tokens:
            raise ValueError('Lyra kontextuskeret túllépve.')
        x = self.embedding(tokens)
        x = x + positions(x.shape[1], x.shape[2], x.device, offset).to(x.dtype)
        result = []
        for i, layer in enumerate(self.decoder):
            x, cache = layer(x, memory, memory_valid, None if caches is None else caches[i], offset)
            result.append(cache)
        return self.project(self.decoder_norm(x[:, -1:])), result


def token_loss(model, hidden, labels, chunk=128, indices=None):
    if indices is None:
        selected = labels != -100
        h, y = hidden[selected], labels[selected]
    else:
        h = hidden.flatten(0, 1).index_select(0, indices)
        y = labels.flatten().index_select(0, indices)
    total = hidden.new_zeros((), dtype=torch.float32)
    def part(a, b):
        return F.cross_entropy(model.project(a).float(), b, reduction='sum')
    for start in range(0, len(y), chunk):
        a, b = h[start:start + chunk], y[start:start + chunk]
        total = total + (checkpoint(part, a, b, use_reentrant=False) if torch.is_grad_enabled() else part(a, b))
    return total, len(y)
