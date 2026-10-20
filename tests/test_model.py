from dataclasses import replace
import torch
from lyra.config import ModelConfig
from lyra.model import Lyra, token_loss
from lyra.tokenizer import Tokenizer


def test_cached_decoder_matches_full_including_cached_multi_token_blocks():
    torch.manual_seed(5)
    model = Lyra(len(Tokenizer()), ModelConfig.tiny()).eval()
    tokens = torch.randint(1, len(Tokenizer()), (1, 21))
    memory = torch.randn(1, 12, model.config.width)
    valid = torch.ones(1, 12, dtype=torch.bool)
    with torch.no_grad():
        expected = model.project(model.hidden(tokens, memory, valid))
        caches, values = None, []
        for i in range(tokens.shape[1]):
            value, caches = model.step(tokens[:, i:i + 1], memory, valid, caches, i)
            values.append(value)
        torch.testing.assert_close(torch.cat(values, 1), expected, atol=2e-5, rtol=2e-4)
        _, cache = model.step(tokens[:, :7], memory, valid)
        last, _ = model.step(tokens[:, 7:12], memory, valid, cache, 7)
        torch.testing.assert_close(last, expected[:, 11:12], atol=2e-5, rtol=2e-4)


def test_future_tokens_do_not_change_past_and_audio_receives_mapping_gradients():
    torch.manual_seed(7)
    model = Lyra(len(Tokenizer()), replace(ModelConfig.tiny(), checkpointing=True)).train()
    b = dict(mel=torch.randn(1, 128, 64), phase=torch.zeros(1, 64, 3),
             tokens=torch.randint(1, len(Tokenizer()), (1, 18)))
    hidden, _ = model(b)
    labels = torch.randint(1, len(Tokenizer()), (1, 18))
    ce, n = token_loss(model, hidden, labels)
    (ce / n).backward()
    assert model.frontend[0].weight.grad.abs().sum() > 0
    assert model.encoder[0].attn.q.weight.grad.abs().sum() > 0
    model.eval()
    memory, valid, _ = model.encode(b['mel'])
    other = b['tokens'].clone(); other[:, 10:] = 123
    with torch.no_grad():
        a = model.hidden(b['tokens'], memory, valid)
        c = model.hidden(other, memory, valid)
    torch.testing.assert_close(a[:, :10], c[:, :10])


def test_beat_head_cannot_see_ground_truth_phase_or_map_memory():
    model = Lyra(len(Tokenizer()), ModelConfig.tiny()).eval()
    mel = torch.randn(1, 128, 80)
    with torch.no_grad():
        _, _, a = model.encode(mel, torch.zeros(1, 80, 3))
        _, _, b = model.encode(mel, torch.ones(1, 80, 3), torch.randn(1, 3, 68), torch.ones(1, 3, dtype=torch.bool))
    torch.testing.assert_close(a, b)


def test_production_model_size_and_context():
    with torch.device('meta'):
        model = Lyra(len(Tokenizer()))
    assert sum(p.numel() for p in model.parameters()) == 120612482
    assert model.config.output_ms == 24000
    assert model.config.history_ms == 32000
    assert model.config.max_tokens == 6144
