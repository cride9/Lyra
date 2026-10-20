from dataclasses import dataclass
import os
from pathlib import Path

HOME = Path(os.environ.get('LYRA_HOME', 'F:/osu_dataset/_lyra'))
CACHE = Path(os.environ.get('LYRA_CACHE', 'F:/osu_cache/lyra'))
SOURCE = Path('F:/osu_dataset/_training/dataset-full/manifest.json')


@dataclass
class ModelConfig:
    width: int = 640
    encoder_layers: int = 8
    decoder_layers: int = 12
    heads: int = 10
    feedforward: int = 2304
    dropout: float = .1
    audio_ms: int = 64000
    history_ms: int = 32000
    output_ms: int = 24000
    history_tokens: int = 2048
    max_tokens: int = 6144
    summary_tokens: int = 128
    checkpointing: bool = True

    def __post_init__(self):
        if self.width % self.heads or self.max_tokens <= self.history_tokens + 512:
            raise ValueError('Érvénytelen modellméret vagy kontextuskeret.')
        if self.history_ms + self.output_ms > self.audio_ms or self.audio_ms > 64000:
            raise ValueError('A hangablaknak a teljes előzményt és kimenetet tartalmaznia kell.')

    @classmethod
    def tiny(cls):
        return cls(width=48, encoder_layers=1, decoder_layers=2, heads=3,
                   feedforward=128, dropout=0, max_tokens=1024, history_tokens=256,
                   summary_tokens=16, checkpointing=False)


@dataclass
class TrainConfig:
    model: str = 'lyra'
    cycles: int = 3
    effective_batch: int = 32
    batch_size: int = 4
    workers: int = 4
    ram_cache_gib: float = 8
    disk_cache_gb: float = 180
    min_free_gib: float = 8
    lr: float = 3e-4
    warmup: int = 1000
    validate_every: int = 500
    validation_windows: int = 64
    sample_every: int = 1000
    sample_map_id: str = ''
    sample_audio_example: str = ''
    checkpoint_minutes: float = 15
    memory_fraction: float = .80
    seed: int = 42
    device: str = 'auto'
    cpu_threads: int = 2
    steps: int = 0  # 0: all windows, for the requested number of cycles.

    def __post_init__(self):
        if self.batch_size < 1 or self.effective_batch % self.batch_size:
            raise ValueError('Az effektív batch legyen a fizikai batch többszöröse.')
        if self.cycles < 1 or self.workers < 0 or not 0 < self.memory_fraction <= .9:
            raise ValueError('Érvénytelen adatkör-, worker- vagy memóriabeállítás.')
        if self.lr <= 0 or self.warmup < 1 or self.validate_every < 1 or self.validation_windows < 1:
            raise ValueError('Érvénytelen tanítási beállítás.')
        if self.sample_every < 0:
            raise ValueError('A példagenerálás gyakorisága nem lehet negatív.')
