---
model_name: Lyra v2.5
model_status: training-in-progress
model_card_date: 2026-10-20
tags:
  - osu-standard
  - beatmap-generation
  - audio-to-event-sequence
  - encoder-decoder
  - transformer
  - trained-from-scratch
  - experimental
---

# Lyra v2.5 — Model Card

**A 120.6M-parameter, jointly trained audio-to-osu!standard beatmap model.**

Lyra turns music into structured osu!standard hit objects and timing estimates. Its audio frontend, Transformer encoder, autoregressive decoder, and beat/downbeat head are trained together from scratch. The mapping objective backpropagates through the audio encoder.

**Status: training in progress.** This card documents the local `main` run, its actual configuration, the prepared corpus, and observed intermediate results. It does not describe a completed or publicly released checkpoint.

**Training snapshot:** 20 October 2026, **11:03:30 Europe/Budapest** / **09:03:30 UTC**, at optimizer update **6,089**. The read-only metrics snapshot was captured at 09:03:45 UTC. Storage and runtime metadata were inspected separately a few minutes later. All “current” results below refer to this fixed snapshot.

The machine-readable [evidence snapshot](model-card/evidence-2026-10-20.json) records the configurations, dataset statistics, validation history, sample metadata, environment, and source-code SHA-256 hashes used for this card.

Historical diagnostic messages in that snapshot are presented in English. Its measurements, identifiers, and source-code hashes still refer to the original audit; the later English-language update does not change the training snapshot.

## 1. Model identity and intended use

| Field | Description |
|---|---|
| Model name | **Lyra v2.5** |
| Project | Local Lyra audio-to-beatmap project |
| Task | Conditional music-to-osu!standard event generation, with audio-based timing estimation |
| Architecture | Convolutional audio frontend + bidirectional Transformer encoder + causal Transformer decoder |
| Total parameters | **120,612,482** |
| Trainable parameters | **120,612,482** |
| Initialization | From scratch; no pretrained audio or language-model weights are loaded by this training path |
| Training method | Supervised teacher forcing with a joint event-token and beat/downbeat objective |
| Input | Music, target star rating, optional style and difficulty controls; optional BPM/offset or reference timing |
| Output | osu!standard `.osu` beatmaps packaged with the input audio and generation metadata in an importable `.osz` |
| Ruleset | osu!standard / Mode 0 |
| Development status | Experimental; first full training run still underway |
| Public checkpoint / model repository | No published release is established by the inspected local artifacts |
| License | No release license is specified in the inspected project root |

The intended use is assisted beatmap creation, experimentation with audio-conditioned mapping, and inspection of model learning. Generated maps require human review for musical emphasis, readability, geometry, difficulty, and playability.

**Version naming:** “v2.5” is the model name requested for this card. The inspected Python package still declares `lyra-osu==1.0.0`, the checkpoint format is `lyra-1`, and the tokenizer format is `lyra-events-v1-64s-ms-xy2`. These are separate identifiers. This card does not imply that a package or checkpoint-format migration to a literal `2.5` has occurred.

The current joint Lyra model is distinct from the older, separately trained V1/V2 audio and mapper models. Their checkpoints and evaluation results are not interchangeable.

## 2. Architecture

The forward path is:

~~~text
Music
  -> 22,050 Hz mono PCM
  -> 128-band log-mel spectrogram
  -> convolutional frontend, 4x temporal downsampling
  -> 8-layer audio Transformer encoder
       -> audio-only beat/downbeat head
       -> timing-conditioned audio memory + past-map summary memory
  -> 12-layer causal Transformer decoder
  -> shared embedding/output projection
  -> typed beatmap-event tokens
~~~

| Component or setting | Value |
|---|---:|
| Hidden width | 640 |
| Encoder layers | 8 |
| Decoder layers | 12 |
| Attention heads | 10 |
| Width per attention head | 64 |
| Feed-forward hidden width | 2,304 |
| Activation | GELU |
| Dropout | 0.1 |
| Normalization | Pre-normalized residual blocks; final encoder and decoder LayerNorm |
| Position information | Sinusoidal positions |
| Encoder attention | Bidirectional within the available audio window |
| Decoder self-attention | Causal |
| Decoder conditioning | Cross-attention over audio and past-map summary memory |
| Output projection | Tied to the token embedding weights; no separate output bias |
| Maximum decoder input length | 6,144 tokens |
| Maximum raw event-history budget | 2,048 tokens |
| Maximum past-map summaries | 128 |
| Summary feature width | 68 |
| Timing phase feature width | 3 |
| Beat prediction channels | 2: beat and downbeat |

### Parameter allocation

The counts below were recalculated from the production model on the PyTorch meta device, without allocating another model on the training GPU.

| Component | Parameters |
|---|---:|
| Convolutional audio frontend | 2,869,120 |
| Eight encoder blocks | 36,764,672 |
| Twelve decoder blocks | 74,853,888 |
| Shared event embedding / output weights | 6,072,960 |
| Encoder and decoder final norms | 2,560 |
| Timing phase projection | 2,560 |
| Beat/downbeat head | 1,282 |
| Past-map summary projection | 44,160 |
| Audio/summary modality embedding | 1,280 |
| **Total** | **120,612,482** |

The frontend consists of three Conv1d layers: `128 -> 640`, kernel 5, stride 1; then two `640 -> 640`, kernel 3, stride 2 layers. Each is followed by GELU. A full audio window contains **6,416 mel frames**, producing **1,604 encoded audio positions**. Up to 128 summary positions can be appended to the cross-attention memory.

### Audio timing and information separation

The beat head reads normalized audio-encoder output **before** timing-phase features and past-map summaries are added. It therefore cannot directly read reference timing phase, target event tokens, or past-map summaries to predict its training labels.

The decoder receives three phase features derived from the available timing grid: sine and cosine of beat phase, plus cosine of bar phase. Training normally uses the reference grid with the perturbations described in Section 7. Free generation can obtain that grid from the model's audio-only beat head and a deterministic timing estimator.

## 3. Audio representation

| Setting | Value |
|---|---|
| Supported input extensions | `.mp3`, `.ogg`, `.wav`, `.flac` |
| Decoder | FFmpeg, first audio stream |
| Channels | Mono |
| Sample rate | **22,050 Hz** |
| Decoded sample type | Float32 PCM |
| FFT size | 1,024 samples |
| Analysis window | NumPy Hann window |
| Hop length | 220 samples |
| Exact frame interval | Approximately **9.9773 ms** |
| Mel bands | **128** |
| Mel frequency range | 20 Hz to the Nyquist frequency, 11,025 Hz |
| Spectrum | Power spectrum |
| Log transform | Natural logarithm, with a power floor of `1e-8` |
| Input scaling | `(log_mel + 5) / 7` |
| Frame alignment | Centered on the global recording frame grid, with zero padding |
| Audio window | **64 seconds** |
| Cached mel dtype | Float16; restored to Float32 for dataset construction |
| Audio preprocessing identifier | `mel-v1-22050-220-1024-128-centered` |

The loader does not trim silence or normalize each window independently. Fixed scaling and identical global frame alignment keep training and inference preprocessing consistent.

## 4. Context, windows, and long-song generation

Each target interval is normally **24 seconds**. The associated audio window begins at `max(0, target_start - 32 seconds)` and spans 64 seconds.

For a normal interval sufficiently far into a song, this gives **32 seconds of audio history + 24 seconds of target audio + 8 seconds of lookahead**. Near the beginning of a song, more of the window can lie ahead of the target interval. This is an offline model with future-audio access.

The map context has two levels:

1. **Recent event history:** complete previous event records within the recent context, subject to a 2,048-token budget. Active sliders or spinners that began before the audio window can be represented by their remaining duration.
2. **Older map history:** up to 128 summaries made only from events before the audio-window origin. Each contains a 64-value pattern descriptor and four age/span features. The minimum aggregation stride is eight seconds and grows for long histories.

The descriptor summarizes rhythm intervals, spacing normalized by circle size, turning angles, slider curve types, repeats and lengths, object-type proportions, new-combo frequency, playfield occupancy, and density. A learned 68-to-640 projection maps it into model memory.

Training and inference use the same context builder. During teacher forcing, previous events come from the reference map; during free generation, they come from the map already generated.

The target-event budget is **4,081 tokens** under the conservative full-history allocation. If an interval is too dense, indexing shortens it at a complete event-time boundary. It then continues with the remaining interval. A single timestamp whose event records cannot fit is rejected rather than silently truncated.

All target intervals are indexed across the full recording, including song middles, endings, and intervals without objects. Raw historical events may be truncated by the history budget; older information is represented by summaries rather than an unlimited raw token context.

## 5. Event tokenizer

Lyra uses a **fixed, purpose-built vocabulary of 9,489 tokens**, not a natural-language subword tokenizer. No BPE or SentencePiece vocabulary training is involved.

**Format identifier:** `lyra-events-v1-64s-ms-xy2`.

| Token family | Count | Meaning |
|---|---:|---|
| Structural and event tokens | 11 | `PAD`, `BOS`, `CTX`, `GEN`, `EOS`, `END`, `PATH_END`, `CIRCLE`, `SLIDER`, `SPINNER`, `BREAK` |
| `T` | 6,401 | Coarse relative time, in 10 ms units |
| `F` | 10 | Fine relative-time remainder, 0–9 ms |
| `XY` | 2,049 | Quantized coordinate values |
| `COMBO` | 16 | New-combo flag and combo-skip information |
| `HS` | 16 | Four basic hitsound bits |
| `CURVE` | 4 | Bézier, Catmull, linear, and perfect-circle slider curves: B/C/L/P |
| `REP` | 64 | Slider repeat counts 1–64 |
| `BYTE` | 256 | Bytes used in three-token unsigned 24-bit integers |
| `STAR` | 202 | 0.0–20.0 stars at 0.1-star resolution, plus an unspecified value |
| `AIM`, `STREAMS`, `RHYTHM` | 4 each | Unspecified or one of three style levels, 0/1/2 |
| `AR`, `OD`, `CS`, `HP` | 112 each | 0.0–11.0 at 0.1 resolution, plus an unspecified value |
| **Total** | **9,489** | |

Object times are rounded to **1 ms** and encoded as a coarse/fine pair relative to the audio-window base. Coordinates are rounded to the nearest **2 osu! pixels**. The coordinate vocabulary represents values from −2,048 to +2,048 pixels; generation applies narrower, state-dependent bounds.

The decoder can represent:

- Hit circles: time, coordinates, combo information, and basic hitsound bits.
- Sliders: the same fields plus native curve type, repeat count, total duration, slider velocity, and control-point coordinates.
- Spinners: start time, coordinates, combo/hitsound information, and duration.
- Breaks: start time and duration.

Durations and scaled slider velocity use three `BYTE` tokens. Slider velocity is represented at a scale of 10,000. Repeated Bézier control points preserve native segment separators.

The timing grid itself is not a sequence of red-timing-point tokens: the audio-only timing head and timing estimator produce it separately. Slider velocity can be reconstructed as inherited timing points.

The tokenizer does not encode custom audio samples, slider edge-sound arrays, slider edge sample-set arrays, backgrounds, videos, or storyboards. Parsed `.osu` fields are therefore not all reconstruction targets. The decoder input includes condition tokens, interval boundary, `CTX`, past events, `GEN`, and generated events; `EOS` closes the target interval.

The vocabulary permits difficulty-setting tokens through 11.0, while the current generation interface validates user-specified AR/OD/CS/HP values in the range **0–10**.

## 6. Dataset

### Provenance and scale

The corpus contains human-created osu!standard beatmaps and their referenced music from the local osu! dataset pipeline. The pipeline retrieves osu! metadata and prepares a frozen selection of downloaded files. The exact original download URLs, a complete crawl-date range, and redistribution licenses are not established by this card.

The stored upstream build record is dated **4 October 2026, 09:40:13 UTC**. Lyra-specific full-window indexing was completed on **19 October 2026**.

| Quantity | Verified value |
|---|---:|
| Selected source rows examined | **151,051** |
| Unexamined selected rows | **0** |
| Rejected during source preparation | **3,087** |
| Accepted beatmaps | **147,964** |
| Additional rejections during Lyra indexing | **0** |
| Distinct decoded-audio identities | **33,994** |
| Song groups | **29,304** |
| Distinct-audio duration | **1,791.94 hours** |
| Lyra target windows | **1,068,406** |
| Supervised event tokens per full corpus pass, including one EOS per window | **911,903,307** |

The token total is derived from the indexed event lengths, excluding conditioning, history, and padding. It is not the number of unique vocabulary items or the total number of decoder input positions.

Source metadata reports **67,146,731 objects**: 38,407,139 circles, 28,497,082 sliders, and 242,510 spinners. These are counts from the retained source metadata, not a fresh parser recount of every object.

### Referenced file size

The following measures sum the logical file lengths of unique source paths referenced by the accepted corpus. All referenced paths were readable at inspection time.

| Assets | Unique file paths | Logical bytes | Decimal GB |
|---|---:|---:|---:|
| Referenced audio files | 38,367 | 171,264,910,757 | 171.265 |
| Referenced `.osu` files | 147,964 | 3,932,480,857 | 3.932 |
| **Combined source assets** | **186,331** | **175,197,391,614** | **175.197** |

The combined size is approximately **163.17 GiB**. File-path counts exceed distinct-audio identities because multiple physical files can decode to the same audio. This is not the size of the whole downloads folder, filesystem allocation after hardlink/compression deduplication, or the rotating training cache.

The Lyra SQLite index is **474,181,632 bytes** / **452.21 MiB**. Its three compact window-index arrays total **17,094,880 bytes** / **16.30 MiB**. Checkpoints and caches are additional storage.

### Train / validation / test split

| Split | Beatmaps | Target windows | Audio identities | Song groups | Distinct-audio hours |
|---|---:|---:|---:|---:|---:|
| Train | **133,440** | **964,841** | **30,621** | **26,378** | **1,616.59** |
| Validation | **7,526** | **53,947** | **1,749** | **1,510** | **90.60** |
| Test | **6,998** | **49,618** | **1,624** | **1,416** | **84.75** |
| **Total** | **147,964** | **1,068,406** | **33,994** | **29,304** | **1,791.94** |

The intended split is **90% / 5% / 5% by song group**, not by individual window. Unequal group sizes produce slightly different map and window proportions.

The upstream grouping joins related records using exact decoded-audio hash, an audio-energy fingerprint, normalized artist/title key, or beatmap-set ID. Group membership is transitive. The group's audio hashes determine a stable identifier; its first eight hexadecimal digits, modulo 100, select train below 90, validation below 95, and test otherwise.

Lyra preserves this split. A fresh read-only index audit found **zero audio identities shared across splits** and **zero song-group identifiers shared across splits**.

This does not establish artist-disjoint evaluation or perfect detection of every remix, alternate recording, or metadata alias.

### Selection and filtering

The stored metadata-build settings are:

| Rule | Recorded setting |
|---|---|
| Ruleset | osu!standard; converted and deleted entries are excluded during ingestion |
| Set statuses | Ranked, approved, loved |
| Lazer-only maps | Excluded |
| Website star-rating range | 0–15 |
| Minimum per-difficulty play count | 50 in the stored build arguments |
| Hit-length range | 15–600 seconds |
| Minimum tag votes for retained strong tags | 5 |
| Initial nominal validation / test fractions | 0.05 / 0.05 |
| Metadata-build `only_ready` | False; file availability is checked later |

The actual 151,051-row candidate table has a minimum recorded play count of **300** and website stars from **0.0 to 14.9755**. This observed range is reported separately from the saved build arguments; the card does not infer an undocumented later selection command.

Source preparation checks file availability, map checksums, referenced audio consistency, decoded audio validity, supported ruleset and object structure, valid timing, sorted nonnegative object times, slider/spinner validity, audio bounds, tokenizer representability, and window fit. Native slider paths with more than 64 anchors are rejected. Duplicate map/audio identities are skipped rather than added repeatedly.

The recorded rejection breakdown is:

| Reason category | Maps |
|---|---:|
| More than 64 native slider anchors | 1,545 |
| Missing `.osu` path | 1,029 |
| Invalid timing point | 281 |
| Other source-file, parser, or validation failures | 113 |
| Slider repeat count outside the vocabulary | 75 |
| Non-finite or zero timing value | 14 |
| Invalid spinner duration | 10 |
| Invalid slider | 9 |
| Unsorted or negative hit-object time | 6 |
| Hit object extends beyond audio | 3 |
| Coordinates outside the vocabulary | 2 |
| **Total** | **3,087** |

Lyra then verifies map SHA-256 identities, checks source split consistency, parses each retained map, and indexes every representable target interval under its own context configuration. **All 147,964 accepted maps survived this stage.**

The full selected corpus was examined. Cache budgets do not remove maps from the dataset. Lyra does not apply additional star-band sampling, popularity weighting, or a small cached subset during training.

### Difficulty, status, and content distribution

The following star histogram uses the **recalculated star rating stored in the prepared corpus**, with integer bands `floor(stars)`; it is separate from the website-star filter above.

| Star band | Maps |
|---|---:|
| 0–<1 | 91 |
| 1–<2 | 18,961 |
| 2–<3 | 29,787 |
| 3–<4 | 30,495 |
| 4–<5 | 28,471 |
| 5–<6 | 24,986 |
| 6–<7 | 11,207 |
| 7–<8 | 2,950 |
| 8–<9 | 735 |
| 9–<10 | 202 |
| 10–<11 | 58 |
| 11–<12 | 14 |
| 12–<13 | 5 |
| 13–<14 | 1 |
| 14–<15 | 1 |
| **Total** | **147,964** |

The retained corpus contains **143,932 ranked**, **3,714 loved**, and **318 approved** maps. Recalculated training-map ratings range from approximately **0.14 to 14.16 stars**; the training-map mean is approximately **3.92 stars**. Support at very high difficulty is sparse, even though the generation control accepts ratings up to 20.

The largest genre categories, counted by map rather than unique recording, are Pop (31,422), Anime (30,680), Video Game (26,581), Electronic (24,202), and Rock (23,308). Language metadata is dominated by Japanese (86,450), Instrumental (27,919), and English (25,582). These categories describe the music corpus, not a text-generation capability.

The retained metadata includes **3,320 maps marked NSFW**. No NSFW-exclusion step is established by the recorded build configuration.

### Style-label provenance

Style controls are **aim**, **streams**, and **rhythm**, each with levels 0/1/2 or unspecified.

The preparation code can use strong metadata tags or statistics fitted on training maps within star bands. **In this actual snapshot, 147,962 maps have all three style labels derived from training-band statistics; two maps have no recorded provenance for those labels.** The inspected source index does not show tag-derived labels for these three controls.

These are heuristic descriptors rather than independently verified human style annotations. Thresholds are fitted on the training split and then applied to held-out maps.

The older metadata pipeline also calculates popularity/quality weights, including a loved-status factor of 0.85. **The current Lyra sampler does not consume those sample weights.** Each eligible training window appears once per shuffled cycle; longer or more densely token-limited maps contribute more windows.

## 7. Training configuration and data augmentation

| Setting | Actual main-run value |
|---|---|
| Training objective | Joint event generation and beat/downbeat supervision |
| Trainable model | Entire model |
| Optimizer | AdamW |
| CUDA implementation | Fused AdamW |
| Base learning rate | **0.0003** |
| Adam betas | **0.9, 0.999**; library defaults |
| Adam epsilon | **1e-8**; library default |
| Weight decay | **0.01** |
| Weight-decay parameter groups | One group containing all model parameters |
| Physical microbatch | **4 windows** |
| Effective optimizer batch | **32 windows** |
| Gradient accumulation | **8 microbatches per optimizer update** |
| Precision | FP16 CUDA autocast with dynamic gradient scaling; model parameters are initialized in FP32 |
| Gradient norm clipping | **1.0** |
| Warmup | **1,000 optimizer updates** |
| Learning-rate schedule | Linear warmup, then cosine decay with a 10% floor |
| Schedule horizon | **100,000 updates** |
| Requested training duration | **3 complete training-window cycles** |
| Target optimizer updates | **90,454** |
| Explicit step override | 0, meaning derive the target from cycles |
| Seed | **42** |
| DataLoader workers | **4** |
| Main-process PyTorch CPU threads | **2** |
| Per-loader-worker thread limit | **1** |
| Worker prefetch factor | **2** |
| Validation interval | Update 1, then every **500** updates |
| Validation sample | **64 fixed windows** |
| Qualitative sample interval | Every **1,000** updates |
| Checkpoint timer | **15 minutes**, plus validation boundaries and orderly exit |

The schedule factor for scheduler index `n` is:

~~~text
if n < 1000:
    factor(n) = (n + 1) / 1000
else:
    progress = min(1, (n - 1000) / (100000 - 1000))
    factor(n) = 0.1 + 0.9 * 0.5 * (1 + cos(pi * progress))

learning_rate(n) = 0.0003 * factor(n)
~~~

The three-cycle goal ends before the 100,000-update schedule horizon. If unchanged, the learning rate at the 90,454-update goal will be approximately **3.615e-5**, slightly above the eventual floor of 3e-5.

The sampler shuffles all training-window indices using `seed + cycle`. Within each optimizer update, the selected 32 windows are sorted by their indexed token-length estimate, then divided into microbatches. This reduces padding while retaining the selected update's membership. The final update includes **five windows from the next cycle** to fill the batch:

~~~text
ceil(964,841 * 3 / 32) = 90,454 updates
90,454 * 32           = 2,894,528 window visits
964,841 * 3           = 2,894,523 requested cycle visits
~~~

Three exact training cycles contain **2,470,025,145 supervised event tokens**, including EOS but excluding prefix/history/padding. The five batch-filling windows add their own target tokens.

### Conditioning perturbations

- **Style dropout:** 20% probability of replacing the style labels with unspecified controls.
- **Timing-phase dropout:** 20% probability of zeroing all phase features.
- **Timing-phase roll:** if phase was not dropped, a separate 25% branch applies a roll of −3 to +3 mel frames, including zero. Thus 20% of samples enter the roll branch overall; 60% keep the unmodified reference phase.

The roll spans approximately ±29.93 ms. These perturbations do not alter the target beatmap events. The inspected training path does not apply waveform pitch shifting, speed augmentation, or SpecAugment.

## 8. Loss calculation

The optimized objective is:

$$
L = L_{token} + 0.3 L_{beat}.
$$

Both terms are normalized over the **entire effective batch of 32 windows**, before gradients are accumulated across its eight microbatches.

### Event-token cross-entropy

Let `S` be the set of supervised next-token positions across the effective batch and `N = |S|`. Then:

$$
L_{token} = -\frac{1}{\max(1,N)}
\sum_{i \in S} \log p_\theta(y_i \mid y_{<i}, audio, conditions, history).
$$

The full reference sequence is shifted for next-token prediction. Condition tokens and past-history positions receive label **−100** and are excluded. Padding is also excluded. The current interval's event tokens and its final **EOS** are supervised.

Each supervised token contributes equally. Object types, coordinates, timing tokens, slider anchors, and structural markers do not receive separate event-specific weights. The inspected implementation uses no label smoothing.

The output logits are computed only for supervised hidden states, in **128-position chunks**. Cross-entropy is evaluated in Float32, summed, and divided by the effective batch's supervised-token count. This avoids retaining a full batch × sequence × vocabulary logit tensor. The inference grammar mask is not applied to the training cross-entropy.

### Beat/downbeat binary cross-entropy

The timing head predicts two logits per mel frame. For target `y`, logit `z`, and positive-class weight `p`, the elementwise loss is:

$$
\ell(z,y;p) =
-p y\log\sigma(z) - (1-y)\log(1-\sigma(z)).
$$

The positive weights are:

- **6** for beat targets.
- **12** for downbeat targets.

Let `m[f,c]` be the supervision mask and `M = sum(m)` over both channels and all windows in the effective batch:

$$
L_{beat} =
\frac{\sum_{f,c}m[f,c]\,\ell(z[f,c],y[f,c];p[c])}
{\max(1,M)}.
$$

The denominator counts valid frame-channel entries. It is not the sum of positive-class weights.

Timing labels are built from the source maps' uninherited timing grids. Each beat receives a Gaussian target with **15 ms standard deviation**, evaluated over a seven-frame neighborhood. Downbeats are selected using the source timing point's meter.

For an audio identity with multiple distinct reference grids, targets are averaged. A frame-channel entry is supervised only when:

1. The largest and smallest grid targets differ by **less than 0.35**.
2. Its time is between **four seconds before the first object** and **two seconds after the last object**, clipped at zero on the left.
3. It is not in an invalid or unresolvable timing segment. Non-finite or sub-1-ms beat-length segments, with a four-frame boundary margin, are masked.
4. It corresponds to real audio rather than padding.

### Accumulation, logging, and model selection

For microbatch `k`, the implementation backpropagates:

~~~text
microbatch_loss =
    token_cross_entropy_sum[k] / total_valid_target_tokens_in_32_windows
    + 0.3 * masked_beat_bce_sum[k] / total_valid_frame_channels_in_32_windows
~~~

Summing these gradients gives the effective-batch objective above. Microbatches with different token counts are therefore not incorrectly weighted as equal means.

Logged `token_loss` and `beat_loss` are the two normalized components; **the logged beat loss has not yet been multiplied by 0.3**. Logged `loss` equals their weighted sum. The displayed EMA is `0.98 * previous_ema + 0.02 * current_loss`.

Validation reports token and beat losses separately, weighted by their corresponding valid counts across selected windows. **The best checkpoint is selected by validation token loss alone**, not by the weighted joint loss or a playability score.

## 9. Training progress and duration

### Observed snapshot

| Metric | Value at update 6,089 |
|---|---:|
| Training state | Running |
| Optimizer updates | **6,089 / 90,454** |
| Progress toward the three-cycle update goal | **6.73%** |
| Training windows visited | **194,848** |
| Complete training cycles | **0** |
| Progress through the first cycle | **20.19%** |
| Supervised target tokens processed in the logged trajectory | **166,408,025** |
| Cumulative active-run elapsed time | **89,649.657 seconds** |
| Elapsed time in hours | **24 h 54 min 10 s** |
| Latest joint training loss | **1.870322** |
| Latest token loss | **1.716057** |
| Latest beat loss | **0.514218** |
| Training-loss EMA | **1.914690** |
| Learning rate | **0.0002982435** |
| Latest pre-clipping gradient norm | **0.492151** |
| Latest gradient scale | **524,288** |
| AMP retries in retained logged updates | **0** |
| Retained logged updates | **6,089** |
| Archived rollback metrics | **0** |

The earliest retained training-job launch was **19 October 2026, 09:03:58 Europe/Budapest**. The wall-clock span from that launch to the metric snapshot is approximately **25 h 59 min 32 s**. The difference from the cumulative active-run timer includes time between resumed processes and other time outside that timer.

The latest worker process was launched on **19 October 2026 at 14:43:58 Europe/Budapest**, resuming the same `main/last.pt` trajectory. Previous launches are not separate completed training runs.

The active-run elapsed timer includes loading, training, validation, checkpointing, and sampling while the training process is inside its timed run. It is not a CUDA-kernel-only measurement, and it does not include every paused interval or the initial dataset preparation.

### Throughput and estimated remaining time

| Measure | Observed value |
|---|---:|
| Mean update duration, last 1,000 updates | **14.558 s** |
| Mean cumulative elapsed-time slope, last 1,000 updates | **14.694 s/update** |
| Mean per-update target-token throughput, last 1,000 updates | **1,881 tokens/s** |
| Mean per-update window throughput, last 1,000 updates | **2.21 windows/s** |
| Remaining updates | **84,365** |
| Projected remaining active-run time | **Approximately 344.36 hours / 14.35 days** |
| Projected total active-run time at the current goal | **Approximately 369.26 hours / 15.39 days** |
| Indicative finish if uninterrupted | **Around 3 November 2026, Europe/Budapest** |

The estimate is `remaining_updates * recent_elapsed_seconds_per_update`, using the cumulative elapsed delta across the last 1,000 logged updates. That slope captures intervening periodic overhead more closely than the per-update compute timer.

**This is an extrapolation, not a completed training duration or a guaranteed finish date.** It assumes similar window lengths, hardware availability, validation cost, and generation cost. Successful longer samples, pauses, cache behavior, or configuration changes can change the result.

## 10. Validation and observed generation results

### Validation protocol

Validation selects **64 windows without replacement** using a fixed NumPy generator seed of **2026** from the complete validation-window index. The same 64 windows, containing **53,638 target tokens**, are reused at every check. They can come from the middle and end of recordings.

The model is switched to evaluation mode, and RNG state is restored afterward. Validation data does not receive optimizer updates. Repeated use of this fixed subset for checkpoint selection makes it a development-validation sample, not an untouched final test.

| Update | Validation token loss | Validation beat loss |
|---:|---:|---:|
| 1 | 9.212667 | 0.997540 |
| 500 | 3.220329 | 0.651188 |
| 1,000 | 2.838096 | 0.604495 |
| 1,500 | 2.361706 | 0.665564 |
| 2,000 | 2.189288 | 0.705216 |
| 2,500 | 2.077413 | 0.600903 |
| 3,000 | 2.000618 | 0.589732 |
| 3,500 | 1.940549 | 0.576271 |
| 4,000 | 1.897724 | 0.577696 |
| 4,500 | 1.853705 | 0.564112 |
| 5,000 | 1.820776 | 0.553171 |
| 5,500 | 1.782212 | 0.541142 |
| **6,000** | **1.746327** | **0.526758** |

The lowest logged validation token loss at the snapshot is **1.746327 at update 6,000**. Losses demonstrate learning on this teacher-forced subset; they do not measure free-generation success, timing F1, ranking eligibility, or player satisfaction.

**No completed final test-set result is established by the inspected run artifacts.** The evaluation utility defaults to 128 sampled test windows and can be given a larger count. A default evaluation call therefore also does not automatically cover every test window.

### Fixed external-audio previews

The current periodic preview uses a frozen **40-second excerpt, 00:50–01:30**, from a local music file. The excerpt is stored independently as Ogg audio and never added to the training corpus. A prior check found that the original full recording's exact decoded-PCM hash was absent from the corpus; this is not a guarantee against every alternate edit or recording.

Preview controls are fixed: **6.5 requested stars**, Auto style, seed **2026**, temperature **0.6**, top-p **0.95**, one candidate, AR **9.4**, OD **9.0**, CS **4.0**, HP **5.0**. Timing is inferred from audio for the external excerpt.

| Update / preview | Timing source | Outcome | Measured stars | Objects |
|---|---|---|---:|---:|
| 1,000, earlier full-song Hakai example | Reference map timing | Exported | 5.90 | 69 |
| 1,115, external excerpt check | Audio-inferred | Exported | 4.68 | 22 |
| 2,000, external excerpt | Audio-inferred | Rejected: object geometry after retry limit | — | — |
| 3,000, external excerpt | Audio-inferred | Exported | 3.12 | 27 |
| 4,000, external excerpt | Audio-inferred | Rejected: object geometry after retry limit | — | — |
| 5,000, external excerpt | Audio-inferred | Exported | 7.78 | 221 |
| 6,000, external excerpt | Audio-inferred | Rejected: object geometry after retry limit | — | — |

The update-1,000 example requested approximately 6.53 stars and uses a different song and reference timing, so it is not directly comparable with the fixed external excerpt.

These results expose substantial current variability in geometry, object density, and difficulty control. The successful exports are examples, not a representative benchmark; failed scheduled previews are retained here rather than omitted.

## 11. Memory, speed, and implementation optimizations

### Training memory and compute

- **Joint optimization:** one model and one optimizer update both the audio and event-generation components.
- **FP16 automatic mixed precision** with dynamic loss scaling.
- **Gradient checkpointing** in encoder blocks, decoder blocks, and chunked token-loss projection.
- **128-position output-loss chunks**, computed only at supervised positions, with Float32 cross-entropy.
- **Tied embedding/output weights**, avoiding a second vocabulary projection matrix.
- **Fourfold audio downsampling** before Transformer encoding.
- **Fused AdamW** on CUDA.
- **Gradient accumulation** to obtain batch 32 from physical batch 4.
- **PyTorch scaled-dot-product attention**; the backend is chosen by PyTorch and hardware. A specific FlashAttention backend is not established by this card.
- **Length-aware microbatch grouping** within each selected optimizer batch.

The inspected path does not enable a `torch.compile` training stage or distributed training.

### Input pipeline and cache

- Four persistent data-loader workers, each restricted to one CPU thread, with prefetch factor two.
- Pinned CPU tensors and a dedicated CUDA transfer stream for nonblocking preloading.
- Window-index arrays opened through read-only memory mapping.
- Parsed-map reuse and globally aligned **4,096-frame FP16 mel tiles**.
- Per-recording locks to prevent workers from decoding the same missing recording concurrently.
- A byte-bounded RAM LRU cache and an SSD LRU cache with CRC checks.
- Uncompressed NumPy cache containers to avoid compression work in the loading path.
- Read-only borrowing of compatible older mel/timing cache entries. Older learned encoder features are not model inputs.

The configured RAM-array budget is **8 GiB total**: up to 0.5 GiB for validation and 7.5 GiB split across the four training workers, or 1.875 GiB each. This is a cache budget, not a limit on total process RAM.

The SSD-cache quota is **180 decimal GB**, with a minimum **8 GiB free-space reserve**. Its entries are derived data and may be evicted. Cache eviction or a cache-write bypass does not alter dataset membership.

The PyTorch per-process CUDA allocator fraction is set to **0.8**. This does not reserve or cap every allocation made by the display driver or other applications.

### Hardware and runtime

| Component | Inspected environment |
|---|---|
| GPU | **NVIDIA GeForce RTX 3050, 8,192 MiB VRAM** |
| GPU count used by this training path | 1 |
| CPU | AMD Ryzen 5 5600, 6 cores / 12 logical processors |
| Reported system RAM | 34,281,668,608 bytes, approximately 31.93 GiB |
| Operating system | Windows 11, build 26100 |
| Python | 3.12.14 |
| PyTorch | **2.8.0+cu126** |
| PyTorch CUDA build | **12.6** |
| cuDNN version integer | 91002 |
| NVIDIA driver | 617.42 |
| Core numeric dependencies, project pins | NumPy 2.2.6; SciPy 1.15.3 |
| Difficulty calculation, project pin | rosu-pp-py 3.1.0 |
| FFmpeg wrapper, project pin | imageio-ffmpeg 0.6.0 |

The maximum PyTorch allocated-memory peak reported across retained metrics is **3.894 GiB**. The latest record reports **5.982 GiB reserved** by the allocator and **0.732 GiB globally free VRAM**. Allocated, reserved, and globally free memory are different measurements; system/display usage can account for additional VRAM.

### Earlier maximum-shape memory benchmark

The saved local benchmark uses the full **64-second / 6,144-token** shape with forward, backward, and AdamW steps. It measures three microbatches after warmup.

| Physical batch | Seconds per maximum-shape microbatch | Windows/s | Peak allocated VRAM |
|---:|---:|---:|---:|
| 1 | 1.292 | 0.774 | 2.05 GiB |
| **4** | **4.901** | **0.816** | **3.40 GiB** |
| 8 | 9.911 | 0.807 | 5.37 GiB |

Batch 4 is the selected profile. These are synthetic maximum-shape microbatch measurements, not the actual effective-batch update time. The live corpus contains shorter token sequences, and a real optimizer update accumulates eight microbatches.

No integrated training-energy measurement or carbon-emissions estimate is available. An instantaneous GPU power reading is insufficient to establish either quantity.

## 12. Inference and decoding

The generator first obtains timing from one of three sources: an optional reference timing map, manually supplied BPM/offset, or audio-derived beat/downbeat activations followed by timing estimation.

Automatic timing uses peak evidence, autocorrelation and tempo-octave candidates, robust local fits over approximately 32-second segments, and downbeat evidence for bar phase. It can output a piecewise timing grid. Confidence below 0.55, coverage below 0.5, or a missing valid grid marks the result uncertain; strict generation raises an error requesting corrected timing.

Objects are then generated autoregressively in approximately 24-second sections using cached decoder self-attention keys/values and static cross-attention keys/values.

The finite-state grammar constrains event types, times, field order, coordinates, durations, repeat counts, and complete records. Object start times increase, and new generated notes do not begin inside an ongoing slider or spinner according to the generator's busy-state rule.

Each complete object is decoded and validated. Geometric checks include the played slider path and circle-size radius, not only anchor coordinates. Invalid objects can be regenerated up to three times after the initial attempt before the candidate fails.

The default general-generation configuration is:

| Control | Default |
|---|---:|
| Requested stars | 5.0 |
| Style | Auto |
| Seed | 42 |
| Candidate count | 3 |
| Temperature | 0.8 |
| Top-p | 0.95 |
| Fallback difficulty settings | AR 8, OD 7, CS 4, HP 5 |

The regular generator scores valid candidates by absolute measured-star error plus half the repetition penalty. It also applies heuristic circle-coordinate logit penalties for persistent stacks or two-point loops. These are inference-time heuristics and are not additional training-loss terms.

Difficulty is measured using rosu-pp-py with `lazer=False`. A difference greater than 0.5 stars is reported as a difficulty miss; the requested rating is a conditioning target, not a guaranteed measured output rating.

An exported `.osz` contains the generated map, a usable copy of the provided audio, and `generation.json` with settings and provenance. Lyra does not synthesize the music itself.

## 13. Checkpoints, logging, and reproducibility

| Artifact | Role |
|---|---|
| `last.pt` | Resume checkpoint from the most recent saved optimizer boundary |
| `best.pt` | Lowest observed validation-token-loss checkpoint |
| `config.json` | Actual model/training settings, target steps, and schedule horizon |
| `dataset.json` | Frozen dataset identity copied into the run |
| `metrics.sqlite` | Durable per-update metrics and validation records |
| `status.json` | Latest run status |
| `samples/step-*/` | Exported examples or failure reports |
| `jobs/*.log` | Background-worker text logs |

At inspection, each `best.pt` and `last.pt` file was **1,447,916,077 bytes**, approximately **1.448 GB / 1.348 GiB**. These are resumable training containers with optimizer state, not minimal weight-only exports.

Checkpoints include model weights, model configuration, training configuration, optimizer state, scheduler state, AMP scaler, Python/NumPy/PyTorch/CUDA RNG states, step count, cumulative elapsed time, best validation loss, dataset hash, coverage counters, and generation defaults.

Resume verifies the dataset identity and stored index digest, restores optimizer/scheduler/scaler/RNG state, and reconstructs the next deterministic window selection. Operational settings such as workers and physical batch can be changed within the code's supported resume rules; the effective batch and core optimization configuration are restored from the checkpoint.

Metrics are committed per completed optimizer update. If a resume rolls back to an older checkpoint, later metric rows are archived rather than silently lost. The inspected trajectory had no archived rollback rows.

The background worker is independent of the browser connection. The UI can reconnect after a refresh or server restart. Periodic preview failures are recorded in `rejected.json` and do not invalidate completed optimizer updates.

Exact numerical reproducibility across different GPUs, CUDA stacks, or library versions has not been established. Fixed seeds and retained RNG state alone are not a cross-platform bitwise guarantee.

### Frozen identifiers

~~~text
Lyra dataset format:
  lyra-dataset-1

Lyra dataset identity:
  a6afcf65729b9cb06cf3116c91e50f0b5e2080d35a323a931ceea81270381081

Source corpus identity:
  a71f5bf2471b5fa31dd2338462405e4f2d66da3f107123a1d3511f180d432f79

Lyra index SHA-256:
  e5923fda9245f53da3930c9597ace4b7f05390b8e202b42bc2035ab023263242

Train window-array SHA-256:
  31d9982a59b1fe734cc90c0cf6f6dba910183207f0dfcd8f982f5f89ad2d2688

Validation window-array SHA-256:
  d4d0e9aff2872cd5d37ab0f2783440694ccce59a47fc7696f6295ee9645d1bf8

Test window-array SHA-256:
  0209817571877b88d239b1e0c542db48c3e83871530281f241087554001a22a0

Tokenizer:
  lyra-events-v1-64s-ms-xy2

Audio preprocessing:
  mel-v1-22050-220-1024-128-centered

Checkpoint container:
  lyra-1
~~~

The digest values above are those recorded in the frozen manifests. Dataset membership and split counts were independently queried for this card; source-code hashes were recalculated. Large live checkpoint files were not rehashed or loaded into the training GPU for documentation.

### Local artifact locations

~~~text
Source code:
  E:/osu_ai/osu_ai_final

Prepared Lyra dataset:
  F:/osu_dataset/_lyra/dataset/manifest.json

Current run:
  F:/osu_dataset/_lyra/runs/main

Background jobs:
  F:/osu_dataset/_lyra/jobs

Frozen external preview:
  F:/osu_dataset/_lyra/examples/music-50-90/example.json

Rotating cache:
  F:/osu_cache/lyra
~~~

Inference requires the Lyra source/runtime, compatible dependencies, a Lyra checkpoint, and input audio. It does not require the training corpus, a separate pretrained audio checkpoint, or a learned encoder-feature manifest. Resume and held-out evaluation require the matching dataset.

## 14. Usage

From an environment with this project's dependencies installed:

~~~powershell
# Open the local interface.
python -m lyra ui --port 7862 --open-browser

# Generate from an existing checkpoint.
python -m lyra generate F:/osu_dataset/_lyra/runs/main/best.pt F:/Music/song.mp3 --stars 5

# Provide manual timing when appropriate.
python -m lyra generate F:/osu_dataset/_lyra/runs/main/best.pt F:/Music/song.mp3 --stars 5 --bpm 180 --offset 25

# Evaluate the default 128 sampled test windows.
python -m lyra evaluate F:/osu_dataset/_lyra/runs/main/best.pt --count 128

# Request all 49,618 test windows with the current frozen corpus.
python -m lyra evaluate F:/osu_dataset/_lyra/runs/main/best.pt --count 49618
~~~

The local worker uses a GPU lease. Run generation or evaluation when the training worker has released that lease.

For a separate reproduction, prepare the complete corpus and start a run in a **new output directory**:

~~~powershell
python -m lyra prepare
python -m lyra train --run-dir F:/osu_dataset/_lyra/runs/reproduction --cycles 3 --batch-size 4 --workers 4 --effective-batch 32
~~~

The basic training CLI does not configure the custom external preview excerpt. Reproducing that aspect requires the saved `sample_audio_example` setting in the dispatched training configuration. The existing main run was resumed with its saved configuration; no new training was launched to prepare this card.

## 15. Limitations and interpretation

- **Training is incomplete.** The snapshot has not completed even one full training-window cycle.
- **Free-generation reliability is unresolved.** Three of the five scheduled external previews at updates 2,000–6,000 failed geometric validation. This small, fixed preview sequence is not a general success-rate benchmark.
- **Difficulty control is approximate.** Successful 6.5-star-target excerpt exports measured 3.12–7.78 stars in the recorded examples.
- **Teacher-forcing losses are not playability metrics.** They do not establish timing accuracy in milliseconds, beat/downbeat F1, rhythmic emphasis quality, player readability, or enjoyable patterns.
- **Long-term summaries are lossy.** Whole-song consistency is not guaranteed by the availability of 128 summary positions.
- **Training/inference history differs.** Training sees reference past events; inference conditions on its own accumulated errors.
- **Reference timing remains an important training condition.** Phase dropout and rolls reduce reliance but do not prove robust generation from uncertain predicted timing.
- **Dataset selection is biased.** Downloaded, popular, established-status maps dominate; map-weighted language and genre statistics are concentrated in Japanese and game/anime-related music.
- **Style labels are heuristic.** Their numerical controls are not verified labels of human intent.
- **Extreme difficulty is underrepresented.** A 20-star control range does not establish training coverage or quality at 20 stars.
- **Technical filters change coverage.** Maps with complex slider paths, very high repeats, malformed timing, or missing files are absent.
- **Representation is lossy.** Two-pixel coordinates and omitted custom/edge samples prevent exact reproduction of every original map detail.
- **Generation imposes structural choices.** Increasing start times and the busy-state rule restrict overlaps that some human maps may contain.
- **The model is offline.** Its bidirectional audio windows and lookahead do not establish real-time streaming capability.
- **Other osu! rulesets are unsupported.** This card does not cover taiko, catch, or mania.

## 16. Rights, release status, and outstanding evaluation

The dataset contains third-party music and human-authored beatmaps. Local availability, ranked/loved status, and this model card do not establish redistribution permissions. No universal dataset license, public weights license, or license for bundled example audio was found in the inspected release artifacts.

The model card does not grant those rights. Author and artist attribution, permitted audio use, code licensing, weight licensing, and any public sample packaging need explicit release documentation.

Before a completed release can be described, the following evidence remains to be added:

- Final training step, completed cycles, cumulative training duration, and the chosen checkpoint's checksum.
- Held-out test results with checkpoint identity, sampling/full-set scope, and count-weighted token/beat losses.
- Audio-only beat/downbeat timing metrics and timing-estimator failure coverage.
- A fixed, broader free-generation benchmark reporting failed candidates, geometry rejection rate, measured-star error, density, repetition, and performance by difficulty/style.
- Human playtesting and map review covering musicality, readability, playability, and long-song consistency.
- An explicit release license, dependency/source revision, and packaged inference instructions.

These are outstanding measurements and release details, not claims that the present checkpoint has passed them.

## 17. Evidence and implementation references

| Subject | Primary local evidence |
|---|---|
| Actual settings and target | `F:/osu_dataset/_lyra/runs/main/config.json` |
| Durable training and validation results | `F:/osu_dataset/_lyra/runs/main/metrics.sqlite` |
| Dataset counts, hashes, and context | `F:/osu_dataset/_lyra/dataset/manifest.json`, `index.sqlite` |
| Upstream corpus report and filtering failures | `F:/osu_dataset/_training/dataset-full/manifest.json`, `rejected.json` |
| Recorded source build | `F:/osu_dataset/_training/audio-input/source.sqlite`, whitelisted `meta.last_build` fields |
| Model and token loss | [lyra/model.py](lyra/model.py) |
| Joint objective, schedule, validation, and resume | [lyra/training.py](lyra/training.py) |
| Dataset import, masks, perturbations, and sampler | [lyra/data.py](lyra/data.py) |
| Audio preprocessing | [lyra/audio.py](lyra/audio.py) |
| Event representation and grammar | [lyra/tokenizer.py](lyra/tokenizer.py) |
| Context construction | [lyra/context.py](lyra/context.py), [lyra/style.py](lyra/style.py) |
| Cache and transfer pipeline | [lyra/cache.py](lyra/cache.py), [lyra/runtime.py](lyra/runtime.py) |
| Timing and generation | [lyra/timing.py](lyra/timing.py), [lyra/generation.py](lyra/generation.py) |
| Preview protocol | [lyra/examples.py](lyra/examples.py) |
| Geometric and repetition checks | [lyra/geometry.py](lyra/geometry.py), [lyra/quality.py](lyra/quality.py) |
| Durable metrics and export | [lyra/metrics.py](lyra/metrics.py) |
| Runtime pins | [pyproject.toml](pyproject.toml) |
| Frozen audit supporting this card | [model-card/evidence-2026-10-20.json](model-card/evidence-2026-10-20.json) |

Existing saved verification reports record checks for causal decoding and KV-cache equivalence, mapping gradients reaching the audio encoder, timing-head information separation, shared train/inference context, complete event-window coverage, split separation, cache equivalence, checkpoint resume, background-worker persistence, and self-contained exports.

The earlier base verification report lists 22 unique tests; later targeted reports overlap with that suite and are not added together as a new test count. Those reports are implementation evidence, not quality benchmarks of the unfinished model. This documentation pass used read-only training/index inspection and CPU-only model-shape inspection; it did not rerun GPU training benchmarks or interrupt the active training worker.
