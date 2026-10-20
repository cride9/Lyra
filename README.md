# Lyra v2.5

A single audio-to-osu!standard model, trained jointly from scratch. Lyra uses an encoder–decoder architecture: the map-generation loss also trains the audio encoder. There is no separate audio-training stage, frozen encoder, or required learned-feature preparation stage.

See the [Lyra v2.5 model card](MODEL_CARD.md) for the architecture, loss equations, dataset provenance and filtering, tokenizer, training configuration, intermediate results, and current limitations.

## Getting started

Run **StartLyra.cmd**. The local interface is available at **http://127.0.0.1:7862**.

On this machine, the launcher can use the existing CUDA-enabled Python environment. On another machine, `Install.ps1` installs dependencies in the user's `AppData/Local/Lyra/venv` directory. It does not copy the older project or create a virtual environment inside the source directory.

- **Training:** name the run and set the total number of dataset cycles. Start/Resume automatically uses that run's `last.pt`. The cycle count is the total goal: changing it from three completed cycles to five requests two additional cycles.
- **Create map:** select a Lyra `best.pt` checkpoint, music, target star rating, and style. The output is an importable `.osz` archive. If automatic timing is uncertain, provide BPM/offset or a reference `.osu` timing map.
- **Data:** index every usable map and every Lyra window from the existing full-dataset manifest. Interrupted preparation can resume. The existing song-disjoint train/validation/test assignment is preserved.

Training runs in a separate background process. Reopening or refreshing the frontend, or restarting the UI server, reconnects to the saved job and loss history. Temporary network errors trigger automatic reconnection. The browser connection does not control the training process's lifetime. After a machine or Python-process shutdown, resume from `last.pt`.

## Complete training history and statistics

Every completed optimizer update is recorded in the run's `metrics.sqlite` database. The chart downsamples displayed points; it does not discard stored history. Metrics include UTC timestamp, step, elapsed time, joint/token/beat losses, validation results, learning rate, gradient norm and AMP scale, update/loading/training time, token and window throughput, GPU memory, cache hits/misses/decodes, and dataset coverage.

The Training tab offers **CSV metrics** and **Full JSONL history** downloads. Exports stream directly from the database, so a complete history does not require continuously maintained duplicate files. Worker text logs remain in the `jobs` directory.

After an unexpected shutdown, a checkpoint may be older than the last logged update. When those updates are repeated after resume, their previous metrics are retained in the `archived_metrics` table. The normal chart and CSV show the resumed model's active trajectory; the full JSONL export also includes archived rows marked `rolled_back_update`. An orderly stop and resume preserves the existing history.

## Model and context

| Setting | Value |
|---|---|
| Jointly trained parameters | **120,612,482** |
| Audio encoder / decoder | 8 / 12 Transformer layers |
| Width / heads / feed-forward width | 640 / 10 / 2,304 |
| Audio | 22.05 kHz mono, 128 mel bands, approximately 10 ms frame interval |
| Audio window | 64 seconds |
| Recent map history | 32 seconds, up to 2,048 raw event tokens |
| Typical generation section | **24 seconds** |
| Decoder context | 6,144 tokens |
| Older history | Up to 128 past-pattern descriptors, projected by the same model |
| Training | FP16, AdamW, gradient checkpointing; effective batch 32 |
| Default profile | Physical batch **4**, four data-loader workers, 8 GiB total RAM-array cache |

Generation retains the previously created map. Recent raw history enters the decoder token stream directly; summaries of older history use only completed past objects. Training and generation share the same context builder. Active sliders crossing section boundaries continue into the next section. Dense sections can be shortened at complete object boundaries, with the remaining interval generated afterward.

This provides a longer direct context than the earlier eight-second task. Pattern consistency and playability must still be assessed on the trained model's outputs; architecture and memory measurements alone do not establish either.

## Full local dataset

Lyra indexing **completed on 19 October 2026**, with no additional map exclusions from the already prepared source corpus:

| Split | Maps | Lyra windows |
|---|---:|---:|
| Train | 133,440 | 964,841 |
| Validation | 7,526 | 53,947 |
| Test | 6,998 | 49,618 |
| **Total** | **147,964** | **1,068,406** |

The source preparation had already rejected 3,087 of 151,051 selected rows. The model card explains that filtering and the retained corpus's **33,994 audio identities**, **1,791.94 distinct-audio hours**, and approximately **175.20 GB of referenced source files**.

The existing split targets 90/5/5 by song group. Unequal group sizes produce slightly different map proportions. Every training window is visited once per shuffled cycle; filling the final batch can begin the next cycle. There is no additional training subset selected by star band, map-count limit, or cache capacity. Validation and test recordings do not receive weight updates.

Three training cycles at effective batch 32 require **90,454 optimizer updates**. There is no wall-clock limit that cuts this goal short. The initial learning-rate schedule spans at least 100,000 updates. Resume restores the original optimizer, scaler, schedule, random state, and next window selection. Validation samples fixed windows from the complete validation timeline, including song middles and endings.

**Every 1,000 updates, training attempts to generate an external-audio preview** in `runs/main/samples/step-<step>`, exported as an importable `.osz` when successful. The main run uses the **40-second excerpt from 00:50 to 01:30** of `C:/Users/cride/Desktop/music.mp3`. A prior check found that the original recording's exact decoded-PCM hash was absent from the dataset. This does not establish the absence of every alternate edit or recording.

The excerpt is frozen as a separate `.ogg` in `F:/osu_dataset/_lyra/examples/music-50-90`; the training setting `sample_audio_example` points to its `example.json`. Moving the original desktop file does not affect future previews. The excerpt is not added to the training dataset.

Timing and objects are generated from the excerpt without a reference map. Preview settings stay fixed: 6.5 requested stars, Auto style, seed 2026, and temperature 0.6. Mel arrays use the existing quota-managed SSD cache. The playable excerpt starts at zero; its original 50–90-second source range, training step, checksums, timing report, and generation settings are stored in `generation.json` inside the archive. The earlier full-song Hakai preview is retained; subsequent periodic previews use the external excerpt.

The `sample_audio_example` setting survives resume. Without an external excerpt, `sample_map_id` can select a dataset map; otherwise, the first validation map is used. Generation failures produce a detailed `rejected.json`, and training continues. If training stopped after saving an optimizer checkpoint but before completing its preview, resume attempts to recover that missing preview. Independent validation continues alongside these qualitative samples.

The run is still training. Successful previews, failed previews, and teacher-forced losses are documented separately in the model card; they do not establish a completed quality benchmark.

## SSD and RAM caches

The input pipeline uses globally aligned **4,096-frame FP16 mel blocks**, parsed maps, a byte-bounded RAM LRU, a quota-managed SSD cache, separate data-loader processes, length-aware microbatches, and asynchronous GPU transfers. Missing blocks are created under per-recording locks so parallel workers do not decode the same recording simultaneously.

Compatible mel blocks and timing targets from the older cache are borrowed read-only. The model does not use the older learned encoder features. Its own SSD cache defaults to 180 decimal GB and maintains at least 8 GiB of free-space reserve. Eviction replaces only its own derived entries. A sample remains eligible even if its arrays cannot be stored in the cache. Cache capacity never determines dataset membership.

A previously verified validation-window load took **0.344 s cold** and **0.0103 s from RAM**, with identical mel values. This is a single-window loading measurement, not an end-to-end training speedup claim.

## RTX 3050 8 GB verification

The saved benchmark performs real forward/backward/AdamW updates at the full **64-second / 6,144-token** shape, with checkpointing. After warmup, three updates were measured:

| Physical batch | Seconds / maximum-size microbatch | Windows / s | Peak PyTorch allocation |
|---|---:|---:|---:|
| 1 | 1.292 | 0.774 | 2.05 GiB |
| **4** | **4.901** | **0.816** | **3.40 GiB** |
| 8 | 9.911 | 0.807 | 5.37 GiB |

Batch 4 had the highest throughput of the three tested sizes. Batch 8 left only 0.55 GiB of free VRAM, so batch 4 is the default. These are synthetic maximum-shape microbatch measurements. Actual optimizer-update time also depends on the effective batch, sequence lengths, cache behavior, validation, and sample generation.

Automated checks cover causal decoding and KV-cache equivalence, mapping gradients reaching the audio encoder, timing-head information separation, shared training/generation context, complete event-window coverage, song-disjoint splits, warm/cold cache equivalence, exact optimizer resume, self-contained `.osz` export, durable workers, and frontend reconnection.

## Files and commands

The repository contains source code, the interface, tests, launch/install scripts, documentation, and the model-card evidence snapshot. Runtime data is stored separately:

- `F:/osu_dataset/_lyra/dataset`: persistent Lyra index and dataset manifest.
- `F:/osu_dataset/_lyra/runs/main`: `last.pt`, `best.pt`, metrics database, and previews.
- `F:/osu_dataset/_lyra/jobs`: background-job records and logs.
- `F:/osu_dataset/_lyra/generated`: exported `.osz` maps.
- `F:/osu_cache/lyra`: replaceable mel/map caches and temporary export work.

The original dataset and older V1/V2 project remain separate. Lyra does not import modules from the older project for training or generation. The launcher can reuse that project's existing Python environment.

~~~powershell
python -m lyra ui --port 7862 --open-browser
python -m lyra prepare
python -m lyra benchmark --batch-size 4
python -m lyra train --cycles 3 --batch-size 4 --workers 4
python -m lyra train --cycles 3 --batch-size 4 --resume F:/osu_dataset/_lyra/runs/main/last.pt
python -m lyra generate F:/osu_dataset/_lyra/runs/main/best.pt F:/Music/song.mp3 --stars 5
python -m lyra evaluate F:/osu_dataset/_lyra/runs/main/best.pt
python -m pytest -q -p no:cacheprovider
~~~

Use the resume command for an existing run. GPU benchmark, generation, and evaluation jobs require the training worker to release its GPU lease first.

The Lyra architecture requires its own training; V1/V2 checkpoints cannot be converted by renaming them. For generation, a Lyra checkpoint works with the Lyra runtime and dependencies without a separate audio checkpoint, learned-feature manifest, or access to the training maps.
