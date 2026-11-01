# Spleeter alternatives: architecture, evidence and selection

Reviewed **2026-11-01 (Asia/Singapore)**. This is source research, not a new local
benchmark. The scope is six representative alternatives with public papers,
implementations or tool integrations. Open-Unmix is included as a research
reference; inclusion does not imply equal popularity or market share. MSST and
MVSep's catalogs demonstrate practical availability of several newer families
([R9], [R11]), not a universal quality ordering.

The recommendation for this repository is to retain converted official Spleeter
weights as its compatibility baseline, compare HTDemucs first for four stems,
and add a pinned vocal RoFormer in a separate vocal/accompaniment track. SCNet is
a useful subsequent efficiency candidate. These are our evaluation priorities.

## Architecture map

[![Architecture comparison of seven music source-separation families](assets/separator-architectures.svg)](assets/separator-architectures.svg)

These are original explanatory drawings, not copied paper figures. They omit
normalization, detailed block counts, chunking and most skip paths. They describe
representative masking/spectral versions, not every later variant of a family.
The column alignment compares processing stages; it does not imply equal model
size, latency, training cost or quality.

| Figure row | What the architecture changes | Source of the drawing |
|---|---|---|
| Spleeter / this port | Independent six-level U-Nets predict magnitudes; this port converts these to squared-magnitude ratio masks on the complex mixture | [Local model and synthesis](architecture.md); upstream [R1] also supports multichannel Wiener filtering |
| Open-Unmix | Dense frequency projection, three bidirectional LSTM layers per target, magnitude estimates, then joint multichannel Wiener reconstruction | Author implementation and model description [R2] |
| HTDemucs | Parallel waveform/spectral encoders exchange information through self- and cross-attention; decoded waveforms are summed | Paper [R3]; released code's complex-as-channels path [R4] |
| MDX23C / TFC-TDF v3 | Real/imaginary STFT channels are packed into channel-wise subbands; residual conv/dense U-Net blocks and a final head estimate complex spectra | KUIELab code [R5]; TFC means time-frequency convolution, TDF time-distributed fully connected |
| BS-RoFormer | Disjoint frequency bands become tokens; alternating temporal and band attention predicts complex masks | Paper [R6]; public reimplementation [R8] |
| Mel-Band RoFormer | Overlapping mel-spaced bin groups replace disjoint bands; overlapping mask estimates are averaged | Paper [R7], implementation [R8]; this is not simply feeding a lossy mel-spectrogram to a vocoder |
| SCNet | Stronger compression in higher bands, dual-path recurrent processing with feature FFT/iFFT conversion, skip fusion and sparse upsampling | Paper [R10]; current author code and checkpoints [R12] |

Complex-spectrum estimation/masking can alter phase. That is an architectural
capability, not a guarantee of fewer audible artifacts. Spleeter's ratio masks
retain mixture phase. Refer to [the README selection table](../README.md#spleeter-and-its-alternatives)
for the output targets and suggested uses of each family.

## Published quality evidence — not a common leaderboard

All entries below are **source-reported SDR in dB**, not measurements of downloaded
checkpoints here. Higher values are meaningful within the stated protocol. The
non-HQ and HQ audio, training data, reconstruction, aggregation, ensembles and
checkpoint identities differ. Do not subtract rows to claim a measured upgrade.

| Reported system / variant | Reported result | Data and protocol context | What the number does not establish |
|---|---|---|---|
| Spleeter soft masking | Vocals 6.55; drums 5.93; bass 5.10; other 4.24 | Original non-HQ MUSDB18 evaluation; private Deezer training data; [R1], separation-performance table | Not the port's current ten-song validation result |
| Spleeter + multichannel Wiener | Vocals 6.86; drums 6.71; bass 5.51; other 4.55 | Same publication; different reconstruction from the preceding row [R1] | This port does not implement that MWF path |
| Open-Unmix, historical reference | Vocals 6.32; drums 5.73; bass 5.23; other 4.02 | Original Spleeter paper's comparison [R1] | Not a score for today's default `umxl` weights |
| HTDemucs, dense / per-source fine-tuned | Four-stem average 8.80 / 9.00 | MUSDB18-HQ test; 800 extra training songs; paper Table 1 [R3] | Do not assign a paper variant's score to an arbitrary downloaded checkpoint |
| Sparse HTDemucs + fine-tuning | Four-stem average 9.20 | Same table, longer-context sparse variant [R3] | Original repository says this sparse model was not released [R4] |
| BS-RoFormer, L=6, overlap-and-average | Four-stem average 9.80 | MUSDB18-HQ, no extra training data; median SDR, paper Table 2 [R6] | Not the large competition system or every community weight |
| Mel-RoFormer, L=6 | Four-stem average 9.64; vocals 11.21 | MUSDB18-HQ, no extra data; paper Table 1 [R7]; matched BS-RoFormer L=6 in that paper: average 9.92, vocals 10.78 | Vocal gain is not a four-stem average win; bass is a weakness in this comparison |
| SCNet, selected sparse configuration | Four-stem average 9.00 | MUSDB18-HQ without extra data; museval SDR; paper Table 1 [R10] | Not a score for every SCNet-large / XL community model |
| MDX23C / TFC-TDF v3 | No single checkpoint score assigned here | KUIELab's linked release documents SDX23 A/B noise/bleed tracks and a three-model submission [R5] | An A/B ensemble result is not a clean MUSDB18-HQ single-model score |

The two BS-RoFormer numbers (9.80 and 9.92) come from **different papers/runs**;
they are deliberately not treated as the same experiment. SDX23 leaderboard C
global SDR is also distinct from its paper's MUSDB median SDR [R6]. SI-SDR and
BSSEval SDR must retain separate labels. The local [frozen-BN report](frozen-bn-41230412543.md)
uses ten songs held out from the MUSDB18 training pool, decoded from AAC; it is
neither the HQ benchmark nor the untouched 50-song test set above.

## Runtime, access and operational tradeoffs

| Family | What can be said from primary sources | What to measure or pin locally |
|---|---|---|
| Spleeter | Original GPU batch example: over 100× realtime on RTX 2080 with dual Xeon Gold 6134, including loading/export [R1] | Historical upstream result, not a speed claim for this PyTorch port or today's competitors; full STFT memory scales with song length |
| Open-Unmix | Bidirectional context plus multichannel filtering [R2] | Model version, filter iterations, CPU/GPU execution and song length |
| HTDemucs | `htdemucs_ft` is documented as about four times slower than `htdemucs`; segment and shift settings affect inference [R4] | Single model versus bag, shifts, overlap, segment length; record exact release |
| MDX23C | CNN/dense blocks and selectable subband/chunk configuration [R5] | Frequency coverage, chunk size, overlap and whether an ensemble is used |
| BS / Mel-Band RoFormer | Attention dimensions and target count vary across configurations [R8], [R9] | Weights plus matching YAML, attention backend, precision, chunk/overlap and target count |
| SCNet | Paper's selected model: CPU RTF 0.669 versus HTDemucs 1.38, single-thread Xeon Platinum 8372HC [R10] | Approximately 48% of that reference's time in that setup; not a GPU or Spleeter comparison |

No reliable **same-workload minimum VRAM** or cost ranking follows from these
sources. A training recipe using a 48/80 GB GPU is not an inference requirement.
Our prior 7.22 GiB peak covered an evaluation process containing multiple models;
it must not be presented as Spleeter's isolated inference footprint. Likewise,
24 GB capacity alone does not guarantee a particular RoFormer configuration fits.

### Releases and licensing metadata

This records upstream statements, not a blanket license for every downloaded
weight. Pin the actual artifact, configuration and applicable terms.

| Project | Available path / maintenance observation | License information verified in the linked source |
|---|---|---|
| Spleeter | Original TensorFlow models; this independent port converts official checkpoints | Author paper explicitly states MIT for source code and pretrained models [R1]; this repo retains [third-party notices](../THIRD_PARTY_NOTICES.md) |
| Open-Unmix | `umx` (AAC MUSDB), `umxhq` (HQ), `umxl` (private extra data) | Code MIT; **default `umxl` weights CC BY-NC-SA 4.0**; inspect other weight releases individually [R2] |
| HTDemucs | Meta repository archived; [author fork](https://github.com/adefossez/demucs) says no active feature work | Repository declares MIT; preserve notices and inspect the selected release's terms [R4] |
| TFC-TDF v3 / MDX23C | KUIELab submission and training code; MSST integrations | KUIELab code MIT; verify the selected community weight's terms separately [R5], [R9] |
| BS / Mel-Band RoFormer | Papers and lucidrains' community implementation; multiple independent weights | Implementation MIT; architecture name does not determine checkpoint license [R8], [R9] |
| SCNet | Current author repo includes training/inference and base/large checkpoints | Code MIT; the README alone does not specify a distinct checkpoint grant [R12] |

An older SCNet repository under `thuhcsi` still advertises training code as
forthcoming. Use the current `starrytong/SCNet` source linked by MSST when checking
availability. Do not confuse a paper implementation with reproduction of the
authors' data, weights or headline score.

### Applications and services are not architectures

| Entry point | Role | How it relates to this comparison |
|---|---|---|
| [Ultimate Vocal Remover / UVR][R13] | Local desktop GUI | Release notes list Demucs and MDX23C support; compare its selected model and settings, not “UVR” as one network |
| [MSST][R9] | Training / inference toolkit | Integrates many families and publishes a checkpoint catalog; useful for a common adapter layer |
| [MVSep][R11] | Hosted separation and ensembles | Catalog includes BS/Mel RoFormer and SCNet variants; provider-specific benchmarks and ensembles are not this repo's test results |

Closed services are not assigned guessed architectures. A separate product
comparison could assess upload/privacy terms, workflow and pricing; those are
outside this local-model selection and no service price is assumed here.

## Where Spleeter adds value

Spleeter is useful when stable behavior, familiar stem definitions and an
inspectable convolutional baseline matter: legacy pipeline migration, dataset
preprocessing for MIR, rough practice/remix drafts, and controlled training
experiments. Its 2/4/5-stem presets include a piano target in the five-stem model.
Availability of that target does not establish piano fidelity.

This **port's** additional value is checked 2/4-stem waveform compatibility,
TensorFlow-free runtime, checkpoint provenance and tested recovery on replacement
Pods. Its BN experiments provide a concrete research/debugging baseline. These
are engineering properties; they neither establish superiority over other open
toolkits nor turn the 90-step fine-tune into improved weights.

For final production stems, audition alternatives on representative material.
The default modeled band ends around 11 kHz and zeros higher-frequency masks;
`mask_extension="average"` changes behavior without learning that missing band.
Mixture-phase reuse, leakage/artifacts and non-streaming full-STFT memory remain
limitations. Inspect the [local architecture](architecture.md) and
[validation evidence](validation.md) before relying on a compatibility claim.

## Next comparison in this repository

This is a proposed protocol; **no new competitor inference or GPU spend was
performed for this research update**.

1. **Fix the tasks.** Four-stem track: converted official Spleeter-4 versus
   `htdemucs`, then official SCNet and a fully specified four-stem BS-RoFormer
   setup. Separate vocal/accompaniment track: Spleeter-2 and a named Mel-Band
   checkpoint. Explicitly record whether accompaniment is a residual or predicted
   output. Do not rank vocal-only models by four-stem average.
2. **Screen on the existing ten validation songs.** Keep the full 50-song test
   set untouched during selection. Record AAC/HQ provenance; a future HQ test
   needs verified HQ files, not files renamed from the AAC decode. Hold out test
   data from training, tuning and checkpoint selection across all candidates.
3. **Pin the complete recipe.** Code commit, weight/config SHA-256, training-data
   disclosure, stems/order, sample rate, frequency coverage, precision, segment,
   overlap, shifts, ensembles and attention backend. Reject `mdx_extra` and
   `mdx_extra_q` for a clean MUSDB test: Demucs documents training on its test set
   [R4]. These Demucs names do not mean MDX23C. Treat undisclosed training data as
   an unresolved comparability issue.
4. **Measure quality consistently.** Align full-length stereo audio without
   per-model normalization tricks. Use the same BSSEval-v4 implementation and
   separately labeled global stereo SI-SDR. Report per-song/per-stem values,
   paired deltas and the aggregation rule, including silent/invalid cases.
   Listen blind to fixed excerpts for bleed, transients, phase and vocal detail.
5. **Measure deployment cost separately.** On the same PRO 6000 MIG 24 GB profile,
   log cold loading, synchronized warm inference, end-to-end decode/export,
   audio duration, RTF, peak allocated/reserved VRAM and CPU RAM. Reset peaks
   between models and record concurrency. Use a controlled FP32 comparison where
   supported; report each model's recommended optimized recipe separately.
   Include startup/storage charges separately from GPU seconds × observed rate.
6. **Freeze the recipe before final testing.** Publish config/data hashes and
   per-track results. A speed/quality winner can only be named after that test;
   this literature survey supplies candidates, not the result.

## Figure regeneration

Documentation-only dependency: `matplotlib==3.10.8`; it is not a runtime package
dependency. From the repository root:

```bash
python scripts/draw_separator_architectures.py
python scripts/draw_separator_architectures.py --preview /tmp/separator-architectures.png
```

The generator writes an accessible, deterministic SVG with selectable text and a
self-contained light background. It contains a manually curated conceptual map,
not generated model introspection; review labels when updating a cited family.
The full-resolution SVG can be opened from the README image.

## Primary sources

Paper versions and repositories checked on the review date above. Branch links
can change; future benchmark recipes must pin commits and artifact hashes.

- **R1:** [Spleeter author paper and original comparison table][R1]; [JOSS DOI](https://doi.org/10.21105/joss.02154).
- **R2:** [Open-Unmix model, checkpoints and license notes][R2]; [model code](https://github.com/sigsep/open-unmix-pytorch/blob/master/openunmix/model.py).
- **R3:** [Hybrid Transformers for Music Source Separation, arXiv v1][R3], Table 1 and architecture section.
- **R4:** [Demucs model zoo, release and maintenance notes][R4]; [HTDemucs implementation](https://github.com/facebookresearch/demucs/blob/main/demucs/htdemucs.py).
- **R5:** [KUIELab SDX23 submission][R5]; [original TFC-TDF v3 implementation](https://github.com/kuielab/sdx23/blob/mdx_AB/my_submission/src/tfc_tdf_v3.py).
- **R6:** [Band-Split RoPE Transformer, arXiv v1][R6], Table 2 and mask-estimation section.
- **R7:** [Mel-Band RoFormer, arXiv v1][R7], Table 1 and overlapping-band projection.
- **R8:** [lucidrains BS / Mel-Band RoFormer implementation][R8]; [MIT license](https://github.com/lucidrains/BS-RoFormer/blob/main/LICENSE).
- **R9:** [MSST toolkit][R9]; [pretrained model catalog](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md).
- **R10:** [SCNet, arXiv v1][R10], architecture and CPU RTF Table 1.
- **R11:** [MVSep's own model and ensemble catalog][R11].
- **R12:** [Current official SCNet code and release links][R12]; [MIT code license](https://github.com/starrytong/SCNet/blob/main/LICENSE).
- **R13:** [UVR project][R13]; [release notes](https://github.com/Anjok07/ultimatevocalremovergui/releases).

[R1]: https://github.com/deezer/spleeter/blob/master/paper.md
[R2]: https://github.com/sigsep/open-unmix-pytorch
[R3]: https://arxiv.org/pdf/2211.08553v1
[R4]: https://github.com/facebookresearch/demucs
[R5]: https://github.com/kuielab/sdx23/tree/mdx_AB
[R6]: https://arxiv.org/pdf/2309.02612v1
[R7]: https://arxiv.org/pdf/2310.01809v1
[R8]: https://github.com/lucidrains/BS-RoFormer
[R9]: https://github.com/ZFTurbo/Music-Source-Separation-Training
[R10]: https://arxiv.org/pdf/2401.13276v1
[R11]: https://mvsep.com/en/algorithms
[R12]: https://github.com/starrytong/SCNet
[R13]: https://github.com/Anjok07/ultimatevocalremovergui
