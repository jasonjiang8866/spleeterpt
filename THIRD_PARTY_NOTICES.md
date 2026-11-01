# Third-party notices

`src/spleeterpt/model.py`, the spectrogram/masking conventions in `audio.py`, and
the model presets are adapted from Deezer Spleeter, copyright (c) 2019-present,
Deezer SA, under the MIT License. The complete upstream notice is preserved in
[licenses/Deezer-Spleeter-MIT.txt](licenses/Deezer-Spleeter-MIT.txt).

Reference revision: `c8854001ac8acad34a9bc2bd15f28475541828b1`.

Primary implementation references:

- https://github.com/deezer/spleeter/blob/c8854001ac8acad34a9bc2bd15f28475541828b1/spleeter/model/functions/unet.py
- https://github.com/deezer/spleeter/blob/c8854001ac8acad34a9bc2bd15f28475541828b1/spleeter/model/__init__.py
- https://github.com/deezer/spleeter/blob/c8854001ac8acad34a9bc2bd15f28475541828b1/spleeter/audio/spectrogram.py
- https://github.com/deezer/spleeter/tree/c8854001ac8acad34a9bc2bd15f28475541828b1/spleeter/resources

Spleeter paper: Romain Hennequin, Anis Khlif, Felix Voituret, Manuel Moussallam,
“Spleeter: a fast and efficient music source separation tool with pre-trained
models,” Journal of Open Source Software 5(50), 2154 (2020).
https://doi.org/10.21105/joss.02154

Official pretrained archives remain hosted by Deezer. This project redistributes
converted 2/4-stem inference checkpoints as GitHub Release assets, with original
and converted SHA-256 hashes, source attribution, validation evidence, and the
complete MIT notice. The JOSS paper explicitly includes pretrained models in its
MIT distribution statement (page 3); that is the recorded licensing basis. The
README's code-only wording and still-open upstream clarification issue #959 are
documented in [the model card](docs/pretrained-release.md). This is an independent
format conversion, not an official Deezer release or a newly trained model.
No third-party audio dataset is included in Git or the model release.
