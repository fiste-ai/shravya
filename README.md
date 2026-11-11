# Shravya (श्रव्य)

**Nepali speech recognition by [Fiste AI](https://fiste-ai.github.io/shravya/).** Version 0.1, released 11 November 2026.

Shravya turns spoken Nepali into Devanagari text: conversations, phone calls, podcasts and lectures with English
words mixed in, as well as read speech. This repository holds the standalone runtime and the benchmark. The
weights are on Hugging Face at [fiste-ai/shravya](https://huggingface.co/fiste-ai/shravya).

- **Accurate across kinds of speech.** The lowest mean word error rate among the 16 publicly available
  Nepali speech recognition systems we have measured so far, ours included: 23.5% over five test sets
  (20.4% after spelling normalisation). The next system, IndicConformer 600M multilingual, is at 27.3%
  (23.8%).
- **Two writing styles.** `standard` spelling, or `verbatim`, the way it was said.
- **Standalone.** One Python file and three weight files. It needs `onnxruntime` and `numpy`, no deep learning
  framework and no graphics card, and runs about 28 times faster than real time on a desktop processor.

## Results

Word error rate in percent on the test sets, lower is better. Each cell is **WER / nWER**. Measured 11 November 2026.

| Model | Size | Mean | FLEURS | IndicVoices | IndicVoices (phone) | Podcast | Code-switch |
|---|---|---|---|---|---|---|---|
| **[Shravya 0.1](https://huggingface.co/fiste-ai/shravya)** | 120M | **23.5** / **20.4** | 23.9 / **18.4** | 11.8 / 11.1 | 13.8 / 13.1 | 34.1 / 31.8 | **33.7** / **27.6** |
| [IndicConformer 600M multilingual](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) | 600M | 27.3 / 23.8 | **23.7** / 18.7 | 15.0 / 14.0 | 17.9 / 17.0 | 41.7 / 35.9 | 38.4 / 33.3 |
| [IndicConformer Nepali](https://huggingface.co/ai4bharat/indicconformer_stt_ne_hybrid_ctc_rnnt_large) | 129M | 27.7 / 24.4 | 24.8 / 19.4 | **11.3** / **10.6** | **13.6** / **12.9** | 48.8 / 43.5 | 40.1 / 35.4 |
| [SraVaani 1.0](https://huggingface.co/ARTPARK-IISc/SraVaani-1.0) | 430M | 29.7 / 26.2 | 25.9 / 20.9 | 13.9 / 13.0 | 19.1 / 18.1 | 45.0 / 39.3 | 44.5 / 39.5 |
| [Kriti Telephony](https://huggingface.co/Aarjan/kriti-telephony) | 129M | 31.2 / 23.8 | 31.8 / 19.8 | 25.1 / 17.2 | 27.2 / 19.9 | **28.3** / **26.4** | 43.5 / 35.6 |
| [IndicConformer Nepali, int8 ONNX](https://huggingface.co/OpenVoiceOS/ai4bharat-indicconformer-ne-onnx) | 120M | 33.3 / 30.2 | 29.2 / 24.6 | 16.3 / 15.3 | 21.3 / 20.3 | 54.5 / 50.2 | 45.1 / 40.4 |
| [SraVaani Nepali-English code-mix](https://huggingface.co/ailabpvtltd/sravaani-nepali-english-codemix-asr) | 449M | 38.2 / 30.3 | 34.5 / 25.2 | 34.0 / 27.5 | 39.6 / 32.7 | 39.3 / 32.0 | 43.6 / 34.0 |
| [IndicConformer Nepali-English code-mix](https://huggingface.co/ailabpvtltd/indicconformer-nepali-english-codemix-asr) | 128M | 38.6 / 31.4 | 34.9 / 26.2 | 34.3 / 28.1 | 38.4 / 32.0 | 42.1 / 36.0 | 43.2 / 34.7 |
| [Whisper large-v3-turbo Nepali-English code-mix](https://huggingface.co/ailabpvtltd/whisper-large-v3-turbo-nepali-english-codemix-asr) | 809M | 42.0 / 34.5 | 39.2 / 29.6 | 39.3 / 34.0 | 46.9 / 40.9 | 39.8 / 32.6 | 44.9 / 35.6 |
| [Ratna (Ampixa Labs)](https://huggingface.co/ampixa/ratna-asr) | 32M | 54.2 / 51.5 | 56.9 / 52.4 | 48.3 / 46.3 | 53.9 / 51.8 | 46.0 / 45.3 | 65.6 / 61.7 |
| [Sagarmatha v4](https://huggingface.co/tonibirat/sagarmatha-v4-nepali-asr) | 809M | 55.4 / 52.1 | 38.5 / 31.9 | 52.3 / 50.5 | 59.6 / 58.3 | 62.7 / 59.7 | 64.1 / 60.2 |
| [Nepali ASR aud8](https://huggingface.co/BlackfireAI/nepali-asr-aud8-325m) | 325M | 57.7 / 54.1 | 43.6 / 38.8 | 54.1 / 51.1 | 62.8 / 60.3 | 63.5 / 61.1 | 64.3 / 59.4 |
| [Rani (Ampixa Labs)](https://github.com/Ampixa/rani-asr) | 121M | n/a | 33.8 / 26.8 | n/a | n/a | n/a | n/a |

WER is the plain word error rate. nWER is the same after writing numbers as words and folding spelling variants
together. Shravya was trained on the training portions of all five kinds of speech, and most other systems were
not trained on podcast or code-switched lecture speech. The full table, three more test sets, 95% intervals and
the method are in [BENCHMARK.md](BENCHMARK.md). The numbers are also in
[benchmark/results.json](benchmark/results.json). On telephone calls (NepTel 0.2)
[Rani (Ampixa Labs)](https://github.com/Ampixa/rani-asr) is ahead of Shravya, 33.5 against
38.2.

## Quick start

The weights are behind an access request on Hugging Face: sign in, fill in the form on the
[model page](https://huggingface.co/fiste-ai/shravya), and we approve by hand. Then:

```bash
pip install onnxruntime numpy soundfile soxr huggingface_hub
hf auth login
hf download fiste-ai/shravya --local-dir shravya-model
python shravya-model/shravya.py clip.wav
```

In Python:

```python
import sys
sys.path.insert(0, "shravya-model")
import shravya

model = shravya.load()                                   # finds the weights beside shravya.py
print(model.transcribe("clip.wav"))                      # standard spelling
print(model.transcribe("call.wav", style="verbatim"))    # written the way it was said
```

`transcribe` takes a sound file (wav, flac, ogg or mp3, any sample rate) or 16 kHz mono samples as a numpy array.
Recordings longer than 40 seconds are cut at quiet points and transcribed piece by piece.

The same `shravya.py` is in this repository. It is 181 lines and has no other dependency: log-mel
features and the transducer search are written in numpy, and the encoder runs in onnxruntime.

## What is here

| File | What it is |
|---|---|
| [`shravya.py`](shravya.py) | the runtime |
| [`BENCHMARK.md`](BENCHMARK.md) | every system we measured, the test sets, the scorer and the intervals |
| [`benchmark/results.json`](benchmark/results.json) | the same numbers for programs |

## Where it falls short

Calls are still hard (38.2% word error rate on NepTel 0.2). There is no punctuation, English words come
out in Devanagari, and numbers come out sometimes as digits and sometimes as words. It is offline only, with no
streaming mode yet. The [model card](https://huggingface.co/fiste-ai/shravya) has the full list, the training
data and the acknowledgements.

## Licence

The code in this repository is MIT. The weights are CC BY-NC 4.0 and available by request, see the model card.

श्रव्य (shravya) means audible, that which can be heard.
