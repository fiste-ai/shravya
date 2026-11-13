# Shravya (श्रव्य)

**Nepali speech recognition by [Fiste AI](https://fiste-ai.github.io/shravya/).** Speech in, Devanagari text out.

![Shravya by Fiste AI](https://fiste-ai.github.io/assets/og-shravya.png)

- **Accurate.** 23.5% word error rate averaged over five test sets, the lowest of the 20 Nepali
  systems in our benchmark.
- **Made for real speech.** Conversations, phone calls, podcasts and Nepali with English mixed in.
- **Light.** One Python file, no graphics card, 28 times faster than real time on a desktop processor.
- **Live.** [Shravya Live](#shravya-live-preview) writes the text while someone is still speaking, with the
  fewest errors of the streaming systems we measured. Preview.

[Model on Hugging Face](https://huggingface.co/fiste-ai/shravya) · [Website](https://fiste-ai.github.io/shravya/) ·
[Full benchmark](BENCHMARK.md) · [Request a demo](https://fiste-ai.github.io/contact/)

## Benchmark

Word error rate in percent, lower is better.

| Model | Size | Mean | FLEURS | IndicVoices | IndicVoices (phone) | Podcast | Code-switch |
|---|---|---|---|---|---|---|---|
| **[Shravya 0.1](https://huggingface.co/fiste-ai/shravya)** | 120M | **23.5** | 23.9 | 11.8 | 13.8 | 34.1 | **33.7** |
| [IndicConformer 600M](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) | 600M | 27.3 | **23.7** | 15.0 | 17.9 | 41.7 | 38.4 |
| [Rani-IndicConformer (Ampixa Labs)](https://github.com/Ampixa/rani-asr) | 129M | 27.5 | 24.9 | 12.2 | 15.1 | 44.9 | 40.4 |
| [IndicConformer Nepali](https://huggingface.co/ai4bharat/indicconformer_stt_ne_hybrid_ctc_rnnt_large) | 129M | 27.7 | 24.8 | **11.3** | **13.6** | 48.8 | 40.1 |
| [SraVaani 1.0](https://huggingface.co/ARTPARK-IISc/SraVaani-1.0) | 430M | 29.7 | 25.9 | 13.9 | 19.1 | 45.0 | 44.5 |
| [Kriti Telephony v1](https://github.com/Naamche-Labs/kriti-telephony/tree/v1) | 129M | 31.2 | 31.8 | 25.1 | 27.2 | **28.3** | 43.5 |
| [Rani (Ampixa Labs)](https://github.com/Ampixa/rani-asr) | 121M | 32.1 | 33.8 | 24.9 | 29.3 | 32.5 | 39.8 |
| [IndicConformer Nepali int8](https://huggingface.co/OpenVoiceOS/ai4bharat-indicconformer-ne-onnx) | 120M | 33.3 | 29.2 | 16.3 | 21.3 | 54.5 | 45.1 |
| [SraVaani code-mix (AI Lab)](https://huggingface.co/ailabpvtltd/sravaani-nepali-english-codemix-asr) | 449M | 38.2 | 34.5 | 34.0 | 39.6 | 39.3 | 43.6 |
| [IndicConformer code-mix (AI Lab)](https://huggingface.co/ailabpvtltd/indicconformer-nepali-english-codemix-asr) | 128M | 38.6 | 34.9 | 34.3 | 38.4 | 42.1 | 43.2 |

Shravya was trained on speech like these five test sets, and most of the other systems were not. All
20 systems, three more test sets and the method are in [BENCHMARK.md](BENCHMARK.md).

## Use it

Request access on the [model page](https://huggingface.co/fiste-ai/shravya). We approve by hand. Then:

```bash
pip install onnxruntime numpy soundfile soxr huggingface_hub
hf auth login
hf download fiste-ai/shravya --local-dir shravya-model
python shravya-model/shravya.py clip.wav
```

```python
import sys
sys.path.insert(0, "shravya-model")
import shravya

model = shravya.load()
print(model.transcribe("clip.wav"))                      # standard spelling
print(model.transcribe("call.wav", style="verbatim"))    # the way it was said
```

[`shravya.py`](shravya.py) is the whole runtime: one file that needs `onnxruntime` and `numpy`.

## Shravya Live (preview)

Shravya Live writes the text while someone is still speaking. When the speaker pauses it reads the phrase once
more as a whole, and that is the text to keep. It is a second model, in the folder `live/` of the same download.

```python
model = shravya.load("shravya-model/live")
stream = model.stream()                    # text every 640 ms; chunk_ms=320 answers sooner
for block in microphone:                   # 16 kHz mono float samples, blocks of any length
    print(stream.feed(block))              # the phrase so far
    if stream.pause > 0.8:                 # the speaker has stopped for a moment
        print(stream.cut())                # the phrase read once more: the text to keep
print(stream.finish())                     # the same for the last phrase
```

To try it on a recording: `python shravya-model/shravya.py --model shravya-model/live --stream clip.wav`

Word error rate in percent on the five test sets above, lower is better.

| Shravya Live | Mean | FLEURS | IndicVoices | IndicVoices (phone) | Podcast | Code-switch |
|---|---|---|---|---|---|---|
| Live text, every 640 ms | 28.4 | 31.4 | 17.1 | 21.4 | 35.3 | 36.8 |
| Live text, every 320 ms | 31.8 | 35.7 | 19.6 | 25.7 | 38.1 | 39.5 |
| Final text of a phrase | 23.5 | 24.5 | 13.6 | 16.1 | 30.9 | 32.5 |

On one thread of a desktop processor it runs 8 times faster than real time
(5.7 times at 320 ms). It is a preview: measured on recordings fed to it in small blocks, not yet
on live calls.

Beside the other streaming systems we have measured, the text each one writes live:

| Streaming system | Size | Step or look-ahead | Mean | FLEURS | IndicVoices | IndicVoices (phone) | Podcast | Code-switch |
|---|---|---|---|---|---|---|---|---|
| **[Shravya Live](https://huggingface.co/fiste-ai/shravya)** | 120M | 640 ms | **28.4** | **31.4** | **17.1** | **21.4** | **35.3** | **36.8** |
| **[Shravya Live](https://huggingface.co/fiste-ai/shravya)** | 120M | 320 ms | 31.8 | 35.7 | 19.6 | 25.7 | 38.1 | 39.5 |
| [NepaliConformer streaming (Ampixa Labs)](https://github.com/Ampixa/rani-asr) | 121M | 520 ms look-ahead | 52.3 | 54.7 | 51.4 | 57.4 | 38.6 | 59.3 |
| [Ratna (Ampixa Labs)](https://huggingface.co/ampixa/ratna-asr) | 32M | 320 ms | 54.2 | 56.9 | 48.3 | 53.9 | 46.0 | 65.6 |

Shravya Live makes the fewest errors of the three, on every test set at 640 ms. It is also the
largest: Ratna is about a quarter of its size and takes 2.7 times less processor time. More in
[BENCHMARK.md](BENCHMARK.md#streaming-systems).

## Licence

Code MIT. Weights CC BY-NC 4.0, by request on Hugging Face.

## Using Shravya at work

The weights are non-commercial. For a product or a paid service, a demo on your own recordings, or a pilot in a
call centre or a newsroom, write to us: **[request a demo](https://fiste-ai.github.io/contact/)** ·
<fistelabs@gmail.com>. Bugs and questions belong in the [issues](https://github.com/fiste-ai/shravya/issues).

श्रव्य (shravya) means audible.
