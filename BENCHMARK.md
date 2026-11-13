# Nepali speech recognition benchmark

Nepali speech recognition systems measured on the same audio, with the same scorer. Kept by
[Fiste AI](https://fiste-ai.github.io/shravya/) alongside [Shravya](https://huggingface.co/fiste-ai/shravya).
Last measured 13 November 2026. We add systems over time: open an issue to suggest one or to correct a
number.

## Five test sets

Word error rate in percent, lower is better. Each cell is **WER / nWER**. Bold marks the lowest value in a
column. Sorted by mean WER.

| Model | Size | Mean | FLEURS | IndicVoices | IndicVoices (phone) | Podcast | Code-switch |
|---|---|---|---|---|---|---|---|
| **[Shravya 0.1](https://huggingface.co/fiste-ai/shravya)** | 120M | **23.5** / **20.4** | 23.9 / **18.4** | 11.8 / 11.1 | 13.8 / 13.1 | 34.1 / 31.8 | **33.7** / **27.6** |
| [IndicConformer 600M](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) | 600M | 27.3 / 23.8 | **23.7** / 18.7 | 15.0 / 14.0 | 17.9 / 17.0 | 41.7 / 35.9 | 38.4 / 33.3 |
| [Rani-IndicConformer (Ampixa Labs)](https://github.com/Ampixa/rani-asr) | 129M | 27.5 / 24.0 | 24.9 / 19.5 | 12.2 / 11.5 | 15.1 / 14.3 | 44.9 / 39.1 | 40.4 / 35.6 |
| [IndicConformer Nepali](https://huggingface.co/ai4bharat/indicconformer_stt_ne_hybrid_ctc_rnnt_large) | 129M | 27.7 / 24.4 | 24.8 / 19.4 | **11.3** / **10.6** | **13.6** / **12.9** | 48.8 / 43.5 | 40.1 / 35.4 |
| [SraVaani 1.0](https://huggingface.co/ARTPARK-IISc/SraVaani-1.0) | 430M | 29.7 / 26.2 | 25.9 / 20.9 | 13.9 / 13.0 | 19.1 / 18.1 | 45.0 / 39.3 | 44.5 / 39.5 |
| [Kriti Telephony v1](https://github.com/Naamche-Labs/kriti-telephony/tree/v1) | 129M | 31.2 / 23.8 | 31.8 / 19.8 | 25.1 / 17.2 | 27.2 / 19.9 | 28.3 / 26.4 | 43.5 / 35.6 |
| [Rani (Ampixa Labs)](https://github.com/Ampixa/rani-asr) | 121M | 32.1 / 27.7 | 33.8 / 26.9 | 24.9 / 22.8 | 29.3 / 27.4 | 32.5 / 27.3 | 39.8 / 34.2 |
| [IndicConformer Nepali int8](https://huggingface.co/OpenVoiceOS/ai4bharat-indicconformer-ne-onnx) | 120M | 33.3 / 30.2 | 29.2 / 24.6 | 16.3 / 15.3 | 21.3 / 20.3 | 54.5 / 50.2 | 45.1 / 40.4 |
| [SraVaani code-mix (AI Lab)](https://huggingface.co/ailabpvtltd/sravaani-nepali-english-codemix-asr) | 449M | 38.2 / 30.3 | 34.5 / 25.2 | 34.0 / 27.5 | 39.6 / 32.7 | 39.3 / 32.0 | 43.6 / 34.0 |
| [IndicConformer code-mix (AI Lab)](https://huggingface.co/ailabpvtltd/indicconformer-nepali-english-codemix-asr) | 128M | 38.6 / 31.4 | 34.9 / 26.2 | 34.3 / 28.1 | 38.4 / 32.0 | 42.1 / 36.0 | 43.2 / 34.7 |
| [NepaliConformer offline (Ampixa Labs)](https://github.com/Ampixa/rani-asr) | 121M | 38.7 / 30.6 | 42.5 / 28.7 | 40.5 / 32.2 | 43.8 / 36.0 | **20.4** / **19.2** | 46.5 / 36.8 |
| [Whisper turbo code-mix (AI Lab)](https://huggingface.co/ailabpvtltd/whisper-large-v3-turbo-nepali-english-codemix-asr) | 809M | 42.0 / 34.5 | 39.2 / 29.6 | 39.3 / 34.0 | 46.9 / 40.9 | 39.8 / 32.6 | 44.9 / 35.6 |
| [NepaliConformer streaming (Ampixa Labs)](https://github.com/Ampixa/rani-asr) | 121M | 52.3 / 48.0 | 54.7 / 48.6 | 51.4 / 46.2 | 57.4 / 53.4 | 38.6 / 37.6 | 59.3 / 54.4 |
| [Ratna (Ampixa Labs)](https://huggingface.co/ampixa/ratna-asr) | 32M | 54.2 / 51.5 | 56.9 / 52.4 | 48.3 / 46.3 | 53.9 / 51.8 | 46.0 / 45.3 | 65.6 / 61.7 |
| [Sagarmatha v4](https://huggingface.co/tonibirat/sagarmatha-v4-nepali-asr) | 809M | 55.4 / 52.1 | 38.5 / 31.9 | 52.3 / 50.5 | 59.6 / 58.3 | 62.7 / 59.7 | 64.1 / 60.2 |
| [Nepali ASR aud8](https://huggingface.co/BlackfireAI/nepali-asr-aud8-325m) | 325M | 57.7 / 54.1 | 43.6 / 38.8 | 54.1 / 51.1 | 62.8 / 60.3 | 63.5 / 61.1 | 64.3 / 59.4 |
| [MMS 1B](https://huggingface.co/facebook/mms-1b-all) | 965M | 60.5 / 56.9 | 31.4 / 27.0 | 59.7 / 56.2 | 66.0 / 62.7 | 74.9 / 72.1 | 70.7 / 66.6 |
| [Nepali ASR XLS-R](https://huggingface.co/BlackfireAI/nepali-asr-xlsr-300m) | 315M | 61.1 / 58.7 | 37.0 / 32.7 | 56.3 / 54.5 | 66.2 / 64.6 | 74.7 / 72.4 | 71.4 / 69.0 |
| [SeamlessM4T v2 large](https://huggingface.co/facebook/seamless-m4t-v2-large) | 2.31B | 69.4 / 65.4 | 24.5 / 19.1 | 69.3 / 66.8 | 80.2 / 77.9 | 89.2 / 86.8 | 83.6 / 76.6 |
| [Nepali ASR Whisper turbo](https://huggingface.co/sumanpaudel1997/nepali-asr-whisper-turbo) | 809M | 72.2 / 68.8 | 39.4 / 33.6 | 63.0 / 60.4 | 71.1 / 69.0 | 111.4 / 108.6 | 76.3 / 72.5 |

| Set | Clips | What it is |
|---|---|---|
| FLEURS | 726 | read sentences, the official Nepali (`ne_np`) test split |
| IndicVoices | 1,000 | spontaneous, conversational and read speech from 11 speakers of the IndicVoices Nepali validation set |
| IndicVoices (phone) | 1,000 | the same clips through a simulated telephone channel (300 to 3400 Hz band, G.711 companding, packet drops, line noise) |
| Podcast | 798 | clips from two episodes of `Bijay13/nepali-podcast-code-switch-asr-dataset`, references in verbatim spelling |
| Code-switch | 1,440 | lecture clips with English mixed in, the test split of `saileshbro/nepali-cs-asr`, English words written in Devanagari |

## Three more sets, for reporting only

| Model | Size | NepTel 0.2 (calls) | PVSpeech (read) | New read speech |
|---|---|---|---|---|
| [Rani (Ampixa Labs)](https://github.com/Ampixa/rani-asr) | 121M | **33.6** / **31.3** | 18.1 / 14.1 | 22.4 / 16.6 |
| [Kriti Telephony v1](https://github.com/Naamche-Labs/kriti-telephony/tree/v1) | 129M | 37.9 / 34.5 | 16.8 / 11.2 | n/a |
| **[Shravya 0.1](https://huggingface.co/fiste-ai/shravya)** | 120M | 38.2 / 34.6 | 16.2 / 12.8 | 18.4 / **11.0** |
| [NepaliConformer offline (Ampixa Labs)](https://github.com/Ampixa/rani-asr) | 121M | 39.6 / 36.1 | 20.5 / 13.6 | 29.6 / 23.3 |
| [Rani-IndicConformer (Ampixa Labs)](https://github.com/Ampixa/rani-asr) | 129M | 41.2 / 38.6 | 19.7 / 17.3 | 17.9 / 12.4 |
| [SraVaani 1.0](https://huggingface.co/ARTPARK-IISc/SraVaani-1.0) | 430M | 42.9 / 40.6 | 24.7 / 20.5 | 20.0 / 13.9 |
| [IndicConformer 600M](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) | 600M | 47.5 / 45.4 | **12.5** / **9.6** | **16.5** / 11.7 |
| [IndicConformer Nepali](https://huggingface.co/ai4bharat/indicconformer_stt_ne_hybrid_ctc_rnnt_large) | 129M | 47.6 / 45.6 | 22.1 / 19.4 | 18.0 / 12.2 |
| [IndicConformer code-mix (AI Lab)](https://huggingface.co/ailabpvtltd/indicconformer-nepali-english-codemix-asr) | 128M | 48.7 / 44.6 | 21.3 / 17.8 | 26.3 / 19.3 |
| [SraVaani code-mix (AI Lab)](https://huggingface.co/ailabpvtltd/sravaani-nepali-english-codemix-asr) | 449M | 49.0 / 44.1 | 18.1 / 13.8 | 24.7 / 17.3 |
| [Whisper turbo code-mix (AI Lab)](https://huggingface.co/ailabpvtltd/whisper-large-v3-turbo-nepali-english-codemix-asr) | 809M | 54.3 / 50.1 | 22.9 / 19.4 | 30.4 / 22.3 |
| [NepaliConformer streaming (Ampixa Labs)](https://github.com/Ampixa/rani-asr) | 121M | 63.3 / 61.7 | 38.8 / 34.6 | 40.4 / 34.6 |
| [Ratna (Ampixa Labs)](https://huggingface.co/ampixa/ratna-asr) | 32M | 67.5 / 66.2 | 43.1 / 39.9 | 45.3 / 40.6 |
| [Nepali ASR aud8](https://huggingface.co/BlackfireAI/nepali-asr-aud8-325m) | 325M | 79.2 / 78.5 | 25.3 / 23.9 | n/a |
| [MMS 1B](https://huggingface.co/facebook/mms-1b-all) | 965M | 81.6 / 79.8 | 32.7 / 29.0 | n/a |
| [Sagarmatha v4](https://huggingface.co/tonibirat/sagarmatha-v4-nepali-asr) | 809M | 91.5 / 90.4 | 33.5 / 30.9 | n/a |
| [Nepali ASR Whisper turbo](https://huggingface.co/sumanpaudel1997/nepali-asr-whisper-turbo) | 809M | 94.8 / 94.7 | 24.2 / 20.5 | 28.3 / 18.9 |

| Set | Clips | What it is |
|---|---|---|
| NepTel 0.2 | 78 | segments of real telephone calls, published by Ampixa Labs for evaluation only |
| PVSpeech | 50 | read clips, published by Ampixa Labs |
| New read speech | 172 | sentences recorded for us in November 2026 |

"n/a" means not measured.

## Paired differences to Shravya

The other system's error rate minus Shravya's on the five-set mean, in points, with 95% intervals. Positive means
the other system makes more errors. The last three columns sort the five sets by whether the interval of the WER
difference lies above zero, includes zero or lies below zero.

| Compared with | Mean WER, points | Mean nWER, points | Shravya has fewer errors on | No clear difference on | Shravya has more errors on |
|---|---|---|---|---|---|
| IndicConformer 600M | +3.9 [+3.2, +4.7] | +3.4 [+2.7, +4.3] | IndicVoices, IndicVoices (phone), Podcast, Code-switch | FLEURS | none |
| Rani-IndicConformer (Ampixa Labs) | +4.0 [+3.7, +4.5] | +3.6 [+3.2, +4.3] | FLEURS, IndicVoices (phone), Podcast, Code-switch | IndicVoices | none |
| IndicConformer Nepali | +4.2 [+3.8, +4.8] | +4.0 [+3.5, +4.7] | FLEURS, Podcast, Code-switch | IndicVoices (phone) | IndicVoices |
| Kriti Telephony v1 | +7.7 [+7.3, +8.1] | +3.4 [+3.0, +4.0] | FLEURS, IndicVoices, IndicVoices (phone), Code-switch | none | Podcast |
| SraVaani 1.0 | +6.2 [+5.6, +7.2] | +5.8 [+5.2, +6.8] | FLEURS, IndicVoices, IndicVoices (phone), Podcast, Code-switch | none | none |
| Rani (Ampixa Labs) | +8.6 [+7.9, +9.2] | +7.3 [+6.7, +8.0] | FLEURS, IndicVoices, IndicVoices (phone), Code-switch | none | Podcast |
| NepaliConformer offline (Ampixa Labs) | +15.2 [+14.5, +16.0] | +10.2 [+9.3, +11.3] | FLEURS, IndicVoices, IndicVoices (phone), Code-switch | none | Podcast |
| SeamlessM4T v2 large | +45.9 [+42.5, +50.1] | +45.0 [+41.6, +49.4] | IndicVoices, IndicVoices (phone), Podcast, Code-switch | FLEURS | none |
| SraVaani code-mix (AI Lab) | +14.7 [+13.3, +16.5] | +9.9 [+8.6, +11.7] | FLEURS, IndicVoices, IndicVoices (phone), Podcast, Code-switch | none | none |
| IndicConformer code-mix (AI Lab) | +15.1 [+14.0, +16.5] | +11.0 [+9.9, +12.5] | FLEURS, IndicVoices, IndicVoices (phone), Podcast, Code-switch | none | none |
| NepaliConformer streaming (Ampixa Labs) | +28.8 [+27.9, +29.8] | +27.6 [+26.6, +28.7] | FLEURS, IndicVoices, IndicVoices (phone), Podcast, Code-switch | none | none |
| Ratna (Ampixa Labs) | +30.7 [+29.6, +31.9] | +31.1 [+30.0, +32.4] | FLEURS, IndicVoices, IndicVoices (phone), Podcast, Code-switch | none | none |

## Shravya Live (preview)

[Shravya Live](https://huggingface.co/fiste-ai/shravya#shravya-live-preview) is Shravya's model for live speech.
It writes text every 640 or 320 ms while the sound comes in, and reads each phrase once more when the speaker
pauses. The same test sets, each cell **WER / nWER**.

| Shravya Live | Mean | FLEURS | IndicVoices | IndicVoices (phone) | Podcast | Code-switch |
|---|---|---|---|---|---|---|
| Live text, every 640 ms | 28.4 / 25.8 | 31.4 / 26.7 | 17.1 / 16.2 | 21.4 / 20.6 | 35.3 / 34.0 | 36.8 / 31.3 |
| Live text, every 320 ms | 31.8 / 29.2 | 35.7 / 31.2 | 19.6 / 18.9 | 25.7 / 25.0 | 38.1 / 36.8 | 39.5 / 34.4 |
| Final text of a phrase | 23.5 / 20.8 | 24.5 / 19.6 | 13.6 / 12.8 | 16.1 / 15.3 | 30.9 / 29.6 | 32.5 / 26.7 |

| Shravya Live | NepTel 0.2 (calls) | PVSpeech (read) | New read speech |
|---|---|---|---|
| Live text, every 640 ms | 44.5 / 41.4 | 19.4 / 17.0 | 25.4 / 18.2 |
| Live text, every 320 ms | 47.0 / 44.6 | 19.7 / 16.2 | 29.6 / 23.4 |
| Final text of a phrase | 37.7 / 34.1 | 14.1 / 11.2 | 21.1 / 12.7 |

- **Live text** is what the stream had written when the clip ended, with no second reading.
- **Final text** is the text after the clip was read once more as a whole. `finish()` does that at the end of the
  sound, and `cut()` does it for a phrase when the speaker pauses.
- Each clip was fed to the published runtime in blocks of the step length, from the published files, on a
  processor. The spelling style is chosen as for Shravya.
- **Speed.** On one thread of an Intel Core i7-10700K (40 clips): 0.120 of real time at
  640 ms, 74 ms of work for each step; 0.175 of real time at 320 ms,
  55 ms for each step. The second reading of a phrase of 9 seconds takes
  0.7 seconds.
- **Checks.** On the 5,264 test clips the published files write exactly the text of the training
  toolkit 5,253 times at 640 ms and 5,256 times at 320 ms. On 24
  clips the text was the same whether the sound came as one block, in blocks of 100 ms or in blocks of random
  length. A recording of 20 minutes went through as one stream at a steady speed, and its
  memory levelled off at 1.7 GB.
- It is a preview: it has been measured on recordings, not yet on live calls.

### Streaming systems

The streaming systems we have measured, on the same test sets. The error rate is that of the text each system
writes live. Each cell **WER / nWER**, bold marks the lowest value in a column.

| Streaming system | Size | Step or look-ahead | Mean | FLEURS | IndicVoices | IndicVoices (phone) | Podcast | Code-switch |
|---|---|---|---|---|---|---|---|---|
| **[Shravya Live](https://huggingface.co/fiste-ai/shravya)** | 120M | 640 ms | **28.4** / **25.8** | **31.4** / **26.7** | **17.1** / **16.2** | **21.4** / **20.6** | **35.3** / **34.0** | **36.8** / **31.3** |
| **[Shravya Live](https://huggingface.co/fiste-ai/shravya)** | 120M | 320 ms | 31.8 / 29.2 | 35.7 / 31.2 | 19.6 / 18.9 | 25.7 / 25.0 | 38.1 / 36.8 | 39.5 / 34.4 |
| [NepaliConformer streaming (Ampixa Labs)](https://github.com/Ampixa/rani-asr) | 121M | 520 ms look-ahead | 52.3 / 48.0 | 54.7 / 48.6 | 51.4 / 46.2 | 57.4 / 53.4 | 38.6 / 37.6 | 59.3 / 54.4 |
| [Ratna (Ampixa Labs)](https://huggingface.co/ampixa/ratna-asr) | 32M | 320 ms | 54.2 / 51.5 | 56.9 / 52.4 | 48.3 / 46.3 | 53.9 / 51.8 | 46.0 / 45.3 | 65.6 / 61.7 |

| Streaming system | Size | Step or look-ahead | NepTel 0.2 (calls) | PVSpeech (read) | New read speech |
|---|---|---|---|---|---|
| **[Shravya Live](https://huggingface.co/fiste-ai/shravya)** | 120M | 640 ms | **44.5** / **41.4** | **19.4** / 17.0 | **25.4** / **18.2** |
| **[Shravya Live](https://huggingface.co/fiste-ai/shravya)** | 120M | 320 ms | 47.0 / 44.6 | 19.7 / **16.2** | 29.6 / 23.4 |
| [NepaliConformer streaming (Ampixa Labs)](https://github.com/Ampixa/rani-asr) | 121M | 520 ms look-ahead | 63.3 / 61.7 | 38.8 / 34.6 | 40.4 / 34.6 |
| [Ratna (Ampixa Labs)](https://huggingface.co/ampixa/ratna-asr) | 32M | 320 ms | 67.5 / 66.2 | 43.1 / 39.9 | 45.3 / 40.6 |

Paired differences, as in the section above: the other system's error rate minus that of Shravya Live.

| Shravya Live | Compared with | Mean WER, points | Mean nWER, points | Shravya Live has fewer errors on | No clear difference on | Shravya Live has more errors on |
|---|---|---|---|---|---|---|
| every 640 ms | NepaliConformer streaming (Ampixa Labs) | +23.9 [+23.1, +24.8] | +22.2 [+21.4, +23.3] | FLEURS, IndicVoices, IndicVoices (phone), Podcast, Code-switch | none | none |
| every 640 ms | Ratna (Ampixa Labs) | +25.8 [+24.8, +27.0] | +25.7 [+24.8, +27.0] | FLEURS, IndicVoices, IndicVoices (phone), Podcast, Code-switch | none | none |
| every 320 ms | NepaliConformer streaming (Ampixa Labs) | +20.5 [+19.7, +21.4] | +18.8 [+17.8, +19.9] | FLEURS, IndicVoices, IndicVoices (phone), Code-switch | Podcast | none |
| every 320 ms | Ratna (Ampixa Labs) | +22.4 [+21.5, +23.6] | +22.3 [+21.3, +23.6] | FLEURS, IndicVoices, IndicVoices (phone), Podcast, Code-switch | none | none |

- **Shravya Live makes the fewest errors of the three systems.** At 640 ms it is
  23.9 points ahead of the next one on the mean and ahead on every set, the three reporting-only sets
  included.
- **It is also the largest and the most expensive to run.** On one thread of the same processor and the same
  clips, Ratna takes 0.044 of real time and 14 ms of work for each of its
  steps of 320 ms. Shravya Live takes 2.7 times that at 640 ms and 4.0 times at
  320 ms. Ratna is about a quarter of its size.
- **How the others were run.** Ratna from its published files with its published runtime, in its own steps of
  320 ms. NepaliConformer streaming from its weights with its makers' toolkit, with its default context (520 ms
  of look-ahead by its makers' description), a whole clip at a time; that gives its makers' own published
  scores on FLEURS and PVSpeech. No streaming package of it is published, so its speed as a stream was not
  measured. Both are by [Ampixa Labs](https://github.com/Ampixa/rani-asr).
- The final text of Shravya Live is not in this table: it is not written live.

## Method

- **Decoding.** Each system was run by us from its weights, with its own default greedy decoding and no
  external language model, on the same 16 kHz audio files. The four models of Ampixa Labs are released by
  approval: [Rani (Ampixa Labs)](https://github.com/Ampixa/rani-asr), Rani-IndicConformer and the two older
  NepaliConformer models were decoded from the approved weights. Kriti Telephony is listed in its first version,
  the file its makers published in August 2026.
- **Shravya** was run from its published files with the published runtime, on a processor, one clip at a time.
  It uses its `verbatim` style on Podcast and NepTel 0.2, whose references are written verbatim, and `standard`
  elsewhere.
- **WER.** Reference and output are put in Unicode NFC, punctuation is removed, and words are compared by edit
  distance.
- **nWER.** Before comparing, digits are written as spoken Nepali words and spelling variants of one word are
  folded together: long and short i and u, the nasal marks, व and ब, the three s letters, ण and न, a final
  halant. It compares systems trained on different spelling conventions on content. It does not turn Latin
  letters into Devanagari.
- **Mean.** The unweighted mean of the five sets.
- **Intervals.** Paired bootstrap with 10,000 draws. Whole groups are resampled, not clips: sentences for FLEURS,
  speakers for IndicVoices, recordings for Code-switch, blocks of 20 consecutive clips for Podcast.

## What to keep in mind

- **Training overlap by kind of speech.** Shravya was trained on the training portions of FLEURS, IndicVoices,
  the podcast dataset and the code-switch dataset. Its test clips, episodes and speakers were kept out, and no
  training transcript equals a test reference or shares a run of eight words with one. Most other systems were
  not trained on the podcast or code-switch datasets (the AI Lab systems were trained on code-mixed speech).
- **Podcast.** The set is two whole episodes of a public podcast corpus. Kriti Telephony was trained on that
  corpus, by its makers' description, and 646 of the 798 clips of this set are in the corpus's own train split.
  NepaliConformer was trained, by its makers' description, on about 1,655 hours of mostly conversational YouTube
  speech, podcasts and interviews among it; whether these two episodes are part of that is not public. Shravya
  has heard neither episode.
- **IndicVoices.** The 11 test speakers are not in Shravya's training data. The multilingual IndicConformer
  models, and the systems fine-tuned from them (Shravya and Kriti Telephony among them), started from training
  data that is not published in detail and may contain these speakers.
- **Latin letters.** The three AI Lab systems write English words in Latin letters, which counts as errors
  against Devanagari references on Podcast and Code-switch.
- **Small sets.** NepTel 0.2 has 78 segments and PVSpeech 50 clips. Differences of a few points there are not
  resolved.
- **Not listed.** These systems gave no usable output in our pipeline and are left out until we find out why:
  whisper-large-v3-nepali.

## History

- 11 November 2026: first version with 16 systems, published with Shravya 0.1.
- 13 November 2026: Shravya Live 0.1 (preview) added, with a table of its own.
- 13 November 2026: Rani, Rani-IndicConformer and the two NepaliConformer models (Ampixa Labs) decoded from
  their weights, and a table of streaming systems. 20 systems.
