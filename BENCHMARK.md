# Nepali speech recognition benchmark

Every Nepali speech recognition system we have measured, on the same audio, with the same scorer. Kept by
[Fiste AI](https://fiste-ai.github.io/shravya/) alongside [Shravya](https://huggingface.co/fiste-ai/shravya).
Last measured 11 November 2026. We add systems as we measure them: open an issue to suggest one or to correct a
number.

## Five test sets

Word error rate in percent, lower is better. Each cell is **WER / nWER**. Bold marks the lowest value in a
column. Sorted by mean WER.

| Model | Size | Mean | FLEURS | IndicVoices | IndicVoices (phone) | Podcast | Code-switch |
|---|---|---|---|---|---|---|---|
| **[Shravya 0.1](https://huggingface.co/fiste-ai/shravya)** | 120M | **23.5** / **20.4** | 23.9 / **18.4** | 11.8 / 11.1 | 13.8 / 13.1 | 34.1 / 31.8 | **33.7** / **27.6** |
| [IndicConformer 600M](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) | 600M | 27.3 / 23.8 | **23.7** / 18.7 | 15.0 / 14.0 | 17.9 / 17.0 | 41.7 / 35.9 | 38.4 / 33.3 |
| [IndicConformer Nepali](https://huggingface.co/ai4bharat/indicconformer_stt_ne_hybrid_ctc_rnnt_large) | 129M | 27.7 / 24.4 | 24.8 / 19.4 | **11.3** / **10.6** | **13.6** / **12.9** | 48.8 / 43.5 | 40.1 / 35.4 |
| [SraVaani 1.0](https://huggingface.co/ARTPARK-IISc/SraVaani-1.0) | 430M | 29.7 / 26.2 | 25.9 / 20.9 | 13.9 / 13.0 | 19.1 / 18.1 | 45.0 / 39.3 | 44.5 / 39.5 |
| [Kriti Telephony](https://huggingface.co/Aarjan/kriti-telephony) | 129M | 31.2 / 23.8 | 31.8 / 19.8 | 25.1 / 17.2 | 27.2 / 19.9 | **28.3** / **26.4** | 43.5 / 35.6 |
| [IndicConformer Nepali int8](https://huggingface.co/OpenVoiceOS/ai4bharat-indicconformer-ne-onnx) | 120M | 33.3 / 30.2 | 29.2 / 24.6 | 16.3 / 15.3 | 21.3 / 20.3 | 54.5 / 50.2 | 45.1 / 40.4 |
| [SraVaani code-mix (AI Lab)](https://huggingface.co/ailabpvtltd/sravaani-nepali-english-codemix-asr) | 449M | 38.2 / 30.3 | 34.5 / 25.2 | 34.0 / 27.5 | 39.6 / 32.7 | 39.3 / 32.0 | 43.6 / 34.0 |
| [IndicConformer code-mix (AI Lab)](https://huggingface.co/ailabpvtltd/indicconformer-nepali-english-codemix-asr) | 128M | 38.6 / 31.4 | 34.9 / 26.2 | 34.3 / 28.1 | 38.4 / 32.0 | 42.1 / 36.0 | 43.2 / 34.7 |
| [Whisper turbo code-mix (AI Lab)](https://huggingface.co/ailabpvtltd/whisper-large-v3-turbo-nepali-english-codemix-asr) | 809M | 42.0 / 34.5 | 39.2 / 29.6 | 39.3 / 34.0 | 46.9 / 40.9 | 39.8 / 32.6 | 44.9 / 35.6 |
| [Ratna (Ampixa Labs)](https://huggingface.co/ampixa/ratna-asr) | 32M | 54.2 / 51.5 | 56.9 / 52.4 | 48.3 / 46.3 | 53.9 / 51.8 | 46.0 / 45.3 | 65.6 / 61.7 |
| [Sagarmatha v4](https://huggingface.co/tonibirat/sagarmatha-v4-nepali-asr) | 809M | 55.4 / 52.1 | 38.5 / 31.9 | 52.3 / 50.5 | 59.6 / 58.3 | 62.7 / 59.7 | 64.1 / 60.2 |
| [Nepali ASR aud8](https://huggingface.co/BlackfireAI/nepali-asr-aud8-325m) | 325M | 57.7 / 54.1 | 43.6 / 38.8 | 54.1 / 51.1 | 62.8 / 60.3 | 63.5 / 61.1 | 64.3 / 59.4 |
| [MMS 1B](https://huggingface.co/facebook/mms-1b-all) | 965M | 60.5 / 56.9 | 31.4 / 27.0 | 59.7 / 56.2 | 66.0 / 62.7 | 74.9 / 72.1 | 70.7 / 66.6 |
| [Nepali ASR XLS-R](https://huggingface.co/BlackfireAI/nepali-asr-xlsr-300m) | 315M | 61.1 / 58.7 | 37.0 / 32.7 | 56.3 / 54.5 | 66.2 / 64.6 | 74.7 / 72.4 | 71.4 / 69.0 |
| [SeamlessM4T v2 large](https://huggingface.co/facebook/seamless-m4t-v2-large) | 2.31B | 69.4 / 65.4 | 24.5 / 19.1 | 69.3 / 66.8 | 80.2 / 77.9 | 89.2 / 86.8 | 83.6 / 76.6 |
| [Nepali ASR Whisper turbo](https://huggingface.co/sumanpaudel1997/nepali-asr-whisper-turbo) | 809M | 72.2 / 68.8 | 39.4 / 33.6 | 63.0 / 60.4 | 71.1 / 69.0 | 111.4 / 108.6 | 76.3 / 72.5 |
| [Rani (Ampixa Labs)](https://github.com/Ampixa/rani-asr) | 121M | n/a | 33.8 / 26.8 | n/a | n/a | n/a | n/a |

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
| [Rani (Ampixa Labs)](https://github.com/Ampixa/rani-asr) | 121M | **33.5** / **31.2** | 18.4 / 14.6 | n/a |
| [Kriti Telephony](https://huggingface.co/Aarjan/kriti-telephony) | 129M | 37.9 / 34.5 | 16.8 / 11.2 | n/a |
| **[Shravya 0.1](https://huggingface.co/fiste-ai/shravya)** | 120M | 38.2 / 34.6 | 16.2 / 12.8 | 18.4 / **11.0** |
| [SraVaani 1.0](https://huggingface.co/ARTPARK-IISc/SraVaani-1.0) | 430M | 42.9 / 40.6 | 24.7 / 20.5 | 20.0 / 13.9 |
| [IndicConformer 600M](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) | 600M | 47.5 / 45.4 | **12.5** / **9.6** | **16.5** / 11.7 |
| [IndicConformer Nepali](https://huggingface.co/ai4bharat/indicconformer_stt_ne_hybrid_ctc_rnnt_large) | 129M | 47.6 / 45.6 | 22.1 / 19.4 | 18.0 / 12.2 |
| [IndicConformer code-mix (AI Lab)](https://huggingface.co/ailabpvtltd/indicconformer-nepali-english-codemix-asr) | 128M | 48.7 / 44.6 | 21.3 / 17.8 | 26.3 / 19.3 |
| [SraVaani code-mix (AI Lab)](https://huggingface.co/ailabpvtltd/sravaani-nepali-english-codemix-asr) | 449M | 49.0 / 44.1 | 18.1 / 13.8 | 24.7 / 17.3 |
| [Whisper turbo code-mix (AI Lab)](https://huggingface.co/ailabpvtltd/whisper-large-v3-turbo-nepali-english-codemix-asr) | 809M | 54.3 / 50.1 | 22.9 / 19.4 | 30.4 / 22.3 |
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
| IndicConformer Nepali | +4.2 [+3.8, +4.8] | +4.0 [+3.5, +4.7] | FLEURS, Podcast, Code-switch | IndicVoices (phone) | IndicVoices |
| Kriti Telephony | +7.7 [+7.3, +8.1] | +3.4 [+3.0, +4.0] | FLEURS, IndicVoices, IndicVoices (phone), Code-switch | none | Podcast |
| SraVaani 1.0 | +6.2 [+5.6, +7.2] | +5.8 [+5.2, +6.8] | FLEURS, IndicVoices, IndicVoices (phone), Podcast, Code-switch | none | none |
| SeamlessM4T v2 large | +45.9 [+42.5, +50.1] | +45.0 [+41.6, +49.4] | IndicVoices, IndicVoices (phone), Podcast, Code-switch | FLEURS | none |
| SraVaani code-mix (AI Lab) | +14.7 [+13.3, +16.5] | +9.9 [+8.6, +11.7] | FLEURS, IndicVoices, IndicVoices (phone), Podcast, Code-switch | none | none |
| IndicConformer code-mix (AI Lab) | +15.1 [+14.0, +16.5] | +11.0 [+9.9, +12.5] | FLEURS, IndicVoices, IndicVoices (phone), Podcast, Code-switch | none | none |
| Ratna (Ampixa Labs) | +30.7 [+29.6, +31.9] | +31.1 [+30.0, +32.4] | FLEURS, IndicVoices, IndicVoices (phone), Podcast, Code-switch | none | none |

## Method

- **Decoding.** Each system was run by us from its public weights, with its own default greedy decoding and no
  external language model, on the same 16 kHz audio files. Two exceptions, both for
  [Rani (Ampixa Labs)](https://github.com/Ampixa/rani-asr), whose weights are not openly available: NepTel 0.2 is
  scored from the transcripts Ampixa Labs published, and FLEURS and PVSpeech from its public demo.
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
