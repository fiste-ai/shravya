"""Shravya (श्रव्य): Nepali speech recognition by Fiste AI. Standalone runtime.

It needs four small packages and nothing else:

    pip install onnxruntime numpy soundfile soxr

    import shravya
    model = shravya.load("path/to/model")                     # the folder with encoder.onnx, decoder.npz, config.json
    print(model.transcribe("clip.wav"))                       # standard written Nepali
    print(model.transcribe("call.wav", style="verbatim"))     # written the way it was said

Live speech, when the folder also holds the streaming encoder:

    stream = model.stream()                                   # text every 640 ms; chunk_ms=320 answers sooner
    for block in microphone:                                  # 16 kHz mono float samples, blocks of any length
        print(stream.feed(block))                             # the text of the phrase so far
        if stream.pause > 0.8:                                # the speaker has stopped for a moment
            print(stream.cut())                               # the phrase read once more as a whole: the text to keep
    print(stream.finish())                                    # the same for the last phrase, at the end of the sound

From a shell:

    python shravya.py clip.wav another.flac --style verbatim
    python shravya.py clip.wav --stream

Sound files of any sample rate and channel count are converted to 16 kHz mono. Recordings longer than
`max_seconds` are cut at quiet points and transcribed piece by piece.

Copyright (c) 2026 Fiste AI. This file is released under the MIT License. The model files have their own
licence: see the model card.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

__version__ = "0.2.0"
SAMPLE_RATE = 16000
STYLES = ("standard", "verbatim")
REPOSITORY = "fiste-ai/shravya"
FRAME_MS = 40  # one encoder frame
SUBSAMPLING = 4  # feature frames (10 ms) per encoder frame


def resample(samples, rate: int) -> np.ndarray:
    """Mono samples at any rate as 16 kHz float32."""
    samples = np.asarray(samples, dtype=np.float32)
    if rate == SAMPLE_RATE:
        return samples
    import soxr

    return np.asarray(soxr.resample(samples, rate, SAMPLE_RATE), dtype=np.float32)


def read_audio(path) -> np.ndarray:
    """A sound file as 16 kHz mono float32 samples."""
    import soundfile as sf

    samples, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return resample(samples.mean(axis=1), rate)


def split(samples: np.ndarray, max_seconds: float = 40.0, search_seconds: float = 5.0) -> list[np.ndarray]:
    """Pieces no longer than max_seconds, each cut at the quietest tenth of a second near its end."""
    limit, search, hop = int(max_seconds * SAMPLE_RATE), int(search_seconds * SAMPLE_RATE), SAMPLE_RATE // 20
    pieces = []
    while len(samples) > limit:
        window = samples[limit - search:limit]
        energy = [float(np.square(window[i:i + 2 * hop]).mean()) for i in range(0, search - 2 * hop + 1, hop)]
        cut = limit - search + int(np.argmin(energy)) * hop + hop
        pieces.append(samples[:cut])
        samples = samples[cut:]
    pieces.append(samples)
    return [piece for piece in pieces if len(piece) >= SAMPLE_RATE // 10]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class _Search:
    """Greedy transducer search, at most max_symbols tokens per frame. Frames can be given a few at a time."""

    def __init__(self, weights: dict, style: str, max_symbols: int) -> None:
        self.w, self.max_symbols = weights, max_symbols
        self.head_w, self.head_b = weights[f"{style}_w"], weights[f"{style}_b"]
        self.blank = self.head_w.shape[0] - 1
        zero = np.zeros(weights["lstm_h"].shape[1], dtype=np.float32)
        self.prediction, self.after = self._step(zero, (zero, zero))  # the very first step sees an empty context
        self.waiting = self._step(weights["start"], (zero, zero))  # from then until the first token: the start vector
        self.ids: list[int] = []
        self.frames = self.last = 0  # frames read, and how many of them had been read when the latest token was written

    def _step(self, vector: np.ndarray, state: tuple) -> tuple:
        """One step of the prediction network (one LSTM layer): its projected output and its new state."""
        w = self.w
        hidden, cell = state
        gates = w["lstm_x"] @ vector + w["lstm_h"] @ hidden + w["lstm_b"]
        i, f, g, o = np.split(gates, 4)
        cell = _sigmoid(f) * cell + _sigmoid(i) * np.tanh(g)
        hidden = _sigmoid(o) * np.tanh(cell)
        return w["pred_w"] @ hidden + w["pred_b"], (hidden, cell)

    def feed(self, encoded: np.ndarray) -> list[int]:
        """encoded [frames, width]: every token so far."""
        w = self.w
        for frame in encoded @ w["enc_w"].T + w["enc_b"]:
            for n in range(self.max_symbols):
                if not self.ids and (self.frames or n):
                    self.prediction, self.after = self.waiting
                token = int(np.argmax(self.head_w @ np.maximum(frame + self.prediction, 0.0) + self.head_b))
                if token == self.blank:
                    break
                self.ids.append(token)
                self.last = self.frames + 1
                self.prediction, self.after = self._step(w["embed"][token], self.after)
            self.frames += 1
        return self.ids


class _LogMel:
    """The log-mel frames of a stream of samples: each frame as soon as its window is complete."""

    def __init__(self, features: dict, weights: dict) -> None:
        self.f, self.window, self.mel = features, weights["window"], weights["mel"]
        self.tail = np.zeros(0, dtype=np.float64)  # samples after the pre-emphasis, from sample `start` on
        self.start = self.total = self.done = 0  # first sample kept, samples heard, frames given out
        self.last = 0.0  # the latest sample, for the pre-emphasis of the next one

    def push(self, samples) -> np.ndarray:
        """Hear more samples: the frames [frames, mel bins] that are complete now."""
        x = np.asarray(samples, dtype=np.float64).reshape(-1)
        if len(x):
            emphasised = x - self.f["preemphasis"] * np.concatenate([[self.last], x[:-1]])  # nothing before the first sample
            self.tail, self.last, self.total = np.concatenate([self.tail, emphasised]), float(x[-1]), self.total + len(x)
        return self._frames(final=False)

    def flush(self) -> np.ndarray:
        """The stream has ended: its last frames, whose windows reach past the end."""
        return self._frames(final=True)

    def _frames(self, final: bool) -> np.ndarray:
        hop, size = self.f["hop_length"], self.f["n_fft"]
        half = size // 2
        last = self.total // hop if final else (self.total - half - 1) // hop  # frame t is centred on sample t * hop
        if self.total <= half or last < self.done:
            return np.zeros((0, self.mel.shape[0]), dtype=np.float64)
        x, base, low = self.tail, self.start, self.done * hop - half
        if low < 0:  # the windows of the first frames reach before the stream: mirrored samples, as in the training front end
            x, base = np.concatenate([x[-low:0:-1], x]), low
        if final:  # and the last windows reach past it
            x = np.concatenate([x, x[-2:-half - 2:-1]])
        windows = np.lib.stride_tricks.sliding_window_view(x[low - base:last * hop + half - base], size)[::hop]
        power = np.abs(np.fft.rfft(windows * self.window, axis=1)) ** 2
        self.done = last + 1
        keep = max(self.done * hop - half, 0)
        self.tail, self.start = self.tail[keep - self.start:], keep
        return np.log(power @ self.mel.T + self.f["log_guard"])


class _RunningNorm:
    """Each frame normalised by the mean and deviation of the frames up to it. The sums start from `prior_frames`
    frames that have the mean and deviation of the training data."""

    def __init__(self, norm: dict) -> None:
        mean, std = np.asarray(norm["mean"], dtype=np.float64), np.asarray(norm["std"], dtype=np.float64)
        self.count = float(norm["prior_frames"])
        self.sum, self.squares = self.count * mean, self.count * (std * std + mean * mean)

    def __call__(self, mel: np.ndarray) -> np.ndarray:
        """mel [frames, mel bins], in the order heard."""
        if not len(mel):
            return np.zeros(mel.shape, dtype=np.float32)
        counts = self.count + np.arange(1, len(mel) + 1)[:, None]
        sums, squares = self.sum + np.cumsum(mel, axis=0), self.squares + np.cumsum(mel * mel, axis=0)
        mean = sums / counts
        deviation = np.sqrt(np.maximum(squares / counts - mean * mean, 1e-10))
        self.count, self.sum, self.squares = float(counts[-1, 0]), sums[-1], squares[-1]
        return ((mel - mean) / (deviation + 1e-5)).astype(np.float32)


class Shravya:
    """The model: an ONNX encoder, and a small prediction and joint network evaluated in numpy."""

    def __init__(self, folder, threads: int = 0, providers: list[str] | None = None) -> None:
        import onnxruntime as ort

        folder = Path(folder)
        self.config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads  # 0 lets onnxruntime use every core
        providers = providers or ["CPUExecutionProvider"]
        self.encoder = ort.InferenceSession(str(folder / self.config["encoder"]), options, providers=providers)
        self.streaming = self.config.get("streaming")
        self.step = None
        if self.streaming and (folder / self.streaming["encoder"]).exists():
            self.step = ort.InferenceSession(str(folder / self.streaming["encoder"]), options, providers=providers)
        with np.load(folder / self.config["decoder"]) as weights:
            self.w = {name: weights[name] for name in weights.files}
        self.tokens: list[str] = self.config["tokens"]
        self.max_symbols = int(self.config["max_symbols"])

    def log_mel(self, samples: np.ndarray) -> np.ndarray:
        """Log-mel frames [frames, mel bins] of a whole recording, not normalised."""
        f, w = self.config["features"], self.w
        x = np.asarray(samples, dtype=np.float64)
        x = np.concatenate([x[:1], x[1:] - f["preemphasis"] * x[:-1]])
        x = np.pad(x, f["n_fft"] // 2, mode="reflect")
        frames = np.lib.stride_tricks.sliding_window_view(x, f["n_fft"])[::f["hop_length"]]
        power = np.abs(np.fft.rfft(frames * w["window"], axis=1)) ** 2
        return np.log(power @ w["mel"].T + f["log_guard"])

    def normalise(self, mel: np.ndarray, like: np.ndarray | None = None) -> np.ndarray:
        """Frames [frames, mel bins] normalised over the utterance, as features [mel bins, frames].

        With `like`, the mean and deviation are those of the frames `like` instead of `mel` itself."""
        like = mel if like is None else like
        mel = (mel - like.mean(axis=0)) / (like.std(axis=0, ddof=1) + self.config["features"]["std_guard"])
        return np.ascontiguousarray(mel.T, dtype=np.float32)

    def features(self, samples: np.ndarray) -> np.ndarray:
        """Log-mel features [mel bins, frames], normalised over the utterance."""
        return self.normalise(self.log_mel(samples))

    def encode_features(self, features: np.ndarray) -> np.ndarray:
        """Encoder output [frames, width] of features [mel bins, frames], with the whole utterance as context."""
        output, length = self.encoder.run(None, {"audio_signal": features[None],
                                                 "length": np.array([features.shape[1]], dtype=np.int64)})
        return output[0].T[:int(length[0])]

    def encode(self, samples: np.ndarray) -> np.ndarray:
        """Encoder output [frames, width], one frame per 40 ms."""
        return self.encode_features(self.features(samples))

    def decode(self, encoded: np.ndarray, style: str = "standard") -> list[int]:
        """Greedy transducer search: at most max_symbols tokens per frame."""
        return _Search(self.w, style, self.max_symbols).feed(encoded)

    def text(self, ids: list[int]) -> str:
        """Token ids as text. Token 0 is the unknown piece and is left out."""
        return "".join(self.tokens[i] for i in ids if i).replace("\u2581", " ").strip()

    def transcribe(self, audio, style: str = "standard", max_seconds: float = 40.0) -> str:
        """Transcribe a sound file, or 16 kHz mono float samples in the range -1 to 1.

        style "standard" writes standard Nepali spelling, the convention of read text. style "verbatim"
        writes words the way they were said, the convention of call and podcast transcripts.
        """
        if style not in STYLES:
            raise ValueError(f"style must be one of {STYLES}, not {style!r}")
        samples = read_audio(audio) if isinstance(audio, (str, Path)) else np.asarray(audio, dtype=np.float32)
        parts = [self.text(self.decode(self.encode(piece), style)) for piece in split(samples, max_seconds)]
        return " ".join(part for part in parts if part)

    def stream(self, style: str = "standard", chunk_ms: int | None = None, second_pass: bool = True) -> "Stream":
        """A phrase heard live. chunk_ms is how much sound is read at a time, and so how far the text lags."""
        if self.step is None:
            raise ValueError("this model folder has no streaming encoder")
        if style not in STYLES:
            raise ValueError(f"style must be one of {STYLES}, not {style!r}")
        return Stream(self, style, chunk_ms or self.streaming["chunk_ms"][0], second_pass)


class Stream:
    """Speech heard live, block by block, one phrase after another.

    `feed` gives the text of the current phrase so far. It is written with the sound heard up to then and the
    last few seconds before it, so it never changes what it has already said. When the speaker pauses, `cut`
    reads the phrase once more as a whole, with every word in view of what follows it; that text is better and
    is the one to keep. The stream goes on listening, and the sound of the phrases before stays in view: up to
    `context_seconds` of it are read again with each later phrase. `finish` is `cut` at the end of the sound;
    after it the stream is empty. A phrase longer than `max_seconds` keeps the text of the first reading.
    """

    def __init__(self, model: Shravya, style: str, chunk_ms: int, second_pass: bool = True, max_seconds: float = 40.0,
                 context_seconds: float = 8.0) -> None:
        setting = model.streaming
        if chunk_ms not in setting["chunk_ms"]:
            raise ValueError(f"chunk_ms must be one of {setting['chunk_ms']}, not {chunk_ms}")
        self.model, self.style, self.second_pass = model, style, second_pass
        self.max_seconds, self.context_seconds = max_seconds, context_seconds
        self.tail_seconds = 0.32  # how much sound after its last piece of text still belongs to a phrase
        self.own_statistics = False  # normalise the second reading by the phrase alone, not by the phrase and its context
        self.keep_search = True  # at a cut the first reading's search goes on; only its text is divided
        self.frames_per_step = chunk_ms // FRAME_MS
        self.lead = int(setting["lead_features"])  # feature frames repeated before a chunk, for the subsampling
        self.shapes = [tuple(i.shape) for i in model.step.get_inputs() if i.name in ("cache_channel", "cache_time")]
        self.reset()

    def reset(self) -> None:
        """Forget everything heard so far."""
        model, bins = self.model, self.model.w["mel"].shape[0]
        self.mel = _LogMel(model.config["features"], model.w)
        self.norm = _RunningNorm(model.streaming["norm"])
        self.search = _Search(model.w, self.style, model.max_symbols)
        self.started = self.mark = 0  # encoder frames read when this search began, and its tokens before the current phrase
        self.heard: list[np.ndarray] = []  # frames from frame `since` on, not normalised, for the second reading
        self.since = 0  # the feature frame at which the current phrase began
        self.before = np.zeros((0, bins), dtype=np.float64)  # the frames just before it: context for that reading
        self.ready = np.zeros((0, bins), dtype=np.float32)  # normalised frames, from frame `first` on
        self.first = self.emitted = 0  # the first feature frame kept, and encoder frames read so far
        self.caches = [np.zeros(shape, dtype=np.float32) for shape in self.shapes] + [np.zeros(1, dtype=np.int64)]

    @property
    def text(self) -> str:
        """The text of the current phrase so far."""
        return self.model.text(self.search.ids[self.mark:])

    @property
    def seconds(self) -> float:
        """How much of the current phrase has been heard."""
        return self.mel.total / SAMPLE_RATE - self.since / 100  # 100 feature frames a second

    @property
    def pause(self) -> float:
        """Seconds of sound read since the last piece of text was written; 0 while the phrase has no text.

        A speaker who has stopped shows as a pause that keeps growing: `if stream.pause > 0.8: stream.cut()`.
        """
        return (self.search.frames - self.search.last) * FRAME_MS / 1000 if len(self.search.ids) > self.mark else 0.0

    def feed(self, samples) -> str:
        """Hear more: 16 kHz mono float samples in the range -1 to 1. Returns the text of the phrase so far."""
        self._read(self.mel.push(samples), final=False)
        return self.text

    def cut(self) -> str:
        """The speaker has paused: the final text of the phrase. The stream keeps listening for the next one.

        The phrase ends `tail_seconds` after its last piece of text, and at the latest where the first reading
        has got to. The rest of the pause opens the next phrase: if the speaker has just started again, that
        beginning is read with what follows it, not cut off. The first reading is not interrupted."""
        mel = np.concatenate(self.heard) if self.heard else self.before[:0]
        read = SUBSAMPLING * self.emitted - self.since
        if len(self.search.ids) > self.mark:
            tail = int(self.tail_seconds * 1000 / FRAME_MS)
            read = min(read, SUBSAMPLING * (self.started + self.search.last + tail) - self.since)
        text = self._final(mel[:read])
        kept = int(self.context_seconds * 100)
        self.before = np.concatenate([self.before, mel[:read]])[-kept:] if kept else self.before[:0]
        self.heard, self.since = [mel[read:]], self.since + read
        if self.keep_search:
            self.mark = len(self.search.ids)
        else:
            self.search, self.started, self.mark = _Search(self.model.w, self.style, self.model.max_symbols), self.emitted, 0
        return text

    def finish(self) -> str:
        """The sound has ended: the final text of the last phrase. The stream is empty afterwards."""
        self._read(self.mel.flush(), final=True)
        text = self._final(np.concatenate(self.heard) if self.heard else self.before[:0])
        self.reset()
        return text

    def _final(self, mel: np.ndarray) -> str:
        """The phrase (its frames, not normalised) read as a whole, in view of the sound before it."""
        model = self.model
        if not self.second_pass or not 2 <= len(mel) <= self.max_seconds * 100:
            return self.text
        lead = len(self.before) // SUBSAMPLING * SUBSAMPLING  # context in whole encoder frames
        both = np.concatenate([self.before[len(self.before) - lead:], mel])
        if model.streaming.get("second_pass", "utterance") == "causal":
            features = np.ascontiguousarray(_RunningNorm(model.streaming["norm"])(both).T)
        else:
            features = model.normalise(both, like=mel if self.own_statistics else None)
        return model.text(model.decode(model.encode_features(features)[lead // SUBSAMPLING:], self.style))

    def _read(self, mel: np.ndarray, final: bool) -> None:
        """Take new frames, and run the encoder over every chunk that is complete (at the end: over what is left)."""
        if len(mel):
            self.heard.append(mel)
            self.ready = np.concatenate([self.ready, self.norm(mel)])
        while True:
            start = SUBSAMPLING * self.emitted  # the feature frame at which the next encoder frame starts
            have = self.first + len(self.ready)
            end = start + SUBSAMPLING * self.frames_per_step
            if have < end:
                if not final or have <= start:
                    return
                end = have
            begin = max(start - self.lead, 0)
            piece = np.ascontiguousarray(self.ready[begin - self.first:end - self.first].T[None])
            frames, *self.caches = self.model.step.run(None, {
                "features": piece, "drop": np.array((start - begin) // SUBSAMPLING, dtype=np.int64),
                "cache_channel": self.caches[0], "cache_time": self.caches[1], "cache_len": self.caches[2]})
            self.search.feed(frames[0].T)
            self.emitted += frames.shape[-1]
            keep = max(end - self.lead, self.first)
            self.ready, self.first = self.ready[keep - self.first:], keep


def load(folder=None, threads: int = 0, providers: list[str] | None = None) -> Shravya:
    """Load the model from a folder. Without a folder: the files beside this script, else the Hugging Face Hub."""
    if folder is None:
        folder = Path(__file__).resolve().parent
        if not (folder / "config.json").exists():
            from huggingface_hub import snapshot_download

            folder = snapshot_download(REPOSITORY)
    return Shravya(folder, threads=threads, providers=providers)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="shravya", description="Transcribe Nepali speech with Shravya.")
    parser.add_argument("audio", nargs="+", help="sound files (wav, flac, ogg, mp3)")
    parser.add_argument("--model", help="folder with the model files (default: beside this script)")
    parser.add_argument("--style", choices=STYLES, default="standard")
    parser.add_argument("--threads", type=int, default=0, help="processor threads (default: all)")
    parser.add_argument("--stream", action="store_true", help="read each file the way live speech is heard, and show the text as it grows")
    parser.add_argument("--chunk-ms", type=int, help="with --stream: how much sound is read at a time")
    parser.add_argument("--pause", type=float, default=0.8, help="with --stream: seconds without text that end a phrase")
    args = parser.parse_args(argv)
    model = load(args.model, threads=args.threads)
    for path in args.audio:
        if args.stream:
            stream, samples, shown, finals = model.stream(args.style, args.chunk_ms), read_audio(path), "", []
            for start in range(0, len(samples), SAMPLE_RATE // 10):
                text = stream.feed(samples[start:start + SAMPLE_RATE // 10])
                if text != shown:
                    print(f"{start / SAMPLE_RATE:7.1f} s  {text}")
                    shown = text
                if stream.pause >= args.pause:
                    finals.append(stream.cut())
                    print(f"{start / SAMPLE_RATE:7.1f} s  = {finals[-1]}")
                    shown = ""
            text = " ".join(final for final in finals + [stream.finish()] if final)
        else:
            text = model.transcribe(path, style=args.style)
        print(text if len(args.audio) == 1 else f"{path}\t{text}")


if __name__ == "__main__":
    main()
