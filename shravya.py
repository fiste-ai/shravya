"""Shravya (श्रव्य): Nepali speech recognition by Fiste AI. Standalone runtime.

It needs four small packages and nothing else:

    pip install onnxruntime numpy soundfile soxr

    import shravya
    model = shravya.load("path/to/model")                     # the folder with encoder.onnx, decoder.npz, config.json
    print(model.transcribe("clip.wav"))                       # standard written Nepali
    print(model.transcribe("call.wav", style="verbatim"))     # written the way it was said

From a shell:

    python shravya.py clip.wav another.flac --style verbatim

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

__version__ = "0.1"
SAMPLE_RATE = 16000
STYLES = ("standard", "verbatim")
REPOSITORY = "fiste-ai/shravya"


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


class Shravya:
    """The model: an ONNX encoder, and a small prediction and joint network evaluated in numpy."""

    def __init__(self, folder, threads: int = 0, providers: list[str] | None = None) -> None:
        import onnxruntime as ort

        folder = Path(folder)
        self.config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads  # 0 lets onnxruntime use every core
        self.encoder = ort.InferenceSession(str(folder / self.config["encoder"]), options,
                                            providers=providers or ["CPUExecutionProvider"])
        with np.load(folder / self.config["decoder"]) as weights:
            self.w = {name: weights[name] for name in weights.files}
        self.tokens: list[str] = self.config["tokens"]
        self.max_symbols = int(self.config["max_symbols"])

    def features(self, samples: np.ndarray) -> np.ndarray:
        """Log-mel features [mel bins, frames], normalised over the utterance."""
        f, w = self.config["features"], self.w
        x = np.asarray(samples, dtype=np.float64)
        x = np.concatenate([x[:1], x[1:] - f["preemphasis"] * x[:-1]])
        x = np.pad(x, f["n_fft"] // 2, mode="reflect")
        frames = np.lib.stride_tricks.sliding_window_view(x, f["n_fft"])[::f["hop_length"]]
        power = np.abs(np.fft.rfft(frames * w["window"], axis=1)) ** 2
        mel = np.log(power @ w["mel"].T + f["log_guard"])
        mel = (mel - mel.mean(axis=0)) / (mel.std(axis=0, ddof=1) + f["std_guard"])
        return np.ascontiguousarray(mel.T, dtype=np.float32)

    def encode(self, samples: np.ndarray) -> np.ndarray:
        """Encoder output [frames, width], one frame per 40 ms."""
        features = self.features(samples)
        output, length = self.encoder.run(None, {"audio_signal": features[None],
                                                 "length": np.array([features.shape[1]], dtype=np.int64)})
        return output[0].T[:int(length[0])]

    def _step(self, vector: np.ndarray, state: tuple) -> tuple:
        """One step of the prediction network (one LSTM layer): its projected output and its new state."""
        w = self.w
        hidden, cell = state
        gates = w["lstm_x"] @ vector + w["lstm_h"] @ hidden + w["lstm_b"]
        i, f, g, o = np.split(gates, 4)
        cell = _sigmoid(f) * cell + _sigmoid(i) * np.tanh(g)
        hidden = _sigmoid(o) * np.tanh(cell)
        return w["pred_w"] @ hidden + w["pred_b"], (hidden, cell)

    def decode(self, encoded: np.ndarray, style: str = "standard") -> list[int]:
        """Greedy transducer search: at most max_symbols tokens per frame."""
        w = self.w
        head_w, head_b = w[f"{style}_w"], w[f"{style}_b"]
        blank = head_w.shape[0] - 1
        frames = encoded @ w["enc_w"].T + w["enc_b"]
        zero = np.zeros(w["lstm_h"].shape[1], dtype=np.float32)
        prediction, after = self._step(zero, (zero, zero))  # the very first step sees an empty context
        waiting = self._step(w["start"], (zero, zero))  # from then until the first token: the start vector
        ids: list[int] = []
        for t, frame in enumerate(frames):
            for n in range(self.max_symbols):
                if not ids and (t or n):
                    prediction, after = waiting
                token = int(np.argmax(head_w @ np.maximum(frame + prediction, 0.0) + head_b))
                if token == blank:
                    break
                ids.append(token)
                prediction, after = self._step(w["embed"][token], after)
        return ids

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
    args = parser.parse_args(argv)
    model = load(args.model, threads=args.threads)
    for path in args.audio:
        text = model.transcribe(path, style=args.style)
        print(text if len(args.audio) == 1 else f"{path}\t{text}")


if __name__ == "__main__":
    main()
