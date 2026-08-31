import os
import re
import subprocess
import tempfile
import logging
from typing import Optional

import whisper

from config import settings

log = logging.getLogger("audio_service")

_model: "whisper.Whisper | None" = None


def _get_model() -> "whisper.Whisper":
    global _model

    if _model is None:
        log.info(f"Loading Whisper model ({settings.WHISPER_MODEL_NAME})...")
        _model = whisper.load_model(settings.WHISPER_MODEL_NAME)

    return _model


def _run_ffmpeg(cmd: list) -> subprocess.CompletedProcess:
    """Run ffmpeg, preferring the configured path and falling back to PATH."""
    cmd[0] = settings.FFMPEG_PATH
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except Exception:
        cmd[0] = "ffmpeg"
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60)


def _get_audio_rms_db(path: str) -> Optional[float]:
    try:
        result = _run_ffmpeg([
            "ffmpeg",
            "-i", path,
            "-af", "volumedetect",
            "-f", "null",
            os.devnull,
        ])

        output = (result.stderr or "") + "\n" + (result.stdout or "")

        match = re.search(
            r"mean_volume:\s*(-?\d+(\.\d+)?)\s*dB",
            output,
        )

        if match:
            return float(match.group(1))

        log.warning("Couldn't detect audio volume")
        return None

    except Exception as e:
        log.warning(f"volumedetect error: {e}")
        return None


def _convert_to_wav(input_path: str) -> str:
    fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        "-vn",
        out_path,
    ]

    result = _run_ffmpeg(cmd)

    if result.returncode != 0:
        if os.path.exists(out_path):
            os.unlink(out_path)

        raise RuntimeError(f"FFmpeg conversion failed:\n{result.stderr}")

    if os.path.getsize(out_path) < 2000:
        os.unlink(out_path)
        raise RuntimeError("Audio file too small — likely empty recording.")

    return out_path


def _detect_language(converted_path: str) -> str:
    try:
        model = _get_model()

        audio = whisper.load_audio(converted_path)
        audio = whisper.pad_or_trim(audio)

        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        _, probs = model.detect_language(mel)

        ar = probs.get("ar", 0)
        en = probs.get("en", 0)

        log.debug(f"Language probs — ar={ar:.2f}, en={en:.2f}")

        if ar >= en * 0.75:
            return "ar"

        return "en"

    except Exception as e:
        log.warning(f"Language detection error: {e}")
        return "ar"


def _initial_prompt(lang: str) -> str:
    if lang == "ar":
        return (
            "هذا تسجيل صوتي باللغة العربية، غالباً باللهجة المصرية. "
            "اكتب الكلام كما قيل بدون ترجمة. "
            "صحّح الكلمات الشائعة إذا كانت واضحة من السياق. "
            "مثال: إذا سمعت ما يشبه 'كل تلك قدم' والمقصود واضح، فاكتب 'كرة القدم'. "
            "حافظ على المصطلحات الإنجليزية كما هي."
        )

    return (
        "Transcribe the speech exactly. "
        "Do not translate. "
        "Keep technical English terms as they are."
    )


def transcribe_audio(audio_bytes: bytes, language: Optional[str] = None) -> str:
    fd, tmp_path = tempfile.mkstemp(suffix=".tmp")
    os.close(fd)

    converted_path = None

    try:
        log.debug(f"Received audio size: {len(audio_bytes)} bytes")

        if len(audio_bytes) < 5000:
            raise RuntimeError("Audio too small — microphone may not be working.")

        with open(tmp_path, "wb") as f:
            f.write(audio_bytes)

        converted_path = _convert_to_wav(tmp_path)

        mean_db = _get_audio_rms_db(converted_path)
        log.debug(f"Mean volume: {mean_db} dB")

        if mean_db is not None and mean_db < settings.SILENCE_THRESHOLD_DB:
            raise RuntimeError(
                f"Audio is silent (mean volume = {mean_db:.1f} dB). "
                "Check microphone permissions."
            )

        model = _get_model()

        forced_lang = language if language in {"ar", "en"} else None

        # Whisper's own default is a temperature *fallback schedule*, not a
        # single value: it retries decoding at progressively higher
        # temperatures whenever a decode fails its internal quality gates
        # (compression_ratio_threshold / logprob_threshold) — very common on
        # real microphone audio. A bare 0.0 disables that retry entirely
        # (one failed attempt = final result) and silently turns best_of
        # into a no-op, since best_of only applies when temperature > 0.
        base_kwargs = {
            "fp16": False,
            "task": "transcribe",
            "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            "condition_on_previous_text": False,
            "beam_size": 5,
            "best_of": 5,
        }

        if forced_lang:
            # Manual language selection: force the selected language.
            resolved_lang = forced_lang
        else:
            # Auto mode: decide the language ONCE with a cheap encoder-only
            # language-ID pass (_detect_language — a single forward pass,
            # no decoding) instead of trusting whatever Whisper's own
            # unprompted transcribe pass happens to guess. The previous
            # implementation only gave Arabic its dialect-tuned prompt when
            # an unprompted first pass already said "ar", so a misdetected
            # first guess silently skipped Arabic handling entirely. This
            # also avoids ever running two full transcribe passes: exactly
            # one full pass happens below, in every mode.
            resolved_lang = _detect_language(converted_path)

        result = model.transcribe(
            converted_path,
            language=resolved_lang,
            initial_prompt=_initial_prompt(resolved_lang),
            **base_kwargs,
        )

        text = (result.get("text") or "").strip()
        detected_lang = result.get("language", resolved_lang)

        if not text:
            raise RuntimeError("No speech detected in audio.")

        log.info(f"Transcription success [{detected_lang}]: {text[:80]}...")

        return text

    finally:
        for p in [tmp_path, converted_path]:
            if p and os.path.isfile(p):
                try:
                    os.unlink(p)
                except Exception:
                    pass


def transcribe_audio_path(file_path: str, language: Optional[str] = None) -> str:
    with open(file_path, "rb") as f:
        return transcribe_audio(f.read(), language=language)
