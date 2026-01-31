#!/usr/bin/env python3
"""Voice-to-text transcription using OpenAI Whisper API."""

import argparse
import io
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from openai import OpenAI


def get_api_key() -> str:
    """Get OpenAI API key from environment or shell_gpt config."""
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key

    # Try shell_gpt config
    sgpt_config = Path.home() / ".config" / "shell_gpt" / ".sgptrc"
    if sgpt_config.exists():
        for line in sgpt_config.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip()

    raise ValueError("OPENAI_API_KEY not found in environment or ~/.config/shell_gpt/.sgptrc")


def record_audio(duration: float | None = None, sample_rate: int = 16000) -> np.ndarray:
    """Record audio from microphone.

    If duration is None, records until Enter is pressed.
    """
    print("Recording... (press Enter to stop)" if duration is None else f"Recording for {duration}s...")

    recording = []
    stop_flag = threading.Event()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"Status: {status}", file=sys.stderr)
        recording.append(indata.copy())

    def wait_for_enter():
        input()
        stop_flag.set()

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype=np.float32, callback=callback):
        if duration is None:
            thread = threading.Thread(target=wait_for_enter, daemon=True)
            thread.start()
            while not stop_flag.is_set():
                time.sleep(0.1)
        else:
            time.sleep(duration)

    print("Recording stopped.")
    return np.concatenate(recording, axis=0)


def record_while_key_held(sample_rate: int = 16000) -> np.ndarray:
    """Record audio while waiting for stdin signal (for hold-to-record)."""
    recording = []
    stop_flag = threading.Event()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"Status: {status}", file=sys.stderr)
        recording.append(indata.copy())

    def wait_for_signal():
        # Read a single byte from stdin to signal stop
        sys.stdin.read(1)
        stop_flag.set()

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype=np.float32, callback=callback):
        thread = threading.Thread(target=wait_for_signal, daemon=True)
        thread.start()
        while not stop_flag.is_set():
            time.sleep(0.05)

    if not recording:
        return np.array([], dtype=np.float32)
    return np.concatenate(recording, axis=0)


def transcribe(audio_data: np.ndarray, sample_rate: int = 16000) -> str:
    """Transcribe audio using OpenAI Whisper API."""
    if len(audio_data) == 0:
        return ""

    # Convert to WAV in memory
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format="WAV")
    buffer.seek(0)
    buffer.name = "audio.wav"

    client = OpenAI(api_key=get_api_key())

    print("Transcribing...")
    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=buffer,
        response_format="text"
    )

    return response.strip()


def type_text(text: str) -> None:
    """Type text at cursor position using wtype (Wayland)."""
    if not text:
        return

    try:
        subprocess.run(["wtype", "--", text], check=True)
    except FileNotFoundError:
        print("Error: wtype not found. Install it with: sudo pacman -S wtype", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error typing text: {e}", file=sys.stderr)
        sys.exit(1)


def notify(message: str, urgency: str = "normal") -> None:
    """Send desktop notification."""
    try:
        subprocess.run(["notify-send", "-u", urgency, "Voice Type", message], check=False)
    except FileNotFoundError:
        pass  # Notifications not available


def cmd_record(args: argparse.Namespace) -> None:
    """Record and transcribe, print result."""
    audio = record_audio(duration=args.duration)
    text = transcribe(audio)
    print(f"\nTranscription: {text}")


def cmd_type(args: argparse.Namespace) -> None:
    """Record, transcribe, and type at cursor."""
    notify("Recording...")
    audio = record_audio(duration=args.duration)
    notify("Transcribing...")
    text = transcribe(audio)
    if text:
        type_text(text)
        notify(f"Typed: {text[:50]}...")
    else:
        notify("No speech detected", urgency="low")


def cmd_hold(args: argparse.Namespace) -> None:
    """Hold-to-record mode for keybinding integration."""
    # This mode expects a signal on stdin when key is released
    audio = record_while_key_held()
    text = transcribe(audio)
    if text:
        type_text(text)


def cmd_quick(args: argparse.Namespace) -> None:
    """Quick record with visual feedback - designed for keybindings."""
    notify("🎤 Recording...", urgency="low")
    audio = record_audio(duration=args.duration if args.duration else None)

    if len(audio) < 1600:  # Less than 0.1s at 16kHz
        notify("Recording too short", urgency="low")
        return

    notify("⏳ Transcribing...", urgency="low")
    text = transcribe(audio)

    if text:
        type_text(text)
        notify(f"✓ {text[:50]}", urgency="low")
    else:
        notify("No speech detected", urgency="low")


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice-to-text transcription")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # record command - just transcribe and print
    record_parser = subparsers.add_parser("record", help="Record and transcribe (print result)")
    record_parser.add_argument("-d", "--duration", type=float, help="Recording duration in seconds")
    record_parser.set_defaults(func=cmd_record)

    # type command - transcribe and type at cursor
    type_parser = subparsers.add_parser("type", help="Record, transcribe, and type at cursor")
    type_parser.add_argument("-d", "--duration", type=float, help="Recording duration in seconds")
    type_parser.set_defaults(func=cmd_type)

    # quick command - optimized for keybindings
    quick_parser = subparsers.add_parser("quick", help="Quick mode for keybindings")
    quick_parser.add_argument("-d", "--duration", type=float, help="Recording duration in seconds")
    quick_parser.set_defaults(func=cmd_quick)

    # hold command - hold-to-record for advanced keybinding
    hold_parser = subparsers.add_parser("hold", help="Hold-to-record mode")
    hold_parser.set_defaults(func=cmd_hold)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
