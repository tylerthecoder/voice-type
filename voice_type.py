#!/usr/bin/env python3
"""Voice-to-text transcription using OpenAI Whisper API."""

import argparse
import io
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from openai import OpenAI

PID_FILE = Path(tempfile.gettempdir()) / "voice-type.pid"
AUDIO_FILE = Path(tempfile.gettempdir()) / "voice-type.wav"
LOG_FILE = Path.home() / ".voice-type.log"


def log(msg: str) -> None:
    """Log message to file."""
    with open(LOG_FILE, "a") as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\n")


def get_api_key() -> str:
    """Get OpenAI API key from environment or config files."""
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key

    # Try owl secrets
    secrets_file = Path.home() / "owl" / "setups" / "secrets" / "secrets.sh"
    if secrets_file.exists():
        for line in secrets_file.read_text().splitlines():
            if "OPENAI_API_KEY=" in line:
                # Handle export OPENAI_API_KEY="value" format
                part = line.split("OPENAI_API_KEY=", 1)[1].strip()
                return part.strip('"').strip("'")

    # Try shell_gpt config
    sgpt_config = Path.home() / ".config" / "shell_gpt" / ".sgptrc"
    if sgpt_config.exists():
        for line in sgpt_config.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip()

    raise ValueError("OPENAI_API_KEY not found")


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


def record_until_signal(sample_rate: int = 16000) -> np.ndarray:
    """Record audio until SIGUSR1 is received."""
    recording = []
    stop_flag = threading.Event()

    def handle_signal(signum, frame):
        stop_flag.set()

    signal.signal(signal.SIGUSR1, handle_signal)

    def callback(indata, frames, time_info, status):
        if status:
            print(f"Status: {status}", file=sys.stderr)
        recording.append(indata.copy())

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype=np.float32, callback=callback):
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
    log(f"type_text called with: {text[:50]}")
    if not text:
        return

    # Small delay to let focus settle
    time.sleep(0.1)

    # Try wtype first
    try:
        log("running wtype")
        result = subprocess.run(["wtype", "-d", "10", "--", text], capture_output=True, text=True)
        log(f"wtype returned {result.returncode}, stderr={result.stderr}")
        if result.returncode == 0:
            return
        # wtype failed, fall back to clipboard
        notify(f"wtype failed, copied to clipboard", urgency="low")
    except FileNotFoundError:
        log("wtype not found")
        notify("wtype not found, copied to clipboard", urgency="low")

    # Fallback: copy to clipboard
    try:
        log("falling back to wl-copy")
        subprocess.run(["wl-copy", "--", text], check=True)
    except Exception as e:
        log(f"wl-copy failed: {e}")
        notify(f"Failed to copy: {e}", urgency="critical")


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


def cmd_toggle(args: argparse.Namespace) -> None:
    """Toggle recording - press once to start, again to stop and transcribe."""
    log("toggle called")

    # Check if already recording
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            log(f"found existing PID {pid}, sending SIGUSR1")
            # Send signal to stop recording
            os.kill(pid, signal.SIGUSR1)
            notify("Stopping...", urgency="low")
            return
        except (ProcessLookupError, ValueError) as e:
            log(f"PID file exists but process dead: {e}")
            # Process died, clean up
            PID_FILE.unlink(missing_ok=True)

    # Start recording
    log(f"starting recording, PID={os.getpid()}")
    PID_FILE.write_text(str(os.getpid()))
    try:
        notify("Recording... (press again to stop)", urgency="low")
        audio = record_until_signal()
        log(f"recording stopped, got {len(audio)} samples")

        if len(audio) < 1600:  # Less than 0.1s at 16kHz
            log("recording too short")
            notify("Recording too short", urgency="low")
            return

        notify("Transcribing...", urgency="low")
        text = transcribe(audio)
        log(f"transcribed: {text[:100] if text else '(empty)'}")

        if text:
            log("calling type_text")
            type_text(text)
            log("type_text returned")
            notify(f"{text[:50]}", urgency="low")
        else:
            notify("No speech detected", urgency="low")
    except Exception as e:
        log(f"ERROR: {e}")
        raise
    finally:
        PID_FILE.unlink(missing_ok=True)
        log("toggle finished")


def cmd_quick(args: argparse.Namespace) -> None:
    """Quick record with fixed duration - designed for simple keybindings."""
    duration = args.duration if args.duration else 5.0  # Default 5 seconds
    notify(f"Recording for {duration}s...", urgency="low")
    audio = record_audio(duration=duration)

    if len(audio) < 1600:  # Less than 0.1s at 16kHz
        notify("Recording too short", urgency="low")
        return

    notify("Transcribing...", urgency="low")
    text = transcribe(audio)

    if text:
        type_text(text)
        notify(f"{text[:50]}", urgency="low")
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

    # toggle command - press once to start, again to stop (best for keybindings)
    toggle_parser = subparsers.add_parser("toggle", help="Toggle: press to start, press again to stop")
    toggle_parser.set_defaults(func=cmd_toggle)

    # quick command - fixed duration recording
    quick_parser = subparsers.add_parser("quick", help="Record for fixed duration (default 5s)")
    quick_parser.add_argument("-d", "--duration", type=float, help="Recording duration in seconds (default: 5)")
    quick_parser.set_defaults(func=cmd_quick)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
