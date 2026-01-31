# voice-type

Press a key, speak, and have your words typed at your cursor. Uses OpenAI Whisper for fast, accurate transcription.

## Installation

```bash
# Clone the repo
git clone https://github.com/tylerthecoder/voice-type.git
cd voice-type

# Install Python dependencies
pip install -r requirements.txt

# Install wtype for Wayland typing
sudo pacman -S wtype  # Arch
# or: sudo apt install wtype  # Debian/Ubuntu
```

## Configuration

Set your OpenAI API key in one of these ways:

1. Environment variable: `export OPENAI_API_KEY=your-key`
2. The app also reads from `~/.config/shell_gpt/.sgptrc` if you use shell_gpt

## Usage

### CLI Commands

```bash
# Record until Enter, print transcription
voice-type record

# Record for 5 seconds, print transcription
voice-type record -d 5

# Record until Enter, type result at cursor
voice-type type

# Quick mode with notifications (best for keybindings)
voice-type quick
```

### Sway/i3 Keybinding

Add to your sway config:

```
bindsym $mod+v exec voice-type quick
```

## How it works

1. Records audio from your microphone
2. Sends to OpenAI Whisper API for transcription
3. Types the result at your cursor using `wtype` (Wayland)
