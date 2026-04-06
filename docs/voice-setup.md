# Voice Assistant Setup

Wake word → faster-whisper STT → Sovereign Edge orchestrator → Piper TTS.

## Requirements

- **Hardware**: USB mic or ALSA-compatible audio device (Jetson Orin built-in mic works)
- **Python env**: `mlenv` with voice extras installed

## 1. Install dependencies

```bash
conda activate mlenv
pip install openwakeword sounddevice faster-whisper piper-tts scipy
```

## 2. Identify your microphone

```bash
arecord -l
```

Note the card and device numbers. Example output:
```
card 1: USB [USB Audio Device], device 0: USB Audio [USB Audio]
```

Set the default device if needed:
```bash
export ALSA_CARD=1  # match your card number
```

## 3. Test the pipeline

```bash
# Check all deps are found
cd ~/sovereign-edge
python -m voice --check-deps

# Run without wake word (always listening) — good for first test
python -m voice --no-wake

# Full wake word mode (say "hey sovereign" to activate)
python -m voice
```

## 4. systemd service (Jetson Orin production)

```bash
# Install and configure
sudo cp systemd/voice.service /etc/systemd/system/
sudo sed -i \
    's|<DEPLOY_USER>|omnipotence|g; s|<DEPLOY_ROOT>|/home/omnipotence/sovereign-edge|g' \
    /etc/systemd/system/voice.service

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable --now voice.service

# Verify
sudo systemctl status voice.service
journalctl -u sovereign-edge-voice -f
```

The service requires `/dev/snd` to exist — it will not start on VPS instances without audio hardware.

## 5. Docker (voice profile)

```bash
# Requires /dev/snd on the host
docker compose --profile voice up -d voice
docker compose logs -f voice
```

## Options

| Flag | Description |
|------|-------------|
| `--no-wake` | Skip wake word, always listen (dev mode) |
| `--whisper small.en` | Higher accuracy STT (uses more VRAM) |
| `--tts-voice en_US-ryan-high` | Different Piper voice |
| `--thread-id my_session` | Named LangGraph conversation thread |
| `--check-deps` | Verify all deps installed, then exit |

## Troubleshooting

**No audio devices found**: Run `python -c "import sounddevice; print(sounddevice.query_devices())"` to list available devices.

**Wake word not triggering**: Say "hey sovereign" clearly. Reduce ambient noise or use `--no-wake` to bypass.

**STT returns empty**: Increase mic gain with `alsamixer`, or try `--whisper small.en` for better accuracy.
