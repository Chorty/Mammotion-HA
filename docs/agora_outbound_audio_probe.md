# Agora outbound-audio probe

This probe determines whether a mower consumes an audio track published into
its FPV Agora channel. It is an experiment, not a Home Assistant TTS service.

## Safety

- Keep the mower stationary and within hearing distance.
- Start with the built-in one-second tone, which is capped at 5% Web Audio gain.
- Stop immediately if the mower produces unexpected behavior or excessive volume.
- Do not expose the page or stream credentials publicly.

## Run the probe

### Automated local run

If this repository has access to your Home Assistant URL and long-lived token,
run:

```bash
python3 scripts/mammotion_agora_audio_probe.py camera.your_mower_camera
```

The script reads `HA_URL` and `HA_TOKEN` from the environment, or from a local
`.env` file if present. It calls `mammotion.refresh_stream`, calls
`mammotion.get_tokens`, and opens `agora_test.html` with the live Agora
credentials in the URL fragment. The page then connects in SDK mode and
publishes a one-second quiet tone automatically.

Options:

```bash
python3 scripts/mammotion_agora_audio_probe.py camera.your_mower_camera \
  --frequency 880 \
  --duration 1 \
  --print-only
```

The generated URL contains a live stream token. Treat it as sensitive and do not
paste it into logs or chats.

### Manual run

1. In Home Assistant Developer Tools, call `mammotion.refresh_stream` for the
   mower's camera entity.
2. Call `mammotion.get_tokens` for the same entity and request the response.
3. Open `agora_test.html` from this repository in a browser.
4. Copy `appid`, `channelName`, `token`, and `uid` from the service response into
   the connection fields.
5. Select **Connect (SDK)**. Raw WebSocket mode cannot publish media.
6. Once connected and the mower appears as a remote user, select
   **Publish Quiet Tone**.
7. Confirm physically whether the mower speaker plays the tone. The log entry
   `Outbound Opus track published to Agora` only proves the SFU accepted the
   local track.

To test speech after the tone succeeds, generate a short WAV, MP3, or Opus file,
select it under **Audio file**, and select **Publish Audio File**. File playback
uses the same 5% gain cap.

## Interpret the result

- **Tone plays:** the Agora uplink is viable. The next implementation step is a
  Home Assistant TTS service backed by a server-side WebRTC publisher.
- **Publish succeeds but no tone plays:** inspect whether the official app uses
  a separate intercom command or whether the mower subscribes only to a specific
  Agora UID/stream type.
- **Publish is rejected:** capture the SDK error and verify that the stream token
  grants publisher privileges. A subscriber-only token cannot test speaker
  capability.
