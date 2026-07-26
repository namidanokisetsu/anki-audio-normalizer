# Audio Normalizer - Consistent Card Loudness

Audio from different sources often plays at inconsistent volumes. Audio
Normalizer measures card audio and creates verified, volume-adjusted copies so
you do not need to change the volume while reviewing.

## Highlights

- Normalize a deck, an Anki search, the current Browser results, or selected notes
- Preview a representative sample before changing anything
- MP3 output or source-container preservation for supported formats
- Optional automatic runs after startup, sync, or note edits
- Cancellable background processing with a detailed report
- Restore eligible note references to retained originals

Processing is local. Card text and audio are never uploaded. The add-on does
not download or install executables.

## Important safety information

The add-on creates new media files and relinks matching notes. Original files
are never overwritten or deleted. Before a broad first run, export an all-decks
`.colpkg` with **Include media**. Do not use **Check Media -> Delete Unused**
until you have verified the result, because retained originals may appear
unused.

Automatic normalization is disabled by default. Generated files use additional
local and AnkiWeb media storage.

## Requirements

- Anki 25.09.4 or newer (tested through 26.05)
- Windows x64, macOS Intel/Apple silicon, or Linux x64/ARM64
- A current FFmpeg installation containing `loudnorm`, `libmp3lame`, `libopus`,
  `aac`, `flac`, and `pcm_s16le`

Installation and FFmpeg instructions, source code, privacy details, and issue
reporting:

https://github.com/namidanokisetsu/anki-audio-normalizer

Version 0.9.0 beta. MIT licensed.
