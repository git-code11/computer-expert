## Audio Data information
Audio bit rate is the amount of digital audio data processed per unit of time, typically measured in bits per second (bps) or kilobits per second (kbps).
It directly relates to audio quality and file size: a higher bit rate means better quality but a larger file, while a lower bit rate means more compression artifacts but a smaller file.
Components of Audio Bit Rate The bit rate for uncompressed digital audio (PCM or WAV formats) is determined by three main components, which define how the original analog sound wave is converted into digital data:
Sample Rate:
The number of "snapshots" (samples) of the analog sound wave taken per second. Measured in Hertz (Hz) or kilohertz (kHz), it determines the frequency range of the recorded sound.
The standard for CD quality is 44.1 kHz, meaning 44,100 samples are taken every second.Bit Depth: The number of bits used to represent the amplitude (volume) of each sample.
A higher bit depth allows for a greater dynamic range (the difference between the loudest and quietest sounds) and more precision, with 16-bit being standard for CDs and 24-bit used in professional studios for higher resolution.
Number of Channels: The number of independent audio signals being recorded or played back, typically mono (1 channel) or stereo (2 channels).

The size of an audio frame is calculated by multiplying the sample size in bytes by the number of channels, so a single frame of stereo 16-bit audio is 4 bytes long and a single frame of 5.1 floating-point audio is 24 (4 bytes per sample multiplied by 6 channels).

An audio block size (or buffer size) is the number of audio samples processed at one time by a digital audio workstation (DAW) or audio interface

Calculation of Audio Bit Rate For uncompressed audio, the constant bit rate (CBR) can be calculated using a simple formula:

```
Bit Rate (bps) = Sample Rate (Hz) × Bit Depth (bits) × Number of Channels
File Size = Bit Rate * Duration / 8
Sample Size (Frame Size in bits) = Sample Rate (Hz) × Bit Depth (bits)
No. of frame (Block Size or sample count) = Frame Rate (same as sample rate) * duration (seconds)
```
Example:
For a standard CD-quality stereo recording:
Sample Rate: 44.1 kHz (44,100 Hz)
Bit Depth: 16 bits
Channels: 2 (stereo)
Calculation:
44,100 Hz * 16 bits * 2 channels =1,411,200 bps
This is typically expressed as 1,411.2 kbps or approximately 1.4 Mbps.
For compressed audio formats like MP3 or AAC, the bit rate is variable (VBR) or constant (CBR) but the value itself is lower because data is discarded using psychoacoustic modeling to reduce file size while minimizing perceived quality loss.
For these formats, the simple formula above does not apply.
Common compressed bit rates range from 128 kbps (acceptable for casual listening) to 320 kbps (high quality for MP3s)

https://en.wikipedia.org/wiki/Bit_rate
https://en.wikipedia.org/wiki/Decibel
https://www.animations.physics.unsw.edu.au/jw/dB.htm
