"""Short, amplitude-limited Windows alert; does not change system volume."""
import math
import struct
import wave
from common import ROOT


class SoundAlerts:
    def __init__(self, volume=.15):
        import winsound
        self.backend = winsound
        self.path = ROOT / 'results' / 'alert.wav'
        self.path.parent.mkdir(exist_ok=True)
        rate = 22050
        samples = bytearray()
        # Two 250 ms tones with fades and a pause, avoiding abrupt clicks.
        for frequency in (660, 880):
            for i in range(int(rate * .25)):
                t = i / rate
                fade = max(0, min(1, t/.025, (.25-t)/.025))
                value = int(32767 * volume * fade * math.sin(2*math.pi*frequency*t))
                samples.extend(struct.pack('<h', value))
            samples.extend(b'\x00\x00' * int(rate*.15))
        with wave.open(str(self.path), 'wb') as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(rate)
            output.writeframes(samples)

    def play(self):
        try:
            self.backend.PlaySound(str(self.path), self.backend.SND_FILENAME | self.backend.SND_ASYNC | self.backend.SND_NODEFAULT)
        except RuntimeError:
            print('Alert sound could not play. Check your audio output.')
