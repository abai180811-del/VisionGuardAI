
from array import array
import sys
import wave
from common import ROOT


class SoundAlerts:
    def __init__(self, volume=.35):
        import winsound
        self.backend = winsound
        self.path = ROOT / 'results' / 'alert.wav'
        self.path.parent.mkdir(exist_ok=True)
        with wave.open(str(ROOT / 'assets' / 'alarm.wav'), 'rb') as source:
            rate = source.getframerate()
            channels = source.getnchannels()
            if source.getsampwidth() != 2:
                raise ValueError('Alarm audio must be a 16-bit PCM WAV file.')
            samples = array('h', source.readframes(source.getnframes()))
        if sys.byteorder != 'little':
            samples.byteswap()
        volume = max(0, min(.5, volume))
        samples = array('h', (round(sample * volume) for sample in samples))
        if sys.byteorder != 'little':
            samples.byteswap()
        
        signal = samples.tobytes()
        gap = b'\x00\x00' * channels * int(rate * .4)
        samples = gap.join([signal] * 3)
        with wave.open(str(self.path), 'wb') as output:
            output.setnchannels(channels)
            output.setsampwidth(2)
            output.setframerate(rate)
            output.writeframes(samples)

    def play(self):
        try:
            self.backend.PlaySound(str(self.path), self.backend.SND_FILENAME | self.backend.SND_ASYNC | self.backend.SND_NODEFAULT)
        except RuntimeError:
            print('Alert sound could not play. Check your audio output.')
