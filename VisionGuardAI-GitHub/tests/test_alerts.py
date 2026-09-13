
import sys
import unittest
import wave
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch, MagicMock
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from proximity import ProximityMonitor
from sound_alerts import SoundAlerts
from telegram_alerts import send_message, TelegramAlerts

NEAR = [[80, 30, 110, 90, .9, 0], [100, 10, 200, 110, .9, 1]]


class AlertTests(unittest.TestCase):
    def test_hold_and_cooldown(self):
        m = ProximityMonitor(hold=1, confirmations=2, cooldown=60)
        self.assertFalse(m.update(NEAR, 0))
        self.assertFalse(m.update(NEAR, 1))
        self.assertTrue(m.update(NEAR, 2))
        for t in range(3, 62):
            self.assertFalse(m.update(NEAR, t))
        self.assertTrue(m.update(NEAR, 62))

    def test_default_three_periods(self):
        m = ProximityMonitor()
        for t in (0, .5, 1, 1.5, 2, 2.5):
            self.assertFalse(m.update(NEAR, t))
        self.assertTrue(m.update(NEAR, 3))

    def test_touch_and_small_overlap_do_not_alert(self):
        for edge in (100, 101):
            m = ProximityMonitor()
            rows = [[80, 30, edge, 90, .9, 0], NEAR[1]]
            for t in range(5):
                self.assertFalse(m.update(rows, t))

    def test_missing_and_long_gap_reset(self):
        m = ProximityMonitor(hold=1, confirmations=2, cooldown=60)
        m.update(NEAR, 0)
        m.update(NEAR, 1)
        self.assertFalse(m.update([], 2))
        self.assertFalse(m.update(NEAR, 3))
        self.assertFalse(m.update(NEAR, 10))

    def test_far_low_conf_and_sill_only(self):
        for rows in ([NEAR[0]], [[0, 0, 10, 10, .9, 0], NEAR[1]],
                     [NEAR[0], [100, 10, 200, 110, .4, 1]],
                     [NEAR[0], [100, 10, 200, 110, .9, 2]]):
            m = ProximityMonitor(hold=1, confirmations=2, cooldown=60)
            for t in range(5):
                self.assertFalse(m.update(rows, t))

    def test_switching_pair_resets(self):
        m = ProximityMonitor(hold=1, confirmations=2, cooldown=60)
        m.update(NEAR, 0)
        m.update(NEAR, 1)
        moved = [[v+500 if i < 4 else v for i,v in enumerate(r)] for r in NEAR]
        self.assertFalse(m.update(moved, 2))

    @patch('telegram_alerts.urlopen')
    def test_telegram_success_and_rejection(self, open_url):
        response = MagicMock()
        open_url.return_value.__enter__.return_value = response
        response.read.return_value = b'{"ok": true}'
        send_message('test-token', '123', 'Test')
        request = open_url.call_args.args[0]
        self.assertEqual(request.get_method(), 'POST')
        self.assertIn(b'"chat_id": "123"', request.data)
        self.assertEqual(open_url.call_args.kwargs['timeout'], 10)
        response.read.return_value = b'{"ok": false}'
        with self.assertRaises(RuntimeError):
            send_message('test-token', '123', 'Test')

    @patch.dict('os.environ', {}, clear=True)
    def test_missing_credentials(self):
        with self.assertRaises(ValueError):
            TelegramAlerts()

    @patch.dict(sys.modules, {'winsound': SimpleNamespace(SND_FILENAME=1, SND_ASYNC=2, SND_NODEFAULT=4, PlaySound=MagicMock())})
    def test_signal_sound_file_is_generated(self):
        sound = SoundAlerts(volume=.28)
        with wave.open(str(sound.path), 'rb') as wav:
            self.assertEqual(wav.getnchannels(), 2)
            self.assertEqual(wav.getsampwidth(), 2)
            self.assertEqual(wav.getframerate(), 44100)
            frames = wav.readframes(wav.getnframes())
        peaks = [abs(int.from_bytes(frames[i:i+2], 'little', signed=True)) for i in range(0, len(frames), 2)]
        with wave.open(str(Path(__file__).resolve().parents[1] / 'assets' / 'alarm.wav'), 'rb') as source:
            signal_bytes = source.getnframes() * 4
            gap_bytes = int(source.getframerate() * .4) * 4
            self.assertEqual(len(frames), signal_bytes * 3 + gap_bytes * 2)
            self.assertEqual(frames[:signal_bytes], frames[signal_bytes + gap_bytes:2 * signal_bytes + gap_bytes])
            self.assertEqual(frames[:signal_bytes], frames[-signal_bytes:])
            self.assertEqual(frames[signal_bytes:signal_bytes + gap_bytes], bytes(gap_bytes))
        self.assertLessEqual(max(peaks), int(32767 * .28) + 1)


if __name__ == '__main__':
    unittest.main()
