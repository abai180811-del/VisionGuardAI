"""Bounded background Telegram sender; credentials never printed."""
import json
import os
import queue
import threading
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


def delivery_error(exc):
    """Explain failures without exposing the token-bearing request URL."""
    if isinstance(exc, HTTPError):
        return {
            400: 'Telegram rejected the request. Check TELEGRAM_CHAT_ID and press Start in your bot chat.',
            401: 'Invalid or revoked bot token. Enter the current token from BotFather.',
            403: 'Bot cannot message this chat. Unblock/start the bot or check group permissions.',
            404: 'Bot endpoint not found. Check that you pasted the complete bot token.',
            429: 'Telegram rate limit reached. Wait before trying again.',
        }.get(exc.code, f'Telegram HTTP error {exc.code}. Try again later.')
    if isinstance(exc, (URLError, TimeoutError, OSError)):
        return 'Network connection failed or timed out. Check internet access and firewall access to api.telegram.org.'
    return 'Telegram returned an unexpected response. Check configuration and try again.'


def send_message(token, chat_id, text):
    request = Request(f'https://api.telegram.org/bot{token}/sendMessage',
                      data=json.dumps({'chat_id': chat_id, 'text': text}).encode(),
                      headers={'Content-Type': 'application/json'}, method='POST')
    with urlopen(request, timeout=10) as response:
        if not json.load(response).get('ok'):
            raise RuntimeError('Telegram rejected the message')


class TelegramAlerts:
    def __init__(self):
        self.token = os.environ.get('TELEGRAM_BOT_TOKEN', '').strip()
        self.chat_id = os.environ.get('TELEGRAM_CHAT_ID', '').strip()
        if not self.token or not self.chat_id:
            raise ValueError('--telegram requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables.')
        self.messages = queue.Queue(maxsize=1)
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def submit(self, text):
        try:
            self.messages.put_nowait(text)
        except queue.Full:
            print('Telegram busy; alert not queued. Check connectivity.')

    def _run(self):
        while True:
            text = self.messages.get()
            try:
                send_message(self.token, self.chat_id, text)
                print('Telegram alert delivered.')
            except Exception as exc:
                # HTTP errors may contain the token in their URL. Never print them.
                print('Telegram delivery failed: ' + delivery_error(exc))
            finally:
                self.messages.task_done()


if __name__ == '__main__':
    # Direct delivery test, independent of the camera and proximity heuristic.
    token = os.environ.get('TELEGRAM_BOT_TOKEN', '').strip()
    chat_id = os.environ.get('TELEGRAM_CHAT_ID', '').strip()
    if not token or not chat_id:
        print('Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID first.')
        raise SystemExit(1)
    try:
        send_message(token, chat_id, 'VisionGuardAI connection test: Telegram delivery is working. This is not a proximity alert.')
        print('Telegram connection test delivered successfully.')
    except Exception as exc:
        print('Telegram connection test failed: ' + delivery_error(exc))
        raise SystemExit(1)
