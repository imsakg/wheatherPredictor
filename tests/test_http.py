import io
import unittest
from urllib.error import HTTPError, URLError
from weather_predictor.http import JsonTransport

class HttpTests(unittest.TestCase):
    def test_retry_then_decode(self):
        calls=[]; waits=[]
        def opener(url,timeout):
            calls.append(timeout)
            if len(calls)==1: raise HTTPError(url,429,'limited',{},None)
            return io.BytesIO(b'{"ok":true}')
        self.assertEqual(JsonTransport(opener=opener,sleep=waits.append)('https://example.test'),{'ok':True})
        self.assertEqual(len(calls),2)
        self.assertEqual(waits,[1])
    def test_safe_errors_and_https(self):
        def opener(url,timeout): raise URLError(url)
        with self.assertRaisesRegex(RuntimeError,'request failed') as ctx:
            JsonTransport(opener=opener,sleep=lambda _:None)('https://x.test?appid=secret')
        self.assertNotIn('secret',str(ctx.exception))
        with self.assertRaises(ValueError): JsonTransport()('http://x.test')
    def test_invalid_json(self):
        with self.assertRaises(ValueError): JsonTransport(opener=lambda *a,**k:io.BytesIO(b'bad'))('https://x.test')
