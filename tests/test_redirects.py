import io
import unittest
from urllib.request import Request
from urllib.error import HTTPError
from weather_predictor.http import SameOriginHTTPSRedirect, JsonTransport

class RedirectTests(unittest.TestCase):
    def test_reject_downgrade_and_cross_origin(self):
        handler=SameOriginHTTPSRedirect(); request=Request('https://provider.test/weather?appid=secret')
        for destination in ('http://provider.test/weather?appid=secret','https://other.test/weather'):
            with self.assertRaises(HTTPError): handler.redirect_request(request,None,302,'found',{},destination)
        redirected=handler.redirect_request(request,None,302,'found',{},'https://provider.test/new')
        self.assertEqual(redirected.full_url,'https://provider.test/new')
    def test_nonstandard_json_is_rejected(self):
        with self.assertRaises(ValueError): JsonTransport(opener=lambda *a,**k:io.BytesIO(b'{"x":NaN}'))('https://provider.test')
