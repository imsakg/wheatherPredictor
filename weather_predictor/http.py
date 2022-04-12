"""Bounded HTTPS JSON GETs with retries and credential-safe public errors."""
import json
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, build_opener
from .units import finite


class SameOriginHTTPSRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        source, target = urlsplit(req.full_url), urlsplit(newurl)
        if target.scheme != "https" or (source.hostname,source.port or 443) != (target.hostname,target.port or 443) or target.username or target.password:
            raise HTTPError(req.full_url,403,"unsafe redirect blocked",headers,None)
        return super().redirect_request(req,fp,code,msg,headers,newurl)


def _invalid_constant(value):
    raise ValueError("non-finite JSON constant")


class JsonTransport:
    def __init__(self, timeout=10, attempts=3, opener=None, sleep=time.sleep, max_bytes=2000000):
        self.timeout = finite(timeout)
        if self.timeout <= 0 or not isinstance(attempts,int) or not 1 <= attempts <= 5 or max_bytes < 1:
            raise ValueError("invalid HTTP resource limits")
        self.attempts, self.opener, self.sleep, self.max_bytes = attempts, opener if opener is not None else build_opener(SameOriginHTTPSRedirect()).open, sleep, max_bytes

    def __call__(self, url):
        if urlsplit(url).scheme != "https":
            raise ValueError("only HTTPS endpoints are allowed")
        for attempt in range(self.attempts):
            try:
                with self.opener(url,timeout=self.timeout) as response:
                    body = response.read(self.max_bytes+1)
                if len(body) > self.max_bytes:
                    raise ValueError("provider response exceeds size limit")
                try:
                    return json.loads(body,parse_constant=_invalid_constant)
                except (UnicodeDecodeError,ValueError):
                    raise ValueError("provider returned invalid JSON") from None
            except (HTTPError,URLError,TimeoutError,OSError) as exc:
                retryable = not isinstance(exc,HTTPError) or exc.code in (429,500,502,503,504)
                if not retryable or attempt+1 == self.attempts:
                    raise RuntimeError("weather request failed; check connectivity, quota and credentials") from None
                self.sleep(min(2**attempt,30))
