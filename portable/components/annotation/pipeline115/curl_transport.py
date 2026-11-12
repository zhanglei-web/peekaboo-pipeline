"""Audited curl HTTP/2 negotiation for unchanged Responses SSE requests."""
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import tempfile
import time
from .common import atomic, now


class CurlResponse:
    def __init__(self, proc):
        self.proc = proc
        self.status_code = 0
        self.headers = {}
        self.protocol = None
        # No proxy CONNECT or redirects are used. Skip informational responses.
        while True:
            status_line = proc.stdout.readline()
            if not status_line:
                return
            if not status_line.startswith(b'HTTP/'):
                raise ValueError('Unexpected selector HTTP status line')
            fields = status_line.decode('latin1').strip().split(' ', 2)
            self.protocol = fields[0]
            self.status_code = int(fields[1])
            self.headers = {}
            while True:
                line = proc.stdout.readline()
                if not line or line in (b'\r\n', b'\n'):
                    break
                key, value = line.decode('latin1').split(':', 1)
                self.headers[key.lower()] = value.strip()
            if not 100 <= self.status_code < 200:
                return

    def iter_lines(self, **kwargs):
        for line in self.proc.stdout:
            yield line.rstrip(b'\r\n')


@contextmanager
def post(url, *, json_body, headers, policy, audit_path):
    """Keep credentials in stdin and request bytes in a private temporary file."""
    if any('\n' in v or '\r' in v for v in headers.values()):
        raise ValueError('Invalid selector HTTP header')
    key = headers['Authorization'].removeprefix('Bearer ')
    audit_path = Path(audit_path)
    started = time.monotonic()
    record = {'transport': 'curl_http2_responses_sse', 'started_at': now(),
              'redirects': False, 'proxy_environment': False,
              'credentials_in_argv': False, 'status': 'started'}
    atomic(audit_path, record)
    with tempfile.TemporaryDirectory(prefix='.selector-upload-', dir=audit_path.parent) as tmp:
        tmp = Path(tmp)
        request_path = tmp/'request.json'
        request_path.write_text(json.dumps(json_body, ensure_ascii=False, allow_nan=False))
        request_path.chmod(0o600)
        cmd = ['curl', '--disable', '--config', '-', '--noproxy', '*', '--http2',
               '--silent', '--show-error', '--no-buffer', '--include',
               '--connect-timeout', str(policy['connect_timeout_s']),
               '--max-time', str(policy['total_timeout_s']),
               '--speed-time', str(policy['read_timeout_s']), '--speed-limit', '1',
               '--request', 'POST', '--header', 'Expect:',
               '--data-binary', '@'+str(request_path), url]
        config = '\n'.join('header = '+json.dumps(k+': '+v)
                           for k, v in headers.items())+'\n'
        proc = None
        try:
            with (tmp/'stderr.txt').open('w+b') as errors:
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=errors)
                proc.stdin.write(config.encode()); proc.stdin.close()
                response = CurlResponse(proc)
                record.update(http_status=response.status_code,
                              negotiated_protocol=response.protocol)
                if not response.status_code:
                    code = proc.wait(timeout=10)
                    from astra_memory.provider import TransportError
                    raise TransportError('selector curl connection failed; exit '+str(code), True)
                yield response
                record['status'] = 'received_completed_response'
        except BaseException as exc:
            record.update(status='failed', error_type=type(exc).__name__)
            raise
        finally:
            if proc is not None:
                # Only the owned HTTP client is stopped, after response parsing
                # has finished or failed. No model service is signalled.
                if proc.poll() is None:
                    try: proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.terminate()
                        try: proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            proc.kill(); proc.wait(timeout=5)
                record['curl_exit_code'] = proc.returncode
                proc.stdout.close()
            error_path = tmp/'stderr.txt'
            if error_path.exists():
                record['transport_diagnostic'] = error_path.read_text(errors='replace').replace(key, '[REDACTED]')[-2000:]
            record['wall_seconds'] = time.monotonic()-started
            atomic(audit_path, record)
