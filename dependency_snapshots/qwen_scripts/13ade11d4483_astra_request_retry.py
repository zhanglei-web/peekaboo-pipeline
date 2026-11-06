"""Bounded transport recovery; never accepts an unverified model response."""
import json
from pathlib import Path
import time
import uuid

from astra_memory.provider import post_json, response_text, TransportError, ModelServiceError


def request_verified(folder, model, body, key, signature, attempts=3):
    # Old providers sometimes returned a completed envelope without model identity.
    # A rejected completed response must not be reused via its idempotency key.
    request_id = signature + '-' + uuid.uuid4().hex
    archive = Path(folder)/'transport_attempts'
    archive.mkdir(exist_ok=True)
    prior = Path(folder)/'response.json'
    if prior.exists():
        (archive/('prior-'+uuid.uuid4().hex+'.response.json')).write_bytes(prior.read_bytes())
    for index in range(attempts):
        record = archive/(uuid.uuid4().hex+'.json')
        started = time.time()
        entry = {'request_id': request_id, 'started_unix': started, 'attempt': index+1}
        try:
            raw_bytes = post_json(model['base_url'].rstrip('/')+'/responses', body, key,
                                  model['timeout_s'], request_id)
            raw_bytes = raw_bytes.replace(key.encode(), b'[REDACTED]')
            raw_path = record.with_suffix('.response.json')
            raw_path.write_bytes(raw_bytes)
            (Path(folder)/'response.json').write_bytes(raw_bytes)
            raw = json.loads(raw_bytes)
            entry['response_file'] = raw_path.name
            result = response_text(raw, model)  # Original exact identity/completion gate.
            entry['status'] = 'verified'
            return raw, result
        except TransportError as exc:
            entry.update(status='transport_error', error=str(exc), retryable=exc.retryable)
            if not exc.retryable or index+1 == attempts:
                raise
            # Timeout may still have completed server-side; retain this request ID.
        except ModelServiceError as exc:
            missing_identity = isinstance(raw, dict) and not raw.get('model')
            entry.update(status='unverified_response', error=str(exc), retryable=missing_identity)
            if not missing_identity or index+1 == attempts:
                raise
            request_id = signature + '-' + uuid.uuid4().hex
        except Exception as exc:
            entry.update(status='invalid_response', error_type=type(exc).__name__, retryable=False)
            raise
        finally:
            entry['wall_seconds'] = time.time()-started
            record.write_text(json.dumps(entry, indent=2)+'\n')
        time.sleep(min(5*(index+1), 15))
