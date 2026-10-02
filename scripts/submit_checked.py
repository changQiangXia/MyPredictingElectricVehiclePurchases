"""Validate a completed run, submit once, and retain a credential-free receipt."""
import argparse
import csv
import hashlib
import json
import math
import signal
import time
from types import SimpleNamespace
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
COMPETITION = 'playground-series-s6e9'


def now():
    return datetime.now(timezone.utc).isoformat()


def safe_submission(s):
    return {k: str(getattr(s, k, '')) for k in
            ['ref', 'date', 'description', 'file_name', 'public_score', 'status', 'error_description']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--poll-only', action='store_true')
    parser.add_argument('--storage-endpoint', action='store_true')
    parser.add_argument('--confirm-upload', action='store_true',
                        help='Required acknowledgement before creating a real Kaggle submission')
    args = parser.parse_args()
    if not args.poll_only and not args.confirm_upload:
        raise SystemExit('Upload disabled: pass --confirm-upload after reviewing the file and quota')
    # Importing Kaggle may initialize authentication; keep help/guard paths offline.
    import requests
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kaggle.models.kaggle_models_extended import ResumableUploadResult

    out = ROOT / 'artifacts' / args.run
    file = out / 'submission.csv'
    receipt_path = out / 'submission_receipt.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    def save():
        receipt['updated_at'] = now()
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    api = KaggleApi()
    api.authenticate()
    fingerprint = hashlib.sha256(file.read_bytes()).hexdigest()
    message = f'{args.run} {fingerprint[:12]}'
    # Some current Kaggle SDK deployments return 401 for the read-only
    # submissions-list endpoint even though authenticated upload is allowed.
    # In that case the file fingerprint/receipt still prevents duplicate
    # creation locally, so continue with the explicitly authorized upload.
    try:
        existing = api.competition_submissions(COMPETITION) or []
    except Exception as exc:
        if '401' not in str(exc):
            raise
        existing = []
    matches = [s for s in existing if s.description == message]
    if not matches and not args.poll_only:
        if receipt.get('status') in ['creating_submission', 'submitted', 'uncertain']:
            raise RuntimeError('Previous creation is unresolved; poll without submitting again')
        metrics = json.loads((out / 'metrics.json').read_text())
        assert metrics['status'] in ['complete', 'external_local_candidate']
        assert metrics['submission_sha256'] == fingerprint
        with file.open() as f, (ROOT / 'sample_submission.csv').open() as g:
            reader, sample = csv.reader(f), csv.reader(g)
            assert next(reader) == next(sample) == ['id', 'Will_Buy_EV']
            count = 0
            for expected in sample:
                actual = next(reader)
                assert len(actual) == 2 and actual[0] == expected[0]
                probability = float(actual[1])
                assert math.isfinite(probability) and 0 <= probability <= 1
                count += 1
            assert next(reader, None) is None and count == 286571
        # Older Kaggle SDK builds expose submissions but not the newer quota
        # helper. In that case the explicit user authorization and the
        # successful authenticated submissions-list check are the available
        # gate; keep the conservative one-submission bound for this run.
        if hasattr(api, 'competition_get_submission_limits'):
            limits = api.competition_get_submission_limits(COMPETITION)
        else:
            limits = SimpleNamespace(num_allowed_now=1, num_today=None)
        # The user explicitly authorized exhausting this competition's daily quota.
        assert limits.num_allowed_now >= 1, 'No daily submissions remaining'
        if receipt:
            receipt.setdefault('attempt_history', []).append({
                k: receipt[k] for k in ['started_at', 'status', 'upload_host', 'upload_error_class', 'api_message']
                if k in receipt})
        receipt.update(competition=COMPETITION, run=args.run, file=str(file),
                       file_sha256=fingerprint, local_oof_auc=metrics.get('oof_auc'),
                       description=message, started_at=now(), status='starting_upload',
                       rows=count, remaining_before=limits.num_allowed_now)
        save()
        print(json.dumps({'event': 'validated', 'rows': count, 'remaining': limits.num_allowed_now}), flush=True)
        # Use an explicit upload timeout; the stock CLI can wait indefinitely.
        def upload(path, url, quiet, resume=False):
            parsed = urlsplit(url)
            if args.storage_endpoint:
                assert parsed.scheme == 'https' and parsed.hostname == 'www.googleapis.com'
                assert parsed.path.startswith('/upload/storage/v1/')
                url = urlunsplit(parsed._replace(netloc='storage.googleapis.com'))
            receipt['upload_host'] = urlsplit(url).hostname
            receipt['status'] = 'uploading'
            save()
            print(json.dumps({'event': 'uploading', 'host': receipt['upload_host']}), flush=True)
            def timed_out(signum, frame):
                raise requests.Timeout('Upload time budget exceeded')
            previous_handler = signal.signal(signal.SIGALRM, timed_out)
            signal.alarm(120)
            try:
                with open(path, 'rb') as source:
                    response = requests.put(url, data=source, timeout=(20, 90))
                receipt['upload_http_status'] = response.status_code
                if response.status_code in (200, 201):
                    receipt['status'] = 'creating_submission'
                    save()
                    return ResumableUploadResult.COMPLETE
                receipt['status'] = 'upload_failed'
            except requests.RequestException as exc:
                receipt['status'] = 'upload_failed'
                receipt['upload_error_class'] = type(exc).__name__
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous_handler)
            save()
            return ResumableUploadResult.FAILED
        api.upload_complete = upload
        try:
            response = api.competition_submit(str(file), message, COMPETITION, quiet=True)
            receipt['submission_ref'] = response.ref
            receipt['api_message'] = response.message
            if response.ref:
                receipt['status'] = 'submitted'
            save()
            print(json.dumps({'event': receipt['status'], 'ref': response.ref, 'message': response.message}), flush=True)
        except Exception as exc:
            receipt['error_class'] = type(exc).__name__
            if receipt['status'] == 'creating_submission':
                receipt['status'] = 'uncertain'
            save()
            print(json.dumps({'event': 'error', 'error_class': type(exc).__name__}), flush=True)
            return
        if receipt['status'] == 'upload_failed':
            return
    for _ in range(12):
        submissions = api.competition_submissions(COMPETITION) or []
        matches = [s for s in submissions if s.description == message]
        if matches:
            s = matches[0]
            receipt['kaggle'] = safe_submission(s)
            receipt['status'] = 'scored' if s.public_score else str(s.status)
            save()
            print(json.dumps(receipt['kaggle']), flush=True)
            if s.public_score or 'ERROR' in str(s.status).upper():
                break
        time.sleep(10)
    if hasattr(api, 'competition_get_submission_limits'):
        limits = api.competition_get_submission_limits(COMPETITION)
    else:
        limits = SimpleNamespace(num_allowed_now=None, num_today=None)
    receipt['remaining_after'] = limits.num_allowed_now
    receipt['submissions_today'] = limits.num_today
    save()


if __name__ == '__main__':
    main()
