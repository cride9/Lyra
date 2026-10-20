"""Local HTTP UI. Browser polling can reconnect without owning the worker."""
from dataclasses import asdict
from pathlib import Path
import time
from urllib.parse import urlparse
import uuid
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from . import jobs
from .config import HOME, SOURCE, TrainConfig
from .io import atomic_json, telemetry
from .metrics import Metrics


def create_app(home=HOME):
    home = Path(home).resolve()
    home.mkdir(parents=True, exist_ok=True)
    web = Path(__file__).parent / 'web'
    app = FastAPI(title='Lyra', docs_url=None, redoc_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', 'testserver'])

    @app.middleware('http')
    async def same_origin(request, call_next):
        origin = request.headers.get('origin')
        if request.method != 'GET' and origin and urlparse(origin).netloc != request.headers.get('host'):
            from fastapi.responses import JSONResponse
            return JSONResponse(dict(detail='Invalid request origin.'), status_code=403)
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        return response

    def settings():
        return {**dict(source=str(SOURCE), run='main', cycles=3, batch_size=4, workers=4,
                       stars=5, candidates=3, preset='Auto', output=str(home / 'generated')),
                **telemetry(home / 'settings.json', {})}

    def run_path(name):
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in name):
            raise ValueError('Run names may contain only ASCII letters, digits, hyphens, and underscores.')
        return home / 'runs' / name

    def launch(action, args, values=None):
        try:
            result = jobs.start(home, action, args)
            if values:
                atomic_json(home / 'settings.json', {**settings(), **values})
            return dict(job=result, message='Background job started.')
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get('/')
    def index():
        return FileResponse(web / 'index.html')

    @app.get('/api/state')
    def state(run: str | None = None):
        values = settings()
        try:
            path = run_path(run or values['run'])
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        job = jobs.latest(home)
        train_state = telemetry(path / 'status.json', {})
        graph = Metrics(path).history() if (path / 'metrics.sqlite').exists() else []
        speed = Metrics(path).recent_speed() if (path / 'metrics.sqlite').exists() else None
        runs = sorted(p.name for p in (home / 'runs').glob('*') if p.is_dir())
        return dict(settings=values, home=str(home), run_path=str(path),
                    run=train_state, history=graph, speed_average=speed, config=telemetry(path / 'config.json', {}),
                    job=jobs.status(job) if job else None, runs=runs,
                    dataset=telemetry(home / 'dataset' / 'manifest.json', {}),
                    preparation=telemetry(home / 'dataset' / 'progress.json', {}),
                    gpu_check=telemetry(home / 'checks' / 'gpu.json', {}),
                    last_checkpoint=str(path / 'last.pt') if (path / 'last.pt').exists() else '',
                    best_checkpoint=str(path / 'best.pt') if (path / 'best.pt').exists() else '',
                    server_time=time.time())

    @app.post('/api/prepare')
    async def prepare(request: Request):
        body = await request.json()
        source = body.get('source') or settings()['source']
        return launch('prepare', dict(source=source, destination=str(home / 'dataset')), dict(source=source))

    @app.get('/api/metrics/export')
    def export_metrics(run: str = 'main', format: str = 'csv', include_archived: bool = False):
        try:
            path = run_path(run)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if format not in ('csv', 'jsonl'):
            raise HTTPException(400, 'Choose CSV or JSONL export.')
        if not (path / 'metrics.sqlite').is_file():
            raise HTTPException(404, 'No metrics are available for this training run yet.')
        return StreamingResponse(Metrics(path).export(format, include_archived),
                media_type='text/csv; charset=utf-8' if format == 'csv' else 'application/x-ndjson',
                headers={'Content-Disposition': f'attachment; filename="lyra-{run}-metrics.{format}"'})

    @app.post('/api/train')
    async def train(request: Request):
        body = await request.json()
        try:
            run = run_path(body.get('run', settings()['run']))
            old = telemetry(run / 'config.json', {}).get('training', {})
            options = {**asdict(TrainConfig()), **old, **body.get('config', {})}
            cfg = TrainConfig(**options)
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        if not (home / 'dataset' / 'manifest.json').is_file():
            raise HTTPException(400, 'Prepare the dataset on the Data tab first.')
        resume = str(run / 'last.pt') if (run / 'last.pt').is_file() else None
        return launch('train', dict(dataset_path=str(home / 'dataset' / 'manifest.json'), run_dir=str(run),
                                   config=asdict(cfg), resume=resume),
                      dict(run=run.name, cycles=cfg.cycles, batch_size=cfg.batch_size, workers=cfg.workers))

    @app.post('/api/stop')
    def stop():
        return dict(message=jobs.cancel(home))

    @app.post('/api/benchmark')
    async def benchmark(request: Request):
        body = await request.json()
        batch = int(body.get('batch_size', 4))
        if batch not in (1, 2, 4, 8):
            raise HTTPException(400, 'Batch size must be 1, 2, 4, or 8.')
        return launch('benchmark', dict(output=str(home / 'checks' / 'gpu.json'), batch_size=batch))

    @app.post('/api/evaluate')
    async def evaluate(request: Request):
        body = await request.json()
        try:
            run = run_path(body.get('run', settings()['run']))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return launch('evaluate', dict(checkpoint_path=str(run / 'best.pt'), dataset_path=str(home / 'dataset' / 'manifest.json'),
                      output=str(run / 'evaluation-test.json'), count=128))

    @app.post('/api/generate')
    async def generate(request: Request):
        body = await request.json()
        try:
            from .generation import GenerationConfig
            cfg = GenerationConfig(**body.get('config', {}))
            cfg.validate()
            inputs = body['inputs']
            checkpoint = body['checkpoint']
            output = body.get('output') or settings()['output']
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(400, str(exc)) from exc
        return launch('generate', dict(checkpoint_path=checkpoint, inputs=inputs, output_dir=output, config=asdict(cfg)),
                      dict(stars=cfg.stars, preset=cfg.preset, candidates=cfg.candidates, output=output))

    @app.post('/api/upload')
    async def upload(file: UploadFile = File(...)):
        suffix = Path(file.filename or '').suffix.lower()
        if suffix not in ('.mp3', '.ogg', '.wav', '.flac'):
            raise HTTPException(400, 'Choose an MP3, OGG, WAV, or FLAC file.')
        folder = home / 'inputs'
        folder.mkdir(exist_ok=True)
        path = folder / (Path(file.filename).stem[:80] + '-' + uuid.uuid4().hex[:6] + suffix)
        size = 0
        try:
            with path.open('wb') as out:
                while chunk := await file.read(1024 ** 2):
                    size += len(chunk)
                    if size > 512 * 1024 ** 2:
                        raise HTTPException(400, 'The audio file must not exceed 512 MiB.')
                    out.write(chunk)
            return dict(path=str(path))
        except Exception:
            path.unlink(missing_ok=True)
            raise
        finally:
            await file.close()

    @app.get('/api/download')
    def download(path: str):
        target = Path(path).resolve()
        if not target.is_file() or target.suffix.lower() != '.osz':
            raise HTTPException(404, 'Map not found.')
        output = Path(settings()['output']).resolve()
        if not target.is_relative_to(output):
            raise HTTPException(403, 'The file is outside the generated maps directory.')
        return FileResponse(target, filename=target.name)

    app.mount('/static', StaticFiles(directory=web), name='static')
    return app
