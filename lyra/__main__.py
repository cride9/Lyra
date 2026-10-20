import argparse
import json
from .config import HOME, SOURCE


def main():
    parser = argparse.ArgumentParser(prog='lyra', description='Lyra: one jointly trained audio-to-beatmap model.')
    commands = parser.add_subparsers(dest='command', required=True)
    ui = commands.add_parser('ui')
    ui.add_argument('--home', default=str(HOME)); ui.add_argument('--port', type=int, default=7862)
    ui.add_argument('--open-browser', action='store_true')
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--source', default=str(SOURCE)); prepare.add_argument('--destination', default=str(HOME / 'dataset'))
    train = commands.add_parser('train')
    train.add_argument('--dataset-path', default=str(HOME / 'dataset' / 'manifest.json'))
    train.add_argument('--run-dir', default=str(HOME / 'runs' / 'main'))
    train.add_argument('--resume'); train.add_argument('--cycles', type=int, default=3)
    train.add_argument('--batch-size', type=int, default=4); train.add_argument('--workers', type=int, default=4)
    train.add_argument('--effective-batch', type=int, default=32)
    benchmark = commands.add_parser('benchmark')
    benchmark.add_argument('--output', default=str(HOME / 'checks' / 'gpu.json'))
    benchmark.add_argument('--device', default='auto'); benchmark.add_argument('--tiny', action='store_true')
    benchmark.add_argument('--updates', type=int, default=3)
    benchmark.add_argument('--batch-size', type=int, default=1)
    generate = commands.add_parser('generate')
    generate.add_argument('checkpoint_path'); generate.add_argument('inputs')
    generate.add_argument('--output-dir', default=str(HOME / 'generated'))
    generate.add_argument('--stars', type=float, default=5); generate.add_argument('--bpm', type=float)
    generate.add_argument('--offset', type=float); generate.add_argument('--timing-map')
    evaluate = commands.add_parser('evaluate')
    evaluate.add_argument('checkpoint_path'); evaluate.add_argument('--dataset-path', default=str(HOME / 'dataset' / 'manifest.json'))
    evaluate.add_argument('--output', default=str(HOME / 'evaluation.json')); evaluate.add_argument('--count', type=int, default=128)
    worker = commands.add_parser('worker'); worker.add_argument('path')
    args = vars(parser.parse_args())
    command = args.pop('command')
    if command == 'ui':
        import uvicorn
        from .server import create_app
        port = args.pop('port')
        if args.pop('open_browser'):
            import threading
            import webbrowser
            threading.Timer(2, lambda: webbrowser.open(f'http://127.0.0.1:{port}')).start()
        uvicorn.run(create_app(**args), host='127.0.0.1', port=port)
        return
    if command == 'worker':
        from .jobs import worker
        worker(**args)
        return
    if command == 'train':
        from .config import TrainConfig
        from dataclasses import asdict
        args['config'] = asdict(TrainConfig(**{k: args.pop(k) for k in ('cycles', 'batch_size', 'workers', 'effective_batch')}))
    if command == 'generate':
        args['config'] = {k: args.pop(k) for k in ('stars', 'bpm', 'offset', 'timing_map')}
    from .jobs import dispatch
    print(json.dumps(dispatch(command, args), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
