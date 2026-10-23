"""Ночной запуск 02_Обучение.ipynb обычным Python, без регистрации ядра.

python run_overnight.py
python run_overnight.py --config my_settings.json
"""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import sys
import traceback


def execute(notebook, output, config=None, check_only=False):
    notebook, output = Path(notebook).resolve(), Path(output).resolve()
    if notebook == output:
        raise ValueError('Исполненный результат нужно сохранять отдельно от исходного ноутбука.')
    output.parent.mkdir(parents=True, exist_ok=True)
    nb = json.loads(notebook.read_text(encoding='utf-8'))
    if check_only:
        stop = next(i for i, cell in enumerate(nb['cells']) if cell['cell_type'] == 'code' and
                    'LF.run(CFG)' in ''.join(cell['source']))
        nb['cells'] = nb['cells'][:stop]
    def save(**kwargs):
        tmp = output.with_suffix(output.suffix + '.tmp')
        tmp.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
        os.replace(tmp, output)
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            cell.update(execution_count=None, outputs=[])
    previous = os.environ.get('BOILER_CONFIG_JSON')
    previous_backend = os.environ.get('MPLBACKEND')
    os.environ['MPLBACKEND'] = 'Agg'  # Save plots without GUI windows in a night run.
    if config:
        os.environ['BOILER_CONFIG_JSON'] = str(Path(config).resolve())
    old_cwd, old_path = Path.cwd(), list(sys.path)
    namespace = {'__name__': '__main__', '__file__': str(notebook)}
    os.chdir(notebook.parent)
    sys.path.insert(0, str(notebook.parent))
    if (notebook.parent / 'lib').is_dir():
        sys.path.insert(0, str(notebook.parent / 'lib'))
    print(f'Python: {sys.executable}\nНоутбук: {notebook}\nВывод: {output}', flush=True)
    from IPython.core.interactiveshell import InteractiveShell
    shell = InteractiveShell.instance()
    original_publish = shell.display_pub.publish
    active_cell = None
    def publish(data, metadata=None, **kwargs):
        if active_cell is not None:
            active_cell['outputs'].append(dict(output_type='display_data', data=data, metadata=metadata or {}))
    shell.display_pub.publish = publish
    try:
        count = 0
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] != 'code':
                continue
            count += 1
            print(f'Ячейка {count}: выполнение', flush=True)
            cell['execution_count'] = count
            active_cell = cell
            captured_out, captured_err = io.StringIO(), io.StringIO()
            failure = None
            try:
                with redirect_stdout(_Tee(sys.stdout, captured_out)), redirect_stderr(_Tee(sys.stderr, captured_err)):
                    exec(compile(''.join(cell['source']), f'{notebook.name}:cell_{i}', 'exec'), namespace)
            except BaseException as exc:
                failure = exc
                lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
                cell['outputs'].append(dict(output_type='error', ename=type(exc).__name__,
                                             evalue=str(exc), traceback=lines))
            finally:
                for name, buffer in [('stdout', captured_out), ('stderr', captured_err)]:
                    if buffer.getvalue():
                        cell['outputs'].insert(0, dict(output_type='stream', name=name, text=buffer.getvalue().splitlines(keepends=True)))
                save()
            if failure is not None:
                raise failure
    finally:
        shell.display_pub.publish = original_publish
        os.chdir(old_cwd)
        sys.path[:] = old_path
        if previous is None:
            os.environ.pop('BOILER_CONFIG_JSON', None)
        else:
            os.environ['BOILER_CONFIG_JSON'] = previous
        if previous_backend is None:
            os.environ.pop('MPLBACKEND', None)
        else:
            os.environ['MPLBACKEND'] = previous_backend
    print('Проверка входов завершена.' if check_only else 'Все ячейки выполнены.', flush=True)
    return nb


class _Tee:
    """Keep console progress visible and save it into the executed notebook."""
    def __init__(self, stream, buffer):
        self.stream, self.buffer = stream, buffer

    def write(self, text):
        self.buffer.write(text)
        return self.stream.write(text)

    def flush(self):
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook', default=str(Path(__file__).parent / '02_Обучение.ipynb'))
    parser.add_argument('--output', default=None)
    parser.add_argument('--config')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    suffix = 'preflight' if args.check_only else 'executed'
    output = args.output or str(Path(__file__).parent / 'runs' / f'02_{suffix}.ipynb')
    execute(args.notebook, output, args.config, args.check_only)
