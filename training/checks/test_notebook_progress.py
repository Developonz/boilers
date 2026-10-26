"""Bounded notebook rendering, complete journals and real kernel display updates."""
from contextlib import redirect_stdout
from pathlib import Path
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'lib'))
from monitoring_v3.progress import Progress
from monitoring_v3.progress_display import NotebookDisplay,render


class Handle:
    def __init__(self):self.updates=[]
    def update(self,content):self.updates.append(content)


class NotebookProgressTests(unittest.TestCase):
    def test_one_display_keeps_full_target_journal_without_stdout_spam(self):
        handle=Handle();created=[]
        def display(content,display_id):created.append(content);return handle
        view=NotebookDisplay(display,lambda html:html,refresh_seconds=0)
        stream=io.StringIO()
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(stream),patch(
                'monitoring_v3.progress.notebook_display',return_value=view):
            progress=Progress(tmp)
            with progress.stage('monthly'):
                for index in range(1,101):
                    progress.say('fast start',status='target_started',target='<level>',target_index=index,
                                 target_count=100,iterations=20)
                    progress.set_detail('training',iteration=10,iterations=20)
                    progress.say('fast done',status='target_complete',target='<level>',target_index=index,target_count=100)
                with progress.stage('validation'):
                    progress.say('predict',status='prediction_started',target='pressure',target_index=1,target_count=2)
                progress.say('decision',status='candidate_decision')
            progress.say('done models.csv',status='complete')
            records=[json.loads(line) for line in (Path(tmp)/'progress.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual(stream.getvalue(),'');self.assertEqual(len(created),1)
        self.assertEqual(sum(r['status']=='target_started' for r in records),100)
        self.assertEqual(sum(r['status']=='target_complete' for r in records),100)
        self.assertIn('Расчёт завершён',handle.updates[-1])
        self.assertIn('done models.csv',handle.updates[-1])
        self.assertTrue(any('value="100.000000" max="100"' in html for html in handle.updates))
        self.assertTrue(any('&lt;level&gt;' in html and 'деревья 10/20' in html for html in handle.updates))
        self.assertTrue(any('validation' in html and 'без счётчика' in html for html in handle.updates))

    def test_refresh_is_throttled_but_stage_changes_are_immediate(self):
        handle=Handle();created=[]
        def display(content,display_id):created.append(content);return handle
        view=NotebookDisplay(display,lambda html:html,refresh_seconds=.4)
        snapshot=dict(phase='train',target_count=10,completed=2)
        with patch('monitoring_v3.progress_display.time.perf_counter',side_effect=[0,.1,.5,.6]):
            view.update(snapshot);view.update(snapshot);view.update(snapshot);view.update(snapshot,force=True)
        self.assertEqual(len(created),1);self.assertEqual(len(handle.updates),2)
        html=render(dict(phase='<script>alert(1)</script>',target='<unsafe>',target_count=10,
                         completed=2,target_index=3,iteration=10,iterations=20))
        self.assertNotIn('<script>',html);self.assertIn('&lt;unsafe&gt;',html)
        self.assertIn('20.0%',html)
        self.assertIn('ориентир на остальные цели: 2 мин 00 с',
                      render(dict(phase='train',target_count=10,completed=2,eta=120)))

    def test_display_failure_falls_back_to_console_without_losing_journal(self):
        class Broken:
            def update(self,*args,**kwargs):raise RuntimeError('frontend unavailable')
        stream=io.StringIO()
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(stream),patch(
                'monitoring_v3.progress.notebook_display',return_value=Broken()):
            progress=Progress(tmp);progress.say('still running')
            self.assertIsNone(progress.view)
            record=json.loads((Path(tmp)/'progress.jsonl').read_text(encoding='utf-8'))
        self.assertIn('still running',stream.getvalue());self.assertEqual(record['message'],'still running')

    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('nbformat','nbclient','ipykernel')),
                         'Notebook integration dependencies unavailable')
    def test_real_notebook_pipeline_updates_one_output(self):
        import nbformat
        from nbclient import NotebookClient
        from jupyter_client import KernelManager
        with tempfile.TemporaryDirectory() as tmp:
            source='\n'.join([
                'import sys,json',
                f'sys.path[:0] = {json.dumps([str(ROOT/"lib"),str(ROOT/"checks")])}',
                'from test_runtime_performance import fixture',
                'from monitoring_v3.pipeline import run',
                f'cfg=fixture({json.dumps(tmp)})',
                'result=run(cfg)',
                'print("NOTEBOOK_OK",result["status"].status.eq("complete").all())',
            ])
            book=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)])
            km=KernelManager(kernel_name='python3')
            km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
            try:
                executed=NotebookClient(book,km=km,timeout=120).execute(cwd=str(ROOT))
            finally:
                if km.has_kernel:km.shutdown_kernel(now=True)
            outputs=executed.cells[0].outputs
            displays=[o for o in outputs if o.output_type=='display_data' and 'text/html' in o.data]
            self.assertEqual(len(displays),1)
            self.assertIn('<progress',displays[0].data['text/html'])
            self.assertIn('Расчёт завершён',displays[0].data['text/html'])
            self.assertIn('models.csv',displays[0].data['text/html'])
            text=''.join(o.text for o in outputs if o.output_type=='stream')
            self.assertIn('NOTEBOOK_OK True',text)
            self.assertNotIn('CatBoost · цель',text);self.assertNotIn('прогноз · датчик',text)
            records=[json.loads(line) for line in (Path(tmp)/'out'/'progress.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertTrue(any(r.get('status')=='candidate_decision' for r in records))
            self.assertTrue(any(r.get('status')=='prediction_complete' for r in records))


if __name__=='__main__':unittest.main()
