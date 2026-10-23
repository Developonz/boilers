"""Chronology of filters and information retained by the report plots."""
import io
from contextlib import redirect_stdout
from unittest.mock import patch
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from test_fleet import metric_context, F, M
from legacy_cleaning import causal_clean_numeric
from legacy_plots import plot_record


class CleaningReportTests(unittest.TestCase):
    def test_future_append_and_replacement_cannot_rewrite_cleaned_past(self):
        rng = np.random.default_rng(14)
        ix = pd.date_range('2025-01-01', periods=600, freq='h')
        frame = pd.DataFrame(rng.normal(size=(600, 3)), index=ix, columns=list('abc'))
        frame.iloc[220, 0] = 1000
        frame.iloc[250:350, 1] = frame.iloc[249, 1]
        run = pd.Series(True, index=ix)
        full, flags = causal_clean_numeric(frame, run, frame.columns)
        for cutoff in (221, 260, 274, 350, 480):
            prefix, masks = causal_clean_numeric(frame.iloc[:cutoff], run.iloc[:cutoff], frame.columns)
            pd.testing.assert_frame_equal(prefix, full.iloc[:cutoff])
            for kind in flags:
                pd.testing.assert_frame_equal(masks[kind], flags[kind].iloc[:cutoff])
        altered = frame.copy(); altered.iloc[400:] *= 1e8
        changed, masks = causal_clean_numeric(altered, run, frame.columns)
        pd.testing.assert_frame_equal(full.iloc[:400], changed.iloc[:400])
        for kind in flags:
            pd.testing.assert_frame_equal(flags[kind].iloc[:400], masks[kind].iloc[:400])
        self.assertTrue(flags['spikes'].iloc[220, 0])
        self.assertFalse(flags['frozen']['b'].iloc[249:272].any())
        self.assertTrue(flags['frozen']['b'].iloc[272])

    def test_subhourly_filter_is_also_prefix_invariant(self):
        rng = np.random.default_rng(10)
        frame = pd.DataFrame({'sensor': rng.normal(size=1500)}, index=pd.date_range('2025-01-01', periods=1500, freq='5min'))
        run = np.ones(1500, bool)
        full, _ = causal_clean_numeric(frame, run, frame.columns, per_h=12)
        prefix, _ = causal_clean_numeric(frame.iloc[:1000], run[:1000], frame.columns, per_h=12)
        pd.testing.assert_frame_equal(full.iloc[:1000], prefix)

    def test_original_plot_layout_daily_maximum_and_current_scale(self):
        ns = metric_context(); ix = ns['t_ix']
        score = pd.Series(.4, index=ix); score.iloc[200:204] = 20
        res = dict(score=score, thr=pd.Series(1., index=ix))
        record = dict(meta=dict(group='Группа 1', target='К-1', donors='К-1'), context=ns, views={'test': res})
        fig = plot_record(record, 'test')
        try:
            self.assertEqual(len(fig.axes), 2)
            self.assertEqual(len(fig.axes[0].lines[0].get_xdata()), len(score.resample('1D').max()))
            self.assertEqual(fig.axes[0].get_yscale(), 'symlog')
            self.assertEqual(len(fig.legends[0].get_texts()), 8)
            self.assertEqual([t.get_text() for t in fig.axes[1].get_yticklabels()], ['тревога', 'котёл'])
            self.assertEqual(np.nanmax(fig.axes[0].lines[0].get_ydata()), 20)
            self.assertGreater(fig.axes[0].get_ylim()[1], 20)
            labels = [text.get_text() for text in fig.legends[0].get_texts()]
            for fragment in ('аварийный ремонт', 'плановый ремонт', 'резерв', 'котёл стоит', 'котёл работает', 'максимум за сутки'):
                self.assertTrue(any(fragment in label for label in labels))
        finally:
            plt.close(fig)
        # Cropping does not restart/reclassify a continuing alarm.
        full = M.alarm_series(res, ns)[1].copy()
        fig = plot_record(record, 'test', start=ix[201], end=ix[210]); plt.close(fig)
        pd.testing.assert_frame_equal(full, M.alarm_series(res, ns)[1])

    def test_summary_exposes_counts_and_hours_without_training(self):
        ns = metric_context(); ix = ns['t_ix']
        res = dict(score=pd.Series(.4, index=ix), thr=pd.Series(1., index=ix))
        row = M.evaluate(res, ns)[0]
        meta = dict(group='G', scenario='local_full', target='К-1', donors='К-1')
        table = pd.DataFrame([dict(**meta, view='test', category='основные', **{'источник оценок': 'fast'}, **row)])
        result = dict(status=[], plan=pd.DataFrame({'status': ['запланировано']}), tables={'main': table})
        outputs = []
        with patch.object(F, 'display', lambda html: outputs.append(html.data)):
            F.show_overview(result)
        for column in ('PR AUC случайный уровень', 'полезных эпизодов', 'фоновых эпизодов', 'нейтральных эпизодов', 'часов фона', 'рассчитано рабочих часов'):
            self.assertIn(column, ''.join(outputs))


if __name__ == '__main__':
    unittest.main()
