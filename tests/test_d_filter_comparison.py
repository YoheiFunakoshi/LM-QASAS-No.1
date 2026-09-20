"""Historical replay is separate from the production D-ignored policy; invented rows only."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from lmqasas.takara import parse_workbook
from xlsx_fixture import synthetic_row, synthetic_xlsx

spec=importlib.util.spec_from_file_location('d_comparison',Path(__file__).resolve().parents[1]/'scripts/compute_d_filter_comparison.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class HistoricalDComparisonTests(unittest.TestCase):
    def test_strict_v1_replay_preserves_order_and_rows_without_changing_current_clones(self):
        clones=[]
        for phase in ('Pre','Peak','Post'):
            rows=[synthetic_row(CDR3='CAGGW',D='x'),synthetic_row(CDR3='CASSW'),
                  synthetic_row(CDR3='CAGGW'),synthetic_row(CDR3='CASSW',D_function='ORF')]
            parsed,_=parse_workbook(synthetic_xlsx(rows),Path('synthetic.xlsx'),phase,'synthetic')
            clones.extend(parsed)
        audit={'samples':{p:{'input_policy':'takara-rg-hIGH20181210-v1'} for p in ('Pre','Peak','Post')}}
        result=module.legacy_baseline_from_bundle(SimpleNamespace(clones=clones),audit)
        self.assertEqual([c['cdr3'] for c in result],['CASSW','CAGGW']*3)
        self.assertEqual([c['source_rows'] for c in result],[[2],[3]]*3)
        self.assertEqual([c['source_rows'] for c in clones],[[1,3],[2,4]]*3)

    def test_current_or_missing_policy_is_not_a_historical_comparison(self):
        for policy in ('takara-rg-hIGH20181210-v2-ignore-d',None):
            audit={'samples':{p:{'input_policy':policy} for p in ('Pre','Peak','Post')}}
            with self.assertRaisesRegex(ValueError,'strict-v1'):
                module.legacy_baseline_from_bundle(SimpleNamespace(clones=[]),audit)
