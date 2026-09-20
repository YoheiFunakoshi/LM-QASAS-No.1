"""Synthetic report tests; no source workbook or research sequence is copied."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lmqasas.inputs import InputValidationError, load_three_inputs
from lmqasas.takara import check_workbook, parse_workbook
from lmqasas.pipeline import run_analysis, reselect
from test_pipeline import SyntheticEncoder
from xlsx_fixture import synthetic_xlsx, synthetic_row


class TakaraTests(unittest.TestCase):
    def parse(self, rows=None, **options):
        return parse_workbook(synthetic_xlsx(rows, **options), Path('synthetic.xlsx'), 'Pre', 'synthetic')

    def test_schema_and_second_sheet(self):
        audit=check_workbook(synthetic_xlsx())
        self.assertEqual(audit['source_sheet'], 'Back_data')
        self.assertEqual(audit['source_sheet_index'], 2)
        self.assertEqual(audit['input_format'], 'takara_rg_xlsx')
        for names in [('Back_data','PRINT_hIGH'),('PRINT_hIGH','Sheet2')]:
            with self.subTest(names=names), self.assertRaises(InputValidationError):
                check_workbook(synthetic_xlsx(sheet_names=names))

    def test_dimensions_and_row_one_are_not_trusted_as_header(self):
        rows=[synthetic_row(count=i+1) for i in range(240)]
        rows[-1]=synthetic_row(CDR3='CAGGW',count=240)
        clones,audit=self.parse(rows,reported_dimension='A1:CP2')
        self.assertEqual(audit['total_rows'],240)
        self.assertEqual(audit['last_data_row'],240)
        self.assertEqual(len(clones),2)
        self.assertEqual(clones[0]['source_rows'],list(range(1,240)))
        self.assertEqual(clones[1]['source_rows'],[240])

    def test_top_fifty_block_is_not_source(self):
        clones,audit=self.parse(overrides={'X1':'NOT_A_SEQUENCE','Z1':99999,'AA1':('formula','Z1/C6*100')})
        self.assertEqual(clones[0]['cdr3'],'CASSW')
        self.assertEqual(audit['total_rows'],1)

    def test_unknown_layout_and_version_fail(self):
        for settings in ({'overrides':{'F1':'different layout'}},
                         {'front_overrides':{'F5':'Sheet ver. other'}},
                         {'front_overrides':{'F1':'Unrelated workbook'}}):
            with self.subTest(settings=settings), self.assertRaises(InputValidationError):
                check_workbook(synthetic_xlsx(**settings))

    def test_summary_reconciles_assigned_inframe_and_vendor_key(self):
        rows=[synthetic_row(), synthetic_row(count=20), synthetic_row(D='IGHD2-2*01'),
              synthetic_row(CDR3='CQQQW',frame='out-of-frame')]
        clones,audit=self.parse(rows)
        self.assertEqual(audit['summary']['in_frame_unique'],2)
        # Vendor key includes D; our clone key does not.
        self.assertEqual(len(clones),1)
        self.assertEqual(clones[0]['source_rows'],[1,2,3])
        for address in ('C5','C6','C7'):
            with self.subTest(address=address), self.assertRaises(InputValidationError):
                self.parse(rows,overrides={address:99})

    def test_summary_formula_or_invalid_integer_fails(self):
        for value in [('formula','1+1'), '10', -1, 1.5]:
            with self.subTest(value=value), self.assertRaises(InputValidationError):
                check_workbook(synthetic_xlsx(overrides={'C5':value}))

    def test_raw_cells_sheet_rows_and_boundaries_preserved(self):
        row=synthetic_row(CDR3='AREFY',C='IGHG2*01')
        clones,audit=self.parse([row])
        self.assertEqual(clones[0]['cdr3'],'AREFY')
        self.assertEqual(clones[0]['raw_records'],[row])
        self.assertEqual(clones[0]['source_sheet'],'Back_data')
        self.assertEqual(clones[0]['source_row'],1)
        self.assertFalse(audit['nt_length_verified'])
        self.assertFalse(audit['full_vdj_functionality_verified'])
        self.assertNotIn('Type',clones[0]['raw_records'][0])
        self.assertNotIn('NTlength',clones[0]['raw_records'][0])

    def test_comma_gene_alternatives_keep_sets_and_slash_gene(self):
        clones,_=self.parse([synthetic_row(V='IGHV1-69D*01,IGHV1-69*02',J='IGHJ6*01,IGHJ4*02'),
                             synthetic_row(V='IGHV1-69*01,IGHV1-69D*02',J='IGHJ4*01,IGHJ6*02'),
                             synthetic_row(V='IGHV3-30/OR16-4*01')])
        self.assertEqual(len(clones),2)
        self.assertEqual(clones[0]['v_gene'],'IGHV1-69//IGHV1-69D')
        self.assertEqual(clones[0]['j_gene'],'IGHJ4//IGHJ6')
        self.assertEqual(clones[0]['source_rows'],[1,2])

    def test_same_class_alleles_combine_mixed_class_is_excluded(self):
        clones,audit=self.parse([synthetic_row(C='IGHG1*02,IGHG1*01'),
                                synthetic_row(C='IGHG1*01,IGHG2*01'),
                                synthetic_row(C='IGHG2*01')])
        self.assertEqual([c['isotype'] for c in clones],['IGHG1','IGHG2'])
        self.assertEqual(audit['excluded_rows'],1)
        self.assertIn('cseg_unmapped_or_ambiguous',audit['exclusions'][0]['reasons'])

    def test_d_copy_suffixes_are_accepted_without_modifying_source(self):
        rows=[synthetic_row(D='IGHD2/OR15-2b*01,IGHD3/OR15-3a*01'),
              synthetic_row(D='IGHD2/OR15-2b*02')]
        clones,audit=self.parse(rows)
        self.assertEqual(audit['accepted_rows'],2)
        self.assertEqual(len(clones),1)
        self.assertEqual(clones[0]['raw_records'],rows)

    def test_bad_rows_are_excluded_with_all_reasons(self):
        cases=[({'frame':'out-of-frame'},'frame_not_in_frame'),
               ({'V_function':'P'},'v_function_not_F'),
               ({'D_function':'ORF'},'d_function_not_F'),
               ({'J_function':'x'},'j_function_not_F'),
               ({'CDR3':'CAS*W'},'cdr3_noncanonical'),
               ({'CDR3':'CASS'},'cdr3_too_short'),
               ({'C':'IGHGP*01'},'cseg_unmapped_or_ambiguous'),
               ({'V':'IGHV1-2*01,'},'v_annotation_invalid'),
               ({'D':'x'},'d_annotation_invalid'),
               ({'J':'x'},'j_annotation_invalid')]
        for changes,reason in cases:
            with self.subTest(reason=reason):
                clones,audit=self.parse([synthetic_row(),synthetic_row(**changes)])
                self.assertEqual(audit['accepted_rows'],1)
                self.assertIn(reason,audit['exclusions'][0]['reasons'])

    def test_count_is_metadata_not_clone_weight(self):
        clones,audit=self.parse([synthetic_row(count=100000),synthetic_row(count=1)])
        self.assertEqual(len(clones),1)
        self.assertEqual(audit['merged_rows'],1)
        self.assertEqual([r['count'] for r in clones[0]['raw_records']],[100000,1])

    def test_invalid_counts_and_data_formulas_fail(self):
        for value in (-1, 2.5, 'private-input', ('formula','1+2')):
            with self.subTest(value=value), self.assertRaises(InputValidationError) as caught:
                self.parse(overrides={'Q1':value})
            self.assertNotIn('private-input',str(caught.exception))
        with self.assertRaises(InputValidationError):
            self.parse(overrides={'O1':('formula','"CASSW"')})

    def test_invalid_archive_and_size_limits(self):
        with self.assertRaises(InputValidationError):
            check_workbook(b'not-excel private-content')
        with patch('lmqasas.takara.MAX_EXPANDED_BYTES',1), self.assertRaises(InputValidationError):
            check_workbook(synthetic_xlsx())
        with patch('lmqasas.takara.MAX_ROWS',1), self.assertRaises(InputValidationError):
            self.parse([synthetic_row(),synthetic_row()])

    def test_no_eligible_rows_rejected(self):
        with self.assertRaisesRegex(InputValidationError,'no rows remain'):
            self.parse([synthetic_row(frame='out-of-frame')])

    def test_three_file_import_provenance_and_no_original_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            paths={p:root/(p+'.xlsx') for p in ('Pre','Peak','Post')}
            before={}
            for p,path in paths.items():
                path.write_bytes(synthetic_xlsx())
                before[p]=path.read_bytes()
            bundle=load_three_inputs(paths,'synthetic')
            self.assertEqual(len(bundle.clones),3)
            self.assertEqual(bundle.audit['policies']['input_format'],'takara_rg_xlsx')
            self.assertEqual(bundle.input_hashes,{k:hashlib.sha256(v).hexdigest() for k,v in before.items()})
            self.assertEqual(before,{p:path.read_bytes() for p,path in paths.items()})
            with patch('lmqasas.inputs._sha256_path',return_value='modified'),self.assertRaisesRegex(InputValidationError,'changed during reading'):
                load_three_inputs(paths,'synthetic')
            csvpath=root/'Post.csv'
            csvpath.write_bytes(b'unused')
            with self.assertRaisesRegex(InputValidationError,'same supported format'):
                load_three_inputs({**paths,'Post':csvpath},'synthetic')

    def test_xlsx_pipeline_selection_and_rerank(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            inputs=root/'input'; inputs.mkdir()
            paths={p:inputs/(p+'.xlsx') for p in ('Pre','Peak','Post')}
            records={'Pre':[synthetic_row()],
                     'Peak':[synthetic_row(),synthetic_row(CDR3='CAGGW'),synthetic_row(CDR3='CARFW')],
                     'Post':[synthetic_row()]}
            for phase,path in paths.items():path.write_bytes(synthetic_xlsx(records[phase]))
            encoder=SyntheticEncoder()
            result=run_analysis(paths,'synthetic',root/'unused_model',root/'outputs',
                                top_n=2,n_clusters=1,n_init=1,threads=1,encoder=encoder)
            meta=json.loads((result/'run_metadata.json').read_text())
            self.assertEqual(meta['status'],'completed')
            self.assertEqual(meta['input_format'],'takara_rg_xlsx')
            self.assertEqual(meta['selection']['returned'],2)
            # Three Peak observations / (one Pre+epsilon) twice => 3.
            with (result/'selection_initial/candidates.csv').open(encoding='utf-8-sig') as f:
                selected=list(csv.DictReader(f))
            self.assertEqual({float(row['score']) for row in selected},{3.0})
            reranked=reselect(result,1)
            self.assertEqual(json.loads((reranked/'selection_summary.json').read_text())['returned'],1)
            self.assertEqual(len(encoder.calls),1)


if __name__=='__main__':unittest.main()
