"""Small synthetic fixtures exercise identity, missingness, and metric boundaries."""
from dataclasses import replace
import json
import sys
from types import ModuleType, SimpleNamespace
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stdout
from io import StringIO
from contracts import Record, ToolObservation, TurnResult
from evaluation import EvaluationCase, MeasurementClaim, evaluate_claims, iter_transcripts, legacy_nap, load_tool_csv, main, run_manifest, summarize

def measurement_csv(tmp_path, rows='HR00025.mat,0,,90.5,70.123\n'):
    path = tmp_path / 'measurements.csv'
    path.write_text('ecg_file_path,PR_Interval_ms,QTc_ms,QRS_Duration_ms,Heart_Rate\n' + rows)
    return path

def jsonl(tmp_path, rows):
    path = tmp_path / 'transcripts.jsonl'
    path.write_text('\n'.join((json.dumps(row) for row in rows)) + '\n')
    return path

def user(question='repeat this question'):
    return {'role': 'user', 'content': question}

def reply(answer='synthetic reply'):
    return {'role': 'assistant', 'action': 'response', 'content': answer}

def measurement():
    return {'role': 'assistant', 'action': 'call_measurement_tool', 'tool_output': {'heart_rate': 70.12, 'pr_interval': None, 'qrs_duration': 90, 'qtc_interval': None}}

def observation(**changes):
    result = ToolObservation(record_id='ptbxl:25', record_hash='waveform', call_id='call', evidence_id='evidence', tool='call_measurement_tool', version='synthetic', status='partial', findings={'heart_rate_bpm': 70.123, 'pr_interval_ms': None}, units={'heart_rate_bpm': 'bpm', 'pr_interval_ms': 'ms'}, missingness={'pr_interval_ms': 'unavailable_in_source'})
    return replace(result, **changes)


class FakeSession:
    def __init__(self, case):
        self.record = Record('synthetic.mat', case.record_id, 'waveform', 500, ('II',), ('mV',))
        self.model = SimpleNamespace(model='synthetic/model', max_tokens=128, timeout=5, api_key='must-not-log')
        self.max_responses = 8
        self.max_tool_calls = 4
        self.tool_timeout = 30
        self.request_timeout = 60
        self.history = [{'role': 'system', 'content': 'synthetic prompt'}]
        self.tools = []
        self.log = lambda *_args, **_kwargs: None
        self.close_count = 0

    def close(self):
        self.close_count += 1


class EvaluationTests(unittest.TestCase):

    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp_path = Path(directory.name)

    def test_csv_distinguishes_zero_missing_and_recorded_provenance(self):
        tmp_path = self.tmp_path
        outputs = load_tool_csv(measurement_csv(tmp_path))
        output = outputs['ptbxl:25']
        self.assertEqual(output.findings['pr_interval_ms'], 0)
        self.assertTrue(output.findings['qtc_bazett_ms'] is None)
        self.assertEqual(output.findings['q_s_peak_span_ms'], 90.5)
        self.assertTrue(output.findings['rr_interval_ms'] is None)
        self.assertEqual(output.missingness['qtc_bazett_ms'], 'unavailable_in_source')
        self.assertTrue('pr_interval_ms' not in output.missingness)
        evidence = output.as_observation('call')
        self.assertEqual(evidence.source, 'replay')
        self.assertEqual(evidence.record_hash, '')
        self.assertTrue(evidence.details['waveform_hash_available'] is False)
        self.assertEqual(evidence.details['source_fields']['q_s_peak_span_ms'], 'QRS_Duration_ms')
        self.assertEqual(evidence.units['heart_rate_bpm'], 'bpm')

    def test_csv_rejects_normalized_alias_collision(self):
        tmp_path = self.tmp_path
        path = measurement_csv(tmp_path, 'HR00025.mat,1,2,3,4\nHR25.mat,5,6,7,8\n')
        with self.assertRaisesRegex(ValueError, 'duplicate record alias'):
            load_tool_csv(path)

    def test_csv_rejects_bad_schema(self):
        tmp_path = self.tmp_path
        for text in ['ecg_file_path,Heart_Rate\nHR25.mat,70\n', 'ecg_file_path,PR_Interval_ms,QTc_ms,QRS_Duration_ms,Heart_Rate,Heart_Rate\nHR25.mat,1,2,3,4,5\n']:
            with self.subTest(text=text):
                path = tmp_path / 'bad.csv'
                path.write_text(text)
                with self.assertRaisesRegex(ValueError, 'schema'):
                    load_tool_csv(path)

    def test_csv_rejects_invalid_numbers(self):
        tmp_path = self.tmp_path
        for bad in ['inf', '-inf', 'True', 'not_a_number']:
            with self.subTest(bad=bad):
                path = measurement_csv(tmp_path, f'HR25.mat,1,2,3,{bad}\n')
                with self.assertRaises(ValueError):
                    load_tool_csv(path)

    def test_classification_no_positive_labels_differs_from_missing_join(self):
        tmp_path = self.tmp_path
        path = tmp_path / 'classes.csv'
        path.write_text('ecg_file_path,top_classes\nHR25.mat,"SR (99.01%), NORM (50.01%)"\nHR26.mat,None > 50%\n')
        outputs = load_tool_csv(path, 'classification')
        self.assertEqual(outputs['ptbxl:25'].labels[0]['label'], 'SR')
        self.assertAlmostEqual(outputs['ptbxl:25'].labels[0]['probability'], 0.9901)
        self.assertEqual(outputs['ptbxl:26'].labels, ())
        self.assertEqual(outputs['ptbxl:26'].status, 'success')

    def test_published_replicates_and_repeated_questions_preserve_identity(self):
        tmp_path = self.tmp_path
        dialogue = [user(), measurement(), reply(), user(), reply()]
        rows = [{'ecg_id': '00025', 'batch_idx': 9, 'dialogue': dialogue}, {'ecg_id': '25', 'batch_idx': 9, 'dialogue': [user(), measurement(), reply('different'), user(), reply()]}]
        path = jsonl(tmp_path, rows)
        first, second = list(iter_transcripts(path))
        self.assertTrue(first.record_id == second.record_id == 'ptbxl:25')
        self.assertEqual((first.replicate, second.replicate), (1, 2))
        self.assertTrue(first.case_id != second.case_id)
        self.assertTrue(first.turns[0].turn_id != first.turns[1].turn_id)
        self.assertEqual(first.turns[0].question, first.turns[1].question)
        self.assertEqual(first.turns[0].observations[0].source, 'replay')
        self.assertTrue(first.turns[0].reference_actions is None)
        self.assertEqual(first.source_metadata['batch_idx'], 9)
        self.assertFalse(first.source_metadata['history_declared_in_source'])
        self.assertTrue(legacy_nap(first)['accuracy'] is None)
        self.assertEqual([case.case_id for case in iter_transcripts(path)], [first.case_id, second.case_id])

    def test_different_dialogues_for_same_ecg_are_not_replicates(self):
        tmp_path = self.tmp_path
        rows = [{'ecg_id': '25', 'batch_idx': 9, 'dialogue': [user('first query'), reply()]}, {'ecg_id': '25', 'batch_idx': 9, 'dialogue': [user('different query'), reply()]}]
        self.assertEqual([case.replicate for case in iter_transcripts(jsonl(tmp_path, rows))], [1, 1])

    def test_published_classification_wrapper_is_supported(self):
        row = {
            'ecg_id': '25', 'batch_idx': 1,
            'dialogue': [user(), {
                'role': 'assistant', 'action': 'call_classification_tool',
                'tool_output': {'classification': ['SR (96.19%)']},
            }, reply()],
        }
        case = next(iter_transcripts(jsonl(self.tmp_path, [row])))
        labels = case.turns[0].observations[0].details['labels']
        self.assertEqual(labels[0]['label'], 'SR')
        self.assertAlmostEqual(labels[0]['probability'], 0.9619)

    def test_parenthesized_upstream_classification_label_is_supported(self):
        path = self.tmp_path / 'classes.csv'
        path.write_text('ecg_file_path,top_classes\nHR25.mat,PRC(S) (98.01%)\n')
        output = load_tool_csv(path, 'classification')['ptbxl:25']
        self.assertEqual(output.labels[0]['label'], 'PRC(S)')
        self.assertAlmostEqual(output.labels[0]['probability'], 0.9801)

    def test_current_schema_aligns_by_user_index_with_extra_tool_call(self):
        tmp_path = self.tmp_path
        classification = {'role': 'assistant', 'action': 'call_classification_tool', 'tool_output': "['SR (99.01%)']"}
        generated = [user('first'), classification, measurement(), reply(), user('second'), reply()]
        reference = [user('first'), classification, reply(), user('second'), reply()]
        row = {'sample_id': 0, 'ecg_file': '["HR00025.mat"]', 'generated_dialogue': generated, 'ground_truth_dialogue': reference, 'turns': [{'gt_action': 'response', 'gen_action': 'response', 'action_correct': False}]}
        case = next(iter_transcripts(jsonl(tmp_path, [row])))
        self.assertEqual(case.turns[0].actions, ('call_classification_tool', 'call_measurement_tool', 'response'))
        self.assertEqual(case.turns[0].reference_actions, ('call_classification_tool', 'response'))
        self.assertEqual(case.turns[1].question, 'second')
        self.assertEqual(case.turns[1].reference_actions, ('response',))
        self.assertEqual(len(case.turns[0].observations), 2)
        self.assertEqual(legacy_nap(case)['correct'], 1)

    def test_untrusted_python_text_is_not_executed(self):
        tmp_path = self.tmp_path
        marker = tmp_path / 'must_not_exist'
        row = {'ecg_id': '25', 'batch_idx': 0, 'dialogue': [user(), {'role': 'assistant', 'action': 'call_classification_tool', 'tool_output': f"__import__('pathlib').Path({str(marker)!r}).touch()"}]}
        with self.assertRaisesRegex(ValueError, 'invalid recorded label'):
            list(iter_transcripts(jsonl(tmp_path, [row])))
        self.assertTrue(not marker.exists())

    def test_transcript_rejects_multiple_record_selection(self):
        tmp_path = self.tmp_path
        row = {'ecg_file': '["HR25.mat", "HR26.mat"]', 'generated_dialogue': [user(), reply()]}
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            list(iter_transcripts(jsonl(tmp_path, [row])))

    def test_reference_query_mismatch_is_not_silently_aligned(self):
        tmp_path = self.tmp_path
        row = {'ecg_file': 'HR25.mat', 'generated_dialogue': [user('a'), reply()], 'ground_truth_dialogue': [user('b'), reply()]}
        with self.assertRaisesRegex(ValueError, 'reference user turns'):
            list(iter_transcripts(jsonl(tmp_path, [row])))

    def test_with_gt_is_explicitly_unsupported(self):
        tmp_path = self.tmp_path
        with self.assertRaisesRegex(ValueError, 'with_gt'):
            list(iter_transcripts(jsonl(tmp_path, []), history='with_gt'))

    def test_claims_require_explicit_rounding_and_matching_units(self):
        obs = observation()
        claims = [MeasurementClaim('evidence', 'heart_rate_bpm', 70.12, 'bpm', 2), MeasurementClaim('evidence', 'heart_rate_bpm', 70.12, 'bpm'), MeasurementClaim('evidence', 'heart_rate_bpm', 70.123, 'ms'), MeasurementClaim('evidence', 'pr_interval_ms', 0, 'ms')]
        result = evaluate_claims(claims, (obs,), record_id='ptbxl:25', requested_findings=('heart_rate_bpm', 'pr_interval_ms'))
        self.assertEqual((result['supported'], result['contradicted'], result['unverifiable']), (1, 2, 1))
        self.assertEqual(result['requested_available'], 1)
        self.assertEqual(result['coverage'], 1)
        self.assertEqual(result['scope'], 'explicit_numeric_claims_only')

    def test_wrong_record_and_unknown_evidence_do_not_support_claims(self):
        claim = MeasurementClaim('evidence', 'heart_rate_bpm', 70.123, 'bpm')
        self.assertEqual(evaluate_claims((claim,), (observation(record_id='ptbxl:26'),), record_id='ptbxl:25')['contradicted'], 1)
        result = evaluate_claims((replace(claim, evidence_id='missing'),), (observation(),))
        self.assertEqual(result['unverifiable'], 1)
        self.assertTrue(result['support_rate'] is None)

    def test_failed_tool_evidence_is_unverifiable_and_not_a_fidelity_zero(self):
        claim = MeasurementClaim('evidence', 'heart_rate_bpm', 70.123, 'bpm')
        result = evaluate_claims((claim,), (observation(status='failed'),))
        self.assertEqual(result['checkable'], 0)
        self.assertTrue(result['support_rate'] is None)
        self.assertEqual(result['unverifiable'], 1)

    def test_claim_rounding_rejects_invalid_precision(self):
        for decimals in [-1, 7, True, 2.5]:
            with self.subTest(decimals=decimals):
                with self.assertRaisesRegex(ValueError, 'decimals'):
                    evaluate_claims((MeasurementClaim('evidence', 'heart_rate_bpm', 70, 'bpm', decimals),), (observation(),))

    def test_manifest_uses_shared_runtime_and_stable_tool_ids(self):
        called = []
        obs = observation()
        module = ModuleType('eko')

        def fake_run_turn(session, question):
            called.append((session.record.record_id, question))
            session.log('model_response', model='synthetic/routed-model', usage={'tokens': 3})
            return TurnResult('random_runtime_id', 'success', answer='synthetic', evidence=(obs,))
        module.run_turn = fake_run_turn
        replacement = patch.dict(sys.modules, {'eko': module})
        replacement.start()
        self.addCleanup(replacement.stop)
        sessions = []

        def factory(case):
            session = FakeSession(case)
            sessions.append(session)
            return session

        cases = [EvaluationCase('z', 'ptbxl:25', ('same question', 'same question')), EvaluationCase('a', 'ptbxl:25', ('another question',))]
        report = run_manifest(cases, factory)
        self.assertEqual(called, [('ptbxl:25', 'another question'), ('ptbxl:25', 'same question'), ('ptbxl:25', 'same question')])
        self.assertEqual([row['turn_id'] for row in report['turns']], ['a/user:1', 'z/user:1', 'z/user:2'])
        first, second = report['turns'][1:]
        self.assertEqual(first['evidence'][0]['step_id'], second['evidence'][0]['step_id'])
        self.assertTrue(second['evidence'][0]['reused'] is True)
        self.assertTrue(report['summary']['claim_support_rate'] is None)
        self.assertEqual(report['summary']['completion_rate'], 1)
        self.assertEqual([session.close_count for session in sessions], [1, 1])
        self.assertEqual(report['manifest'][0]['record']['sha256'], 'waveform')
        self.assertEqual(report['manifest'][0]['configuration']['limits']['tool_timeout'], 30)
        self.assertEqual(report['turns'][0]['model_responses_trace'][0]['model'], 'synthetic/routed-model')
        self.assertNotIn('must-not-log', json.dumps(report))
        self.assertNotIn('path', report['manifest'][0]['record'])
        self.assertEqual(len(report['metadata']['source_sha256']['eko.py']), 64)
        self.assertEqual(report['summary']['live_observations'], 2)
        self.assertEqual(report['summary']['reused_observations'], 1)

    def test_manifest_closes_session_on_record_mismatch_and_runner_exception(self):
        case = EvaluationCase('case', 'ptbxl:25', ('question',))
        module = ModuleType('eko')
        module.run_turn = lambda *_: (_ for _ in ()).throw(RuntimeError('synthetic failure'))
        with patch.dict(sys.modules, {'eko': module}):
            session = FakeSession(case)
            with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
                run_manifest((case,), lambda _: session)
            self.assertEqual(session.close_count, 1)
            mismatch = FakeSession(replace(case, record_id='ptbxl:26'))
            with self.assertRaisesRegex(ValueError, 'session record'):
                run_manifest((case,), lambda _: mismatch)
            self.assertEqual(mismatch.close_count, 1)

    def test_manifest_rejects_duplicate_case_ids_and_teacher_forcing(self):
        module = ModuleType('eko')
        module.run_turn = lambda *_: None
        replacement = patch.dict(sys.modules, {'eko': module})
        replacement.start()
        self.addCleanup(replacement.stop)
        case = EvaluationCase('case', 'ptbxl:25', ('question',))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            run_manifest((case, case), lambda _: None)
        with self.assertRaisesRegex(ValueError, 'with_gt'):
            run_manifest((replace(case, history='with_gt'),), lambda _: None)

    def test_completion_includes_provider_failure_without_inventing_claim_score(self):
        report = summarize([{'turn_id': 'case/user:1', 'status': 'success', 'claim_checks': {'supported': 1}}, {'turn_id': 'case/user:2', 'status': 'failed', 'error_kind': 'provider'}])
        self.assertEqual(report['completion_rate'], 0.5)
        self.assertEqual(report['provider_errors'], 1)
        self.assertEqual(report['claim_support_rate'], 1)
        self.assertTrue(report['clinical_accuracy'] is None)
        self.assertTrue(report['prose_faithfulness'] is None)

    def test_native_provider_error_excludes_associated_claim_scores(self):
        native = json.loads(json.dumps(TurnResult(
            'native/user:1', 'failed', error='provider: empty response', elapsed=2.5, tool_calls=2,
        ).to_dict()))
        native['claim_checks'] = {'supported': 5, 'contradicted': 7, 'requested_available': 9}
        report = summarize((native,))
        self.assertEqual(report['provider_errors'], 1)
        self.assertEqual(report['supported'], 0)
        self.assertEqual(report['contradicted'], 0)
        self.assertEqual(report['excluded_claim_turns'], 1)
        self.assertIsNone(report['claim_support_rate'])
        self.assertEqual(report['attempted_tool_calls'], 2)
        self.assertEqual(report['total_elapsed_seconds'], 2.5)
        self.assertEqual(report['mean_elapsed_seconds'], 2.5)

    def test_explicit_judge_error_remains_separate_from_provider(self):
        report = summarize(({
            'turn_id': 'judge/user:1', 'status': 'success', 'error_kind': 'judge',
            'error': 'provider: judge endpoint unavailable', 'claim_checks': {'contradicted': 4},
        },))
        self.assertEqual(report['judge_errors'], 1)
        self.assertEqual(report['provider_errors'], 0)
        self.assertIsNone(report['claim_support_rate'])

    def test_native_summary_aggregates_latency_and_deduplicates_reused_evidence(self):
        obs = observation(source='replay')
        rows = [
            TurnResult('native/user:1', 'success', evidence=(obs,), elapsed=1, tool_calls=1).to_dict(),
            TurnResult('native/user:2', 'success', evidence=(obs,), elapsed=3, tool_calls=0).to_dict(),
        ]
        report = summarize(rows)
        self.assertEqual(report['total_elapsed_seconds'], 4)
        self.assertEqual(report['mean_elapsed_seconds'], 2)
        self.assertEqual(report['attempted_tool_calls'], 1)
        self.assertEqual(report['replay_observations'], 1)
        self.assertEqual(report['reused_observations'], 1)

    def test_summary_counts_failed_tools_even_with_completed_answer(self):
        report = summarize([{'turn_id': 'case/user:1', 'status': 'success', 'evidence': [observation(status='failed').to_dict()]}])
        self.assertEqual(report['completed_answers'], 1)
        self.assertEqual(report['failed_tool_observations'], 1)
        self.assertTrue(report['claim_support_rate'] is None)

    def test_report_rejects_duplicate_turn_ids(self):
        turn = {'turn_id': 'case/user:1', 'status': 'success'}
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            summarize((turn, turn))

    def test_cli_inspects_synthetic_replay_and_marks_no_nap_labels(self):
        tmp_path = self.tmp_path
        path = jsonl(tmp_path, [{'ecg_id': '25', 'batch_idx': 0, 'dialogue': [user(), measurement(), reply()]}])
        captured = StringIO()
        with redirect_stdout(captured):
            self.assertEqual(main(['transcripts', str(path)]), 0)
        report = json.loads(captured.getvalue())
        self.assertEqual(report['source'], 'replay')
        self.assertEqual(report['source_rows'], 1)
        self.assertEqual(report['rows_with_nap_labels'], 0)

    def test_cli_reports_native_turn_results_with_provider_error(self):
        result = TurnResult('native/user:1', 'failed', error='provider: timeout', elapsed=1.5)
        path = jsonl(self.tmp_path, [result.to_dict()])
        captured = StringIO()
        with redirect_stdout(captured):
            self.assertEqual(main(['report', str(path)]), 0)
        report = json.loads(captured.getvalue())
        self.assertEqual(report['provider_errors'], 1)
        self.assertEqual(report['completion_rate'], 0)
        self.assertIsNone(report['claim_support_rate'])
        self.assertEqual(report['total_elapsed_seconds'], 1.5)
if __name__ == '__main__':
    unittest.main()
