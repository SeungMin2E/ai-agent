"""Failure regressions and ordering checks; no model downloads or real GPU required."""
import contextlib
from dataclasses import replace
import importlib.metadata
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, MagicMock, patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import runpod_setup as app


class Base(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = replace(app.Config(),workspace=str(self.root),state_dir=str(self.root/'state'),hf_home=str(self.root/'cache'))
        app.ensure_layout(self.config)
        redirect = contextlib.redirect_stdout(io.StringIO())
        redirect.__enter__()
        self.addCleanup(redirect.__exit__,None,None,None)


class Nvfp4WorkerTests(Base):
    def modules(self):
        tensor = MagicMock(shape=(128,128))
        torch = Mock()
        torch.ones.return_value = tensor
        torch.tensor.return_value = tensor
        torch.full.return_value = tensor
        torch.allclose.return_value = True
        flashinfer = types.ModuleType('flashinfer')
        flashinfer.SfLayout = types.SimpleNamespace(layout_128x4='128x4')
        flashinfer.nvfp4_quantize = Mock(return_value=(tensor,tensor))
        flashinfer.mm_fp4 = Mock(return_value=tensor)
        sampling = types.ModuleType('flashinfer.sampling')
        sampled = Mock(shape=(1,))
        sampled.item.return_value = 1
        sampling.sampling_from_probs = Mock(return_value=sampled)
        return {'torch':torch,'flashinfer':flashinfer,'flashinfer.sampling':sampling},torch,flashinfer

    def test_nvfp4_uses_actual_cutlass_path_and_checks_result(self):
        modules,torch,flashinfer = self.modules()
        with patch.dict(sys.modules,modules),patch('importlib.metadata.version',return_value='0.7.0.post1'),patch.object(app,'emit') as emit:
            app.nvfp4_worker()
        result = emit.call_args.args[0]
        self.assertTrue(result['ok'])
        self.assertEqual(result['backend'],'cutlass')
        self.assertEqual(flashinfer.mm_fp4.call_args.kwargs['backend'],'cutlass')
        self.assertFalse(flashinfer.nvfp4_quantize.call_args.kwargs['do_shuffle'])
        self.assertEqual(torch.cuda.synchronize.call_count,2)
        torch.allclose.assert_called_once()
        self.assertTrue(result['sampling_verified'])
        modules['flashinfer.sampling'].sampling_from_probs.assert_called_once()

    def test_compiler_killed_137_is_reported_at_nvfp4_stage(self):
        modules,torch,flashinfer = self.modules()
        flashinfer.mm_fp4.side_effect = RuntimeError('Ninja build failed. FAILED: [code=137]\nKilled')
        with patch.dict(sys.modules,modules),patch.object(app,'emit') as emit,self.assertRaises(SystemExit) as result:
            app.nvfp4_worker()
        self.assertEqual(result.exception.code,22)
        data = emit.call_args.args[0]
        self.assertFalse(data['ok'])
        self.assertEqual(data['stage'],'nvfp4_cutlass_build_and_run')
        self.assertIn('code=137',data['error'])

    def test_incorrect_result_fails_instead_of_claiming_preflight_success(self):
        modules,torch,flashinfer = self.modules()
        torch.allclose.return_value = False
        with patch.dict(sys.modules,modules),patch.object(app,'emit') as emit,self.assertRaises(SystemExit):
            app.nvfp4_worker()
        self.assertIn('incorrect result',emit.call_args.args[0]['error'])

    def test_sampling_compile_failure_is_detected_before_model_loading(self):
        modules,_,_ = self.modules()
        modules['flashinfer.sampling'].sampling_from_probs.side_effect = RuntimeError('sampling Ninja build failed')
        with patch.dict(sys.modules,modules),patch.object(app,'emit') as emit,self.assertRaises(SystemExit):
            app.nvfp4_worker()
        self.assertEqual(emit.call_args.args[0]['stage'],'sampling_build_and_run')
        self.assertFalse(emit.call_args.args[0]['ok'])

    def test_wrong_sampling_result_cannot_pass_preflight(self):
        modules,_,_ = self.modules()
        modules['flashinfer.sampling'].sampling_from_probs.return_value.item.return_value = 0
        with patch.dict(sys.modules,modules),patch.object(app,'emit') as emit,self.assertRaises(SystemExit):
            app.nvfp4_worker()
        self.assertIn('sampling verification',emit.call_args.args[0]['error'])


class PreflightTests(Base):
    def test_success_records_kernel_and_memory_before_after(self):
        data = {'ok':True,'backend':'cutlass'}
        with patch.object(app,'ensure_build_tools',return_value={'ninja':'verified'}),\
             patch.object(app,'run_logged',return_value=app.MARKER+json.dumps(data)) as run,\
             patch.object(app,'memory_status',side_effect=[{'available':10},{'available':9}]):
            result = app.run_preflight(self.config,'runtime-python')
        self.assertTrue(result['ok'])
        self.assertEqual(result['before'],{'available':10})
        self.assertEqual(result['after'],{'available':9})
        self.assertEqual(run.call_count,2)
        self.assertIn('_nvfp4',run.call_args.args[0])
        self.assertEqual(run.call_args.args[-1],self.config.jit_timeout)
        self.assertTrue(app.load_json(self.config.state/'logs/preflight.json')['ok'])

    def test_failed_compile_preserves_cgroup_oom_evidence_and_does_not_retry(self):
        before = {'memory.events':{'oom_kill':0}}
        after = {'memory.events':{'oom_kill':1}}
        with patch.object(app,'ensure_build_tools',return_value={}),\
             patch.object(app,'memory_status',side_effect=[before,after]),\
             patch.object(app,'run_logged',side_effect=app.SetupError('Ninja build failed, code=137')) as run:
            with self.assertRaisesRegex(app.SetupError,'before model loading') as exc:
                app.run_preflight(self.config,'runtime-python',check_dependencies=False)
        self.assertFalse(exc.exception.retryable)
        self.assertEqual(run.call_count,1)
        record = app.load_json(self.config.state/'logs/preflight.json')
        self.assertFalse(record['ok'])
        self.assertEqual(record['after']['memory.events']['oom_kill'],1)

    def test_false_kernel_result_blocks_progress(self):
        with patch.object(app,'ensure_build_tools',return_value={}),\
             patch.object(app,'run_logged',return_value=app.MARKER+'{"ok":false,"error":"bad result"}'):
            with self.assertRaisesRegex(app.SetupError,'kernel verification failed'):
                app.run_preflight(self.config,'runtime-python',check_dependencies=False)

    def test_interrupted_preflight_records_failure_and_preserves_cache(self):
        cache = Path(self.config.hf_home)/'keep'
        cache.write_bytes(b'cached weights')
        with patch.object(app,'ensure_build_tools',return_value={}),patch.object(app,'run_logged',side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                app.run_preflight(self.config,'runtime-python',check_dependencies=False)
        record = app.load_json(self.config.state/'logs/preflight.json')
        self.assertFalse(record['ok'])
        self.assertEqual(cache.read_bytes(),b'cached weights')

    def main_context(self):
        stack = contextlib.ExitStack()
        stack.enter_context(patch.object(app.Config,'from_env',return_value=self.config))
        stack.enter_context(patch.object(app.platform,'system',return_value='Linux'))
        stack.enter_context(patch.object(app.sys,'version_info',(3,12,3)))
        stack.enter_context(patch.object(app,'lock',return_value=contextlib.nullcontext()))
        stack.enter_context(patch.object(app,'existing_server',return_value=None))
        stack.enter_context(patch.object(app,'port_busy',return_value=False))
        stack.enter_context(patch.object(app.shutil,'which',return_value='/usr/bin/nvidia-smi'))
        stack.enter_context(patch.object(app.subprocess,'run',return_value=types.SimpleNamespace(returncode=0,stdout='RTX 5090',stderr='')))
        stack.enter_context(patch.object(app,'choose_runtime',return_value=('runtime-python',{'torch':'2.13.0','cuda':'13.0'})))
        stack.enter_context(patch.object(app.os,'umask',return_value=0))
        return stack

    def test_setup_does_not_download_or_start_after_preflight_failure(self):
        with self.main_context(),patch.object(app,'run_preflight',side_effect=app.SetupError('JIT failed')),\
             patch.object(app,'download') as download,patch.object(app,'start') as start:
            with self.assertRaisesRegex(app.SetupError,'JIT failed'): app.main(['setup'])
        download.assert_not_called()
        start.assert_not_called()

    def test_preflight_command_does_not_download_or_start_model(self):
        with self.main_context(),patch.object(app,'run_preflight',return_value={'ok':True}),\
             patch.object(app,'download') as download,patch.object(app,'start') as start:
            self.assertEqual(app.main(['preflight']),0)
        download.assert_not_called()
        start.assert_not_called()

    def test_dependency_mismatch_falls_back_without_modifying_system_runtime(self):
        runtime = {'ok':True,'vllm':'0.31.0','gpu':'RTX 5090'}
        with patch.object(app,'probe',return_value=runtime),patch.object(Path,'exists',return_value=True),\
             patch.object(app,'run_logged',side_effect=[app.SetupError('broken dependencies'),None]) as run:
            python,_ = app.choose_runtime(self.config)
        self.assertEqual(python,str(self.config.state/'venv/bin/python'))
        self.assertTrue(all(call.args[0][-2:]==['pip','check'] for call in run.call_args_list))


class DiagnosticsTests(Base):
    def test_original_error_is_kept_despite_long_trailing_wrapper_traceback(self):
        log = self.config.state/'logs/server.log'
        log.write_text("FileNotFoundError: No such file or directory: 'ninja'\n"+'wrapper frame\n'*3000+
                       'RuntimeError: Engine core initialization failed. See root cause above.\n',encoding='utf-8')
        error = app.startup_error(self.config,'Server exited',log)
        self.assertIn("'ninja'",str(error).split('Detected cause lines:')[-1])
        record = app.load_json(self.config.state/'logs/startup-failure.json')
        self.assertEqual(len(record['errors']),1)

    def test_compiler_kill_is_visible_separately_from_engine_failure(self):
        log = self.config.state/'logs/server.log'
        log.write_text('RuntimeError: Ninja build failed.\nFAILED: [code=137] fp4.cuda.o\nKilled\n'
                       'RuntimeError: Engine core initialization failed.\n',encoding='utf-8')
        lines = app.error_lines(log)
        self.assertIn('Killed',lines)
        self.assertFalse(any('Engine core initialization failed' in line for line in lines))

    def test_diagnose_redacts_tokens_and_tolerates_corrupt_diagnostic_file(self):
        secret = 'hf_'+'a'*30
        log = self.config.state/'logs/server.log'
        log.write_text('RuntimeError: access denied '+secret,encoding='utf-8')
        (self.config.state/'logs/preflight.json').write_text('{broken',encoding='utf-8')
        output = io.StringIO()
        with patch.dict(os.environ,{'HF_TOKEN':secret}),contextlib.redirect_stdout(output):
            app.diagnose(self.config)
        self.assertNotIn(secret,output.getvalue())
        data = json.loads(output.getvalue())
        self.assertIn('error',data['preflight'])
        self.assertEqual(len(data['script_sha256']),64)

    def test_host_and_container_memory_are_reported_separately(self):
        contents = {'meminfo':'MemTotal: 32000 kB\nMemAvailable: 8000 kB\n',
                    'memory.max':'16777216','memory.current':'8388608','memory.peak':'9000000',
                    'memory.events':'low 0\noom 2\noom_kill 1\n'}
        def read(path,*args,**kwargs):
            if path.name in contents: return contents[path.name]
            raise FileNotFoundError(path)
        with patch.object(Path,'read_text',autospec=True,side_effect=read):
            result = app.memory_status()
        self.assertEqual(result['MemAvailable_bytes'],8000*1024)
        self.assertEqual(result['memory.max'],16777216)
        self.assertEqual(result['memory.events']['oom_kill'],1)

    def test_malformed_snapshot_or_pid_is_handled_without_type_error(self):
        self.assertFalse(app.complete_snapshot(None))
        self.assertIsNone(app.process_identity(None))


class WarmupTests(Base):
    def test_cold_request_uses_saved_warmup_timeout_instead_of_120_seconds(self):
        config = replace(self.config,warmup_timeout=1800)
        app.save_json(config.state/'server.json',{'config':app.asdict(config)})
        with patch.object(sys,'argv',['runpod_setup.py','_smoke',str(config.state)]),\
             patch.object(app,'smoke') as smoke:
            self.assertEqual(app.entrypoint(),0)
        self.assertEqual(smoke.call_args.kwargs['timeout'],1800)

    def test_regular_test_keeps_short_timeout(self):
        response = {'choices':[{'message':{'content':'OK'}}]}
        with patch.object(app,'ready',return_value=True),patch.object(app,'request_json',return_value=response) as request:
            app.smoke(self.config)
        self.assertEqual(request.call_args.kwargs['timeout'],120)

    def test_warmup_uses_supervised_progress_worker_with_bounded_deadline(self):
        config = replace(self.config,warmup_timeout=240)
        with patch.object(app,'run_logged') as run:
            app.warmup(config,'runtime-python')
        self.assertEqual(run.call_args.args[0],['runtime-python',str(app.SCRIPT),'_smoke',str(config.state)])
        self.assertEqual(run.call_args.args[2:4],('warmup',250))

    def test_failed_warmup_is_not_swallowed(self):
        with patch.object(app,'run_logged',side_effect=app.SetupError('warmup timed out')):
            with self.assertRaisesRegex(app.SetupError,'warmup timed out'):
                app.warmup(self.config,'python')

    def test_warmup_timeout_configuration_rejects_invalid_values(self):
        paths = {'WORKSPACE':str(self.root),'STATE_DIR':str(self.config.state),'HF_HOME':self.config.hf_home}
        with patch.dict(os.environ,dict(paths,WARMUP_TIMEOUT='600')):
            self.assertEqual(app.Config.from_env().warmup_timeout,600)
        for value in ('0','-1','invalid'):
            with self.subTest(value=value),patch.dict(os.environ,dict(paths,WARMUP_TIMEOUT=value)):
                with self.assertRaises(app.SetupError):
                    app.Config.from_env()


if __name__=='__main__':
    unittest.main()
