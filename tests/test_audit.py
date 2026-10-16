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
        return {'torch':torch,'flashinfer':flashinfer},torch,flashinfer

    def test_nvfp4_uses_actual_cutlass_path_and_checks_result(self):
        modules,torch,flashinfer = self.modules()
        with patch.dict(sys.modules,modules),patch('importlib.metadata.version',return_value='0.7.0.post1'),patch.object(app,'emit') as emit:
            app.nvfp4_worker()
        result = emit.call_args.args[0]
        self.assertTrue(result['ok'])
        self.assertEqual(result['backend'],'cutlass')
        self.assertEqual(flashinfer.mm_fp4.call_args.kwargs['backend'],'cutlass')
        self.assertFalse(flashinfer.nvfp4_quantize.call_args.kwargs['do_shuffle'])
        torch.cuda.synchronize.assert_called_once()
        torch.allclose.assert_called_once()

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


if __name__=='__main__':
    unittest.main()
