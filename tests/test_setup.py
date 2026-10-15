import contextlib
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch, Mock

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

    def snapshot(self):
        path = self.root/'snapshot'
        path.mkdir(exist_ok=True)
        for name in ('config.json','tokenizer_config.json','tokenizer.json'):
            (path/name).write_text('{}',encoding='utf-8')
        (path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'layer.weight':'model-1.safetensors'}}))
        (path/'model-1.safetensors').write_bytes(b'fake-weight')
        return path


class ConfigurationTests(Base):
    def test_defaults_match_user_profile(self):
        c = app.Config()
        self.assertEqual((c.model,c.max_model_len,c.max_num_seqs,c.port,c.gpu_memory_utilization),
                         ('Inferact/Qwen3.8-27B-NVFP4',8192,1,8000,0.9))

    def test_invalid_settings_are_rejected(self):
        cases = ({'port':0},{'port':65536},{'max_model_len':0},{'max_num_seqs':-1},
                 {'gpu_memory_utilization':float('nan')},{'gpu_memory_utilization':1.0},
                 {'host':'0.0.0.0'},{'host':'example.com'},{'model':'bad; command'},
                 {'revision':''},{'vllm_version':'latest'},{'retries':0},
                 {'startup_timeout':-1},{'state_dir':str(self.root)})
        for changes in cases:
            with self.subTest(changes=changes),self.assertRaises(app.SetupError):
                replace(self.config,**changes).validate()

    def test_bad_numeric_environment_is_actionable(self):
        with patch.dict(os.environ,{'PORT':'invalid'}),self.assertRaisesRegex(app.SetupError,'environment'):
            app.Config.from_env()

    def test_gpu_visibility_is_preserved(self):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'GPU-uuid-123'}):
            self.assertEqual(self.config.environment()['CUDA_VISIBLE_DEVICES'],'GPU-uuid-123')

    def test_plan_needs_no_gpu(self):
        with patch.dict(os.environ,{'WORKSPACE':str(self.root),'STATE_DIR':str(self.root/'state')}):
            self.assertEqual(app.main(['plan']),0)

    def test_server_command_uses_safe_argv_and_required_flags(self):
        command = app.build_server_command(self.config,'/runtime/python',str(self.root/'a model'))
        self.assertIsInstance(command,list)
        self.assertIn('--enforce-eager',command)
        self.assertEqual(command[command.index('--kv-cache-dtype')+1],'fp8')
        self.assertEqual(command[command.index('--reasoning-parser')+1],'qwen3')
        self.assertIn(str(self.root/'a model'),command)

    def test_ipv6_url_is_bracketed(self):
        self.assertEqual(replace(self.config,host='::1').url,'http://[::1]:8000')


class FailureHandlingTests(Base):
    def test_transient_failure_retries_then_succeeds(self):
        operation = Mock(side_effect=[app.SetupError('network',retryable=True),app.SetupError('network',retryable=True),'OK'])
        sleep = Mock()
        self.assertEqual(app.retry(operation,3,sleep),'OK')
        self.assertEqual(operation.call_count,3)
        self.assertEqual(sleep.call_count,2)

    def test_permanent_failure_is_not_retried(self):
        operation = Mock(side_effect=app.SetupError('401'))
        with self.assertRaises(app.SetupError): app.retry(operation,3,Mock())
        self.assertEqual(operation.call_count,1)

    def test_retry_is_bounded(self):
        operation = Mock(side_effect=app.SetupError('network',retryable=True))
        with self.assertRaises(app.SetupError): app.retry(operation,2,Mock())
        self.assertEqual(operation.call_count,2)

    def test_low_disk_stops_before_download(self):
        with patch.object(app.shutil,'disk_usage',return_value=types.SimpleNamespace(free=1)),self.assertRaisesRegex(app.SetupError,'Not enough space'):
            app.require_space(self.root,app.GIB,'model')

    def test_corrupt_state_is_not_silently_overwritten(self):
        path = self.root/'state.json'
        for text in ('{broken','[]','null'):
            path.write_text(text)
            with self.assertRaises(app.SetupError): app.load_json(path,{})
            self.assertEqual(path.read_text(),text)

    def test_state_write_is_atomic(self):
        path = self.root/'state.json'
        app.save_json(path,{'first':1}); app.save_json(path,{'second':2})
        self.assertEqual(app.load_json(path),{'second':2})
        self.assertEqual(list(self.root.glob('*.tmp')),[])

    def test_secrets_are_redacted(self):
        with patch.dict(os.environ,{'HF_TOKEN':'secret-token','VLLM_API_KEY':'secret-api'}):
            result = app.redact('secret-token secret-api hf_1234567890abcdef')
        self.assertNotIn('secret',result)
        self.assertEqual(result.count('[REDACTED]'),3)

    def test_helper_result_handles_library_stdout(self):
        self.assertEqual(app.extract_result('library log\n'+app.MARKER+'{"ok":true}\n'),{'ok':True})
        for output in ('nothing',app.MARKER+'{broken'):
            with self.assertRaises(app.SetupError): app.extract_result(output)

    def test_readiness_requires_requested_model(self):
        cases = [({},False),([],False),({'data':None},False),({'data':[None]},False),
                 ({'data':[{'id':'another/model'}]},False),({'data':[{'id':self.config.model}]},True)]
        for value,expected in cases:
            with self.subTest(value=value),patch.object(app,'request_json',return_value=value):
                self.assertEqual(app.ready(self.config),expected)

    def test_readiness_handles_connection_failure(self):
        with patch.object(app,'request_json',side_effect=OSError('refused')):
            self.assertFalse(app.ready(self.config))

    def test_inference_rejects_invalid_or_empty_response(self):
        for value in ({},{'choices':[]},{'choices':[{'message':{'content':''}}]}):
            with patch.object(app,'ready',return_value=True),patch.object(app,'request_json',return_value=value),self.assertRaises(app.SetupError):
                app.smoke(self.config)


class CacheTests(Base):
    def test_complete_snapshot_reused_without_network(self):
        record = {'model':self.config.model,'requested_revision':'main','commit':'a'*40,'snapshot':str(self.snapshot())}
        app.save_json(self.config.state/'model.json',record)
        with patch.object(app,'run_logged') as worker:
            self.assertEqual(app.download(self.config,'python'),record)
        worker.assert_not_called()

    def test_missing_and_empty_shards_are_incomplete(self):
        path = self.snapshot()
        self.assertTrue(app.complete_snapshot(path))
        (path/'model-1.safetensors').write_bytes(b'')
        self.assertFalse(app.complete_snapshot(path))
        (path/'model-1.safetensors').unlink()
        self.assertFalse(app.complete_snapshot(path))

    def test_invalid_weight_index_is_incomplete(self):
        path = self.snapshot()
        for value in ({'weight_map':[]},{'weight_map':{}},{'weight_map':{'x':'../outside'}},[]):
            (path/'model.safetensors.index.json').write_text(json.dumps(value))
            self.assertFalse(app.complete_snapshot(path))

    def test_gated_or_missing_model_is_permanent(self):
        error = RuntimeError('access denied'); error.response = types.SimpleNamespace(status_code=403)
        fake = Mock(); fake.HfApi.return_value.model_info.side_effect = error
        with patch.dict(sys.modules,{'huggingface_hub':fake}),self.assertRaises(SystemExit) as result:
            app.download_worker('owner/model','main',str(self.root/'cache'))
        self.assertEqual(result.exception.code,20)

    def test_download_network_failure_is_retryable(self):
        fake = Mock(); fake.HfApi.return_value.model_info.side_effect = TimeoutError('network')
        with patch.dict(sys.modules,{'huggingface_hub':fake}),self.assertRaises(SystemExit) as result:
            app.download_worker('owner/model','main',str(self.root/'cache'))
        self.assertEqual(result.exception.code,21)

    def test_runtime_reuse_does_not_install_packages(self):
        result = {'ok':True,'vllm':'0.31.0','gpu':'RTX 5090'}
        with patch.object(app,'probe',return_value=result),patch.object(app,'run_logged') as install:
            python,_ = app.choose_runtime(self.config)
        self.assertEqual(python,sys.executable)
        install.assert_not_called()

    def test_wrong_hardware_does_not_trigger_reinstall(self):
        with patch.object(app,'probe',return_value={'ok':False,'hardware_error':True,'error':'wrong GPU'}),patch.object(app,'run_logged') as install:
            with self.assertRaisesRegex(app.SetupError,'wrong GPU'): app.choose_runtime(self.config)
        install.assert_not_called()


class StartSafetyTests(Base):
    def test_port_conflict_does_not_launch(self):
        with socket.socket() as occupied:
            occupied.bind(('127.0.0.1',0)); occupied.listen()
            config = replace(self.config,port=occupied.getsockname()[1])
            with patch.object(app.subprocess,'Popen') as launch,self.assertRaisesRegex(app.SetupError,'occupied'):
                app.start(config,'python',{'snapshot':'snapshot'})
            launch.assert_not_called()

    def test_low_gpu_memory_does_not_launch(self):
        result = {'ok':True,'free_bytes':2*app.GIB,'total_bytes':32*app.GIB}
        with patch.object(app,'port_busy',return_value=False),patch.object(app,'probe',return_value=result),patch.object(app.subprocess,'Popen') as launch:
            with self.assertRaisesRegex(app.SetupError,'GPU busy'): app.start(self.config,'python',{'snapshot':'snapshot'})
        launch.assert_not_called()

    def test_pid_reuse_is_not_considered_managed(self):
        with patch.object(app,'process_identity',return_value={'pid':123,'start_ticks':'new','boot_id':'new'}):
            self.assertFalse(app.same_process({'pid':123,'start_ticks':'old','boot_id':'old'}))

    def test_stale_pid_is_never_signalled(self):
        app.save_json(self.config.state/'server.json',{'pid':123,'start_ticks':'old','boot_id':'old'})
        with patch.object(app,'same_process',return_value=False),patch.object(app.os,'killpg',create=True) as kill:
            app.stop(self.config)
        kill.assert_not_called()


@unittest.skipUnless(sys.platform.startswith('linux'),'requires Linux process groups and /proc')
class LinuxIntegrationTests(Base):
    def free_port(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0)); return sock.getsockname()[1]

    def test_shell_launcher_and_syntax(self):
        shell = app.SCRIPT.with_name('setup.sh')
        subprocess.run(['bash','-n',str(shell)],check=True)
        result = subprocess.run(['bash',str(shell),'plan'],capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(result.stdout)['port'],8000)

    def test_lock_blocks_concurrent_setup(self):
        with app.lock(self.config.state/'setup.lock'),self.assertRaisesRegex(app.SetupError,'Another setup'):
            with app.lock(self.config.state/'setup.lock'): pass

    def test_supervised_start_duplicate_and_stop(self):
        fake = self.root/'fake_server.py'
        fake.write_text('''import http.server,json,sys
class Handler(http.server.BaseHTTPRequestHandler):
 def do_GET(self):
  self.send_response(200);self.end_headers();self.wfile.write(json.dumps({"data":[{"id":sys.argv[2]}]}).encode())
 def log_message(self,*args): pass
http.server.HTTPServer(("127.0.0.1",int(sys.argv[1])),Handler).serve_forever()
''')
        config = replace(self.config,port=self.free_port(),startup_timeout=10)
        gpu = {'ok':True,'free_bytes':31*app.GIB,'total_bytes':32*app.GIB}
        self.addCleanup(app.stop,config)
        with patch.object(app,'probe',return_value=gpu),patch.object(app,'build_server_command',return_value=[sys.executable,str(fake),str(config.port),config.model]):
            app.start(config,sys.executable,{'snapshot':str(self.snapshot())})
            first = app.load_json(config.state/'server.json')
            self.assertTrue(app.ready(config))
            app.start(config,sys.executable,{'snapshot':'unused'})
            self.assertEqual(app.load_json(config.state/'server.json')['pid'],first['pid'])
        app.stop(config)
        self.assertFalse(app.same_process(first))
        self.assertTrue(app.complete_snapshot(self.root/'snapshot'))

    def test_failed_start_cleans_managed_process(self):
        config = replace(self.config,port=self.free_port(),startup_timeout=2)
        gpu = {'ok':True,'free_bytes':31*app.GIB,'total_bytes':32*app.GIB}
        with patch.object(app,'probe',return_value=gpu),patch.object(app,'build_server_command',return_value=[sys.executable,'-c','raise SystemExit(3)']):
            with self.assertRaises(app.SetupError): app.start(config,sys.executable,{'snapshot':'unused'})
        self.assertFalse((config.state/'server.json').exists())
        self.assertIn('exited with code 3',app.tail(config.state/'logs/server.log'))

    def test_timed_out_child_is_stopped(self):
        with self.assertRaisesRegex(app.SetupError,'timed out'):
            app.run_logged([sys.executable,'-c','import time; time.sleep(60)'],self.config,'timeout-test',1)


if __name__=='__main__':
    unittest.main()
