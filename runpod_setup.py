#!/usr/bin/env python3
"""RunPod Qwen bootstrap; controller requires only the Python standard library."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
import ipaddress
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request

GIB = 1024**3
SCRIPT = Path(__file__).resolve()
MARKER = 'AI_AGENT_JSON='
MANAGED_CHILDREN = {}


class SetupError(Exception):
    def __init__(self, message, *, retryable=False):
        super().__init__(message)
        self.retryable = retryable


def redact(text):
    for key in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'VLLM_API_KEY', 'GH_TOKEN', 'GITHUB_TOKEN'):
        value = os.environ.get(key)
        if value:
            text = text.replace(value, '[REDACTED]')
    return re.sub(r'\bhf_[A-Za-z0-9]{12,}\b', '[REDACTED]', text)


def say(message):
    print(f'[{datetime.now().astimezone().isoformat(timespec="seconds")}] {redact(str(message))}', flush=True)


def load_json(path, default=None):
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(value,dict):
            raise ValueError('expected a JSON object')
        return value
    except FileNotFoundError:
        return default
    except (OSError, ValueError) as exc:
        raise SetupError(f'Cannot read {path}: {exc}. Preserve the file and inspect it before retrying.') from exc


def save_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    try:
        with tmp.open('w', encoding='utf-8', newline='\n') as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        tmp.chmod(0o600)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def tail(path, limit=16000):
    try:
        with Path(path).open('rb') as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell()-limit))
            return redact(handle.read().decode('utf-8', errors='replace'))
    except FileNotFoundError:
        return '(no log yet)'


def retry(operation, attempts=3, sleep=time.sleep):
    for attempt in range(1, attempts+1):
        try:
            return operation()
        except SetupError as exc:
            if not exc.retryable or attempt == attempts:
                raise
            delay = min(2**attempt, 20)
            say(f'Temporary failure ({attempt}/{attempts}): {exc}. Retrying in {delay}s.')
            sleep(delay)


@dataclass(frozen=True)
class Config:
    workspace: str = '/workspace'
    state_dir: str = '/workspace/ai-agent-state'
    hf_home: str = '/workspace/hf-cache'
    model: str = 'Inferact/Qwen3.8-27B-NVFP4'
    revision: str = 'main'
    vllm_version: str = '0.31.0'
    host: str = '127.0.0.1'
    port: int = 8000
    max_model_len: int = 8192
    max_num_seqs: int = 1
    gpu_memory_utilization: float = 0.90
    startup_timeout: int = 900
    install_timeout: int = 3600
    download_timeout: int = 10800
    retries: int = 3

    @classmethod
    def from_env(cls):
        defaults = cls()
        workspace = os.environ.get('WORKSPACE', defaults.workspace)
        values = {'workspace':workspace,
                  'state_dir':os.environ.get('STATE_DIR', str(Path(workspace)/'ai-agent-state')),
                  'hf_home':os.environ.get('HF_HOME', str(Path(workspace)/'hf-cache'))}
        names = {'model':'MODEL_ID', 'revision':'MODEL_REVISION', 'vllm_version':'VLLM_VERSION',
                 'host':'HOST', 'port':'PORT', 'max_model_len':'MAX_MODEL_LEN', 'max_num_seqs':'MAX_NUM_SEQS',
                 'gpu_memory_utilization':'GPU_MEMORY_UTILIZATION', 'startup_timeout':'STARTUP_TIMEOUT',
                 'install_timeout':'INSTALL_TIMEOUT', 'download_timeout':'DOWNLOAD_TIMEOUT', 'retries':'RETRIES'}
        try:
            for field, name in names.items():
                default = getattr(defaults, field)
                values[field] = type(default)(os.environ.get(name, default))
        except ValueError as exc:
            raise SetupError(f'Invalid environment setting: {exc}') from exc
        result = cls(**values)
        result.validate()
        return result

    def validate(self):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', self.model):
            raise SetupError('MODEL_ID must be a Hugging Face owner/model identifier.')
        if not re.fullmatch(r'\d+\.\d+\.\d+[a-zA-Z0-9.+-]*', self.vllm_version):
            raise SetupError('VLLM_VERSION must be a concrete version, e.g. 0.31.0.')
        if not self.revision or any(c.isspace() for c in self.revision):
            raise SetupError('MODEL_REVISION must be a non-empty branch, tag or commit.')
        try:
            local = ipaddress.ip_address(self.host).is_loopback
        except ValueError:
            local = False
        if not local:
            raise SetupError('HOST must be a loopback IP. Use SSH port forwarding to access this server.')
        if not 1 <= self.port <= 65535:
            raise SetupError('PORT must be between 1 and 65535.')
        if not 0 < self.gpu_memory_utilization < 1:
            raise SetupError('GPU_MEMORY_UTILIZATION must be greater than 0 and less than 1.')
        if min(self.max_model_len, self.max_num_seqs, self.startup_timeout,
               self.install_timeout, self.download_timeout) <= 0:
            raise SetupError('Lengths, concurrency and timeouts must be positive.')
        if not 1 <= self.retries <= 10:
            raise SetupError('RETRIES must be between 1 and 10.')
        for name in ('workspace','state_dir','hf_home'):
            if not (Path(getattr(self,name)).is_absolute() or PurePosixPath(getattr(self,name)).is_absolute()):
                raise SetupError(f'{name} must be an absolute path.')
        if Path(self.state_dir).resolve() in (Path(self.workspace).resolve(), Path('/').resolve()):
            raise SetupError('STATE_DIR must be a dedicated directory, not the workspace root.')

    @property
    def state(self):
        return Path(self.state_dir)

    @property
    def url(self):
        host = f'[{self.host}]' if ':' in self.host else self.host
        return f'http://{host}:{self.port}'

    def environment(self, python=None):
        env = os.environ.copy()
        env.update(HF_HOME=self.hf_home, HF_HUB_CACHE=str(Path(self.hf_home)/'hub'),
                   VLLM_CACHE_ROOT=str(self.state/'vllm-cache'), PIP_CACHE_DIR=str(self.state/'pip-cache'),
                   TMPDIR=str(self.state/'tmp'), PYTHONUNBUFFERED='1')
        env.setdefault('CUDA_VISIBLE_DEVICES', '0')
        # Calling /venv/bin/python does not activate /venv/bin for child executables.
        # FlashInfer launches ninja/nvcc by name, so pass their search path explicitly.
        bins = [str(self.state/'tools-venv/bin')]
        if python is not None:
            bins.insert(0,str(Path(python).absolute().parent))
        for home in (env.get('CUDA_HOME'),env.get('CUDA_PATH'),'/usr/local/cuda'):
            if home and (Path(home)/'bin').is_dir():
                bins.append(str(Path(home)/'bin'))
        env['PATH'] = os.pathsep.join(bins+[env.get('PATH',os.defpath)])
        return env


@contextmanager
def lock(path):
    import fcntl
    with Path(path).open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SetupError('Another setup/start/stop is running. Wait for it to finish.') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def ensure_layout(config):
    if not Path(config.workspace).is_dir():
        raise SetupError(f'{config.workspace} is missing. Run inside the RunPod with its workspace volume mounted.')
    for directory in (config.state, Path(config.hf_home), config.state/'logs', config.state/'tmp'):
        directory.mkdir(parents=True, exist_ok=True)
        if not os.access(directory, os.W_OK):
            raise SetupError(f'Directory is not writable: {directory}')


def require_space(path, needed, reason):
    free = shutil.disk_usage(path).free
    say(f'Disk ({reason}): {free/GIB:.1f} GiB free; {needed/GIB:.1f} GiB required.')
    if free < needed:
        raise SetupError(f'Not enough space for {reason}: need {needed/GIB:.1f} GiB, have {free/GIB:.1f} GiB. Increase the volume; caches are not deleted automatically.')


def emit(data):
    print(MARKER + json.dumps(data), flush=True)


def extract_result(output):
    for line in reversed(output.splitlines()):
        if line.startswith(MARKER):
            try:
                return json.loads(line[len(MARKER):])
            except ValueError as exc:
                raise SetupError('Helper returned invalid JSON. Inspect its log.') from exc
    raise SetupError('Helper returned no result. Inspect its log for a dependency failure or crash.')


def load_vllm_extension():
    """CUDA wheels use the stable-libtorch module in recent vLLM releases."""
    names = ('vllm._C_stable_libtorch', 'vllm._C')
    for name in names:
        try:
            importlib.import_module(name)
            return name
        except ModuleNotFoundError as exc:
            # Fall back only when this exact extension is absent. Missing dependencies,
            # undefined symbols and shared-library errors must remain visible.
            if exc.name != name:
                raise
    raise SetupError('No vLLM CUDA extension found (tried ' + ', '.join(names) + '). '
                     'Inspect the installed vLLM GPU wheel and runtime logs.')


def probe_worker(version):
    details = {'python':sys.version.split()[0], 'python_executable':sys.executable}
    stage = 'package_imports'
    try:
        import importlib.metadata as metadata
        from packaging.version import Version
        import torch
        import transformers
        import huggingface_hub
        installed = metadata.version('vllm')
        details.update(vllm=installed,torch=torch.__version__,cuda=torch.version.cuda,
                       transformers=transformers.__version__,huggingface_hub=huggingface_hub.__version__)
        stage = 'package_versions'
        if Version(installed).base_version != Version(version).base_version:
            raise RuntimeError(f'vLLM {installed} is installed; expected {version}')
        if Version(transformers.__version__) < Version('5.8.0'):
            raise RuntimeError(f'transformers {transformers.__version__} is older than 5.8.0')
        stage = 'cuda_runtime'
        if not torch.cuda.is_available():
            raise RuntimeError('PyTorch cannot access CUDA. Check the RunPod GPU and driver/image compatibility.')
        if torch.cuda.get_device_capability(0)[0] < 10:
            emit(dict(details,ok=False,stage=stage,hardware_error=True,error='This NVFP4 profile needs an NVIDIA Blackwell GPU (RTX 5090 or compatible).')); return
        free,total = torch.cuda.mem_get_info(0)
        if total < 30*GIB:
            emit(dict(details,ok=False,stage=stage,hardware_error=True,error=f'This profile needs a 32 GB-class GPU; CUDA reports {total/GIB:.1f} GiB.')); return
        if not torch.version.cuda or Version(torch.version.cuda) < Version('12.8'):
            raise RuntimeError('Blackwell requires a CUDA 12.8+ PyTorch build.')
        details.update(gpu=torch.cuda.get_device_name(0),free_bytes=free,total_bytes=total)
        stage = 'vllm_extension'
        extension = load_vllm_extension()
        details['extension'] = extension
        stage = 'vllm_cuda_kernel'
        inputs = torch.ones((1,128),device='cuda',dtype=torch.float16)
        weight = torch.ones(128,device='cuda',dtype=torch.float16)
        output = torch.empty_like(inputs)
        torch.ops._C.rms_norm(output,inputs,weight,1e-6)
        torch.cuda.synchronize()
        if not torch.allclose(output,torch.ones_like(output),atol=1e-3,rtol=1e-3):
            raise RuntimeError('vLLM CUDA RMSNorm verification returned an unexpected result.')
        emit(dict(details,ok=True,stage='complete',kernel_check='rms_norm'))
    except Exception as exc:
        emit(dict(details,ok=False,stage=stage,error=redact(f'{type(exc).__name__}: {exc}')))


def probe(python, config):
    try:
        result = subprocess.run([str(python),str(SCRIPT),'_probe',config.vllm_version],
                                env=config.environment(python),capture_output=True,text=True,
                                encoding='utf-8',errors='replace',timeout=120)
        data = extract_result(result.stdout)
    except (OSError,subprocess.TimeoutExpired,SetupError) as exc:
        data = {'ok':False,'stage':'probe_process','error':redact(str(exc))}
    save_json(config.state/'logs/runtime-check.json',
              {'checked_at':datetime.now().astimezone().isoformat(timespec='seconds'),
               'python_executable':str(python),'result':data})
    return data


def failure_hint(error):
    """Offer the relevant next step, instead of showing an OOM hint for all errors."""
    text = str(error).lower()
    if 'ninja' in text and ('no such file' in text or 'missing' in text or 'not found' in text):
        return ('FlashInfer JIT build tool ninja is unavailable. Update this repository and run bash setup.sh; '
                'setup installs ninja and supplies the runtime bin directory in PATH. Model caches are preserved.')
    if 'no vllm cuda extension' in text or "no module named 'vllm." in text:
        return ('vLLM extension check failed. Update this repository, inspect logs/runtime-check.json, '
                'and verify the installed GPU wheel. Cached models are preserved.')
    if any(word in text for word in ('undefined symbol','libtorch','libcudart','libcuda.so')):
        return ('CUDA/PyTorch shared-library mismatch. Inspect logs/runtime-check.json and install logs; '
                'use a compatible isolated vLLM runtime before changing system packages.')
    if 'out of memory' in text or 'gpu busy:' in text:
        return ('GPU memory: inspect nvidia-smi; use MAX_MODEL_LEN=4096 if needed. '
                'Unrelated GPU processes were not stopped.')
    if 'not enough space' in text or 'no space left' in text:
        return ('Disk space: inspect df -h /workspace and expand storage if needed. '
                'Existing model caches were not deleted.')
    if 'occupied' in text or 'address already in use' in text:
        return 'Port conflict: inspect the existing service or select another PORT; no other service was stopped.'
    if '401' in text or '403' in text:
        return 'Access denied: check the model access policy and HF_TOKEN. Do not paste credentials into logs or the repository.'
    if 'modulenotfounderror' in text or 'no module named' in text:
        return 'Python dependency missing: inspect logs/runtime-check.json and the isolated runtime install log.'
    return 'Inspect the stage and log path printed above; runtime details are in STATE_DIR/logs/runtime-check.json.'


def terminate_spawned(process):
    """Only signal a process group created by this invocation."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def run_logged(args, config, label, timeout):
    path = config.state/'logs'/f'{label}.log'
    with path.open('w',encoding='utf-8') as log:
        say(f'{label}: started. Log: {path}')
        process = subprocess.Popen([str(x) for x in args],env=config.environment(args[0]),stdin=subprocess.DEVNULL,
                                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        deadline,next_update = time.monotonic()+timeout,time.monotonic()+20
        try:
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    raise SetupError(f'{label} timed out after {timeout}s; downloaded data remains cached.',retryable=True)
                if time.monotonic() >= next_update:
                    say(f'{label}: still running. Progress: tail -f {path}')
                    next_update = time.monotonic()+20
                time.sleep(0.5)
        except BaseException:
            terminate_spawned(process)
            raise
    output = tail(path,32000)
    if process.returncode:
        transient = process.returncode == 21 or bool(re.search(
            r'ConnectionError|ConnectTimeout|ReadTimeout|Temporary failure|Connection reset|429|503|timed out',output,re.I))
        if process.returncode in (20,22):
            transient = False
        raise SetupError(f'{label} failed (exit {process.returncode}).\n{output[-5000:]}',retryable=transient)
    return output


def choose_runtime(config):
    previous = load_json(config.state/'runtime.json',{})
    seen = set()
    for candidate in (previous.get('python'),sys.executable,str(config.state/'venv/bin/python')):
        if not candidate or candidate in seen or not Path(candidate).exists():
            continue
        seen.add(candidate)
        result = probe(candidate,config)
        if result.get('hardware_error'):
            raise SetupError(result['error'])
        if result.get('ok'):
            say(f'Reusing runtime: {candidate}; vLLM {result["vllm"]}; {result["gpu"]}.')
            save_json(config.state/'runtime.json',{'python':candidate,'probe':result})
            return candidate,result
        say(f'Runtime not reusable: {candidate}: {result.get("error")}')
    require_space(config.state,20*GIB,'isolated runtime installation')
    python = config.state/'venv/bin/python'
    if not python.exists():
        run_logged([sys.executable,'-m','venv',str(config.state/'venv')],config,'venv',120)
    if not python.exists():
        raise SetupError('Virtual environment could not be created. Use an image with Python venv/ensurepip support.')
    retry(lambda: run_logged([python,'-m','pip','install','--disable-pip-version-check','--timeout','60',
                             '--retries','2','--only-binary=vllm,torch',f'vllm=={config.vllm_version}',
                              'transformers>=5.8.0,<6','huggingface_hub>=0.34,<2','ninja>=1.11,<2'],
                             config,'install',config.install_timeout),config.retries)
    run_logged([python,'-m','pip','check'],config,'pip-check',60)
    result = probe(python,config)
    if not result.get('ok'):
        raise SetupError(f'Installed runtime verification failed at {result.get("stage","unknown")}: '
                         f'{result.get("error")}. Diagnostics: {config.state/"logs/runtime-check.json"}. '
                         'Host drivers were not modified.')
    save_json(config.state/'runtime.json',{'python':str(python),'probe':result})
    return str(python),result


def ensure_build_tools(config, python):
    """Repair missing Ninja without replacing torch/vLLM; check JIT tools before weights load."""
    env = config.environment(python)
    if not shutil.which('ninja',path=env['PATH']):
        managed = config.state/'venv/bin/python'
        if Path(python).absolute() == managed.absolute():
            installer = str(python)
        else:
            # A compatible system runtime can be reused without modifying its packages.
            installer = str(config.state/'tools-venv/bin/python')
            if not Path(installer).exists():
                run_logged([sys.executable,'-m','venv',str(config.state/'tools-venv')],
                           config,'build-tools-venv',120)
        retry(lambda: run_logged([installer,'-m','pip','install','--disable-pip-version-check',
                                 '--timeout','60','--retries','2','--only-binary=ninja','ninja>=1.11,<2'],
                                 config,'build-tools-install',config.install_timeout),config.retries)
    env = config.environment(python)
    tools = {'ninja':shutil.which('ninja',path=env['PATH']),
             'cxx':shutil.which(env.get('CXX','c++'),path=env['PATH']),
             'nvcc':shutil.which(env.get('CUDACXX','nvcc'),path=env['PATH'])}
    missing = [name for name,path in tools.items() if not path]
    if missing:
        raise SetupError('Missing FlashInfer JIT build tools: ' + ', '.join(missing) + '. '
                         'Use a CUDA development image with a C++ compiler and nvcc. '
                         'Ninja is installed automatically; model weights have not been loaded.')
    versions = {}
    for name,path in tools.items():
        try:
            result = subprocess.run([path,'--version'],env=env,capture_output=True,text=True,
                                    encoding='utf-8',errors='replace',timeout=20)
        except (OSError,subprocess.TimeoutExpired) as exc:
            raise SetupError(f'FlashInfer build tool {name} could not run: {exc}') from exc
        if result.returncode:
            raise SetupError(f'FlashInfer build tool {name} failed its version check (exit {result.returncode}).')
        versions[name] = {'path':path,'version':redact((result.stdout+result.stderr).strip()[:1200])}
    save_json(config.state/'logs/build-tools.json',versions)
    say('FlashInfer JIT tools verified: ninja, C++ compiler, nvcc.')
    return versions


def complete_snapshot(path):
    path = Path(path)
    try:
        for name in ('config.json','tokenizer_config.json','tokenizer.json'):
            if not (path/name).is_file() or (path/name).stat().st_size == 0:
                return False
        index = json.loads((path/'model.safetensors.index.json').read_text(encoding='utf-8'))
        if not isinstance(index,dict) or not isinstance(index.get('weight_map'),dict):
            return False
        shards = set(index['weight_map'].values())
        if not shards:
            return False
        for name in shards:
            # HF snapshot symlinks normally point outside the snapshot into ../blobs.
            candidate = (path/name).resolve()
            if Path(name).name != name or not candidate.is_file() or candidate.stat().st_size == 0:
                return False
        return True
    except (OSError,ValueError,KeyError,TypeError):
        return False


def download_worker(model, revision, cache):
    try:
        from huggingface_hub import HfApi,snapshot_download,try_to_load_from_cache
        token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
        info = HfApi(token=token).model_info(model,revision=revision,files_metadata=True,timeout=30)
        missing = 0
        for entry in info.siblings:
            size = entry.size
            if size is None:
                if entry.rfilename.endswith('.safetensors'):
                    raise SetupError('Model weight sizes are unavailable; cannot safely check disk space.')
                size = 1024*1024
            cached = try_to_load_from_cache(model,entry.rfilename,revision=info.sha,cache_dir=cache)
            if not isinstance(cached,str) or not Path(cached).is_file() or Path(cached).stat().st_size != size:
                missing += size
        require_space(Path(cache).parent,missing+3*GIB,'uncached model files plus reserve')
        snapshot = snapshot_download(model,revision=info.sha,cache_dir=cache,token=token,max_workers=4)
        if not complete_snapshot(snapshot):
            raise SetupError('Snapshot is incomplete: metadata/tokenizer/weight shards are missing.')
        emit({'snapshot':snapshot,'commit':info.sha,'model':model,'requested_revision':revision})
    except Exception as exc:
        status = getattr(getattr(exc,'response',None),'status_code',None)
        permanent = isinstance(exc,SetupError) or status in (400,401,403,404,422) or getattr(exc,'errno',None) in (13,28)
        emit({'error':redact(f'{type(exc).__name__}: {exc}'),'http_status':status})
        raise SystemExit(20 if permanent else 21)


def download(config, python):
    previous = load_json(config.state/'model.json',{})
    if (previous.get('model') == config.model and previous.get('requested_revision') == config.revision
            and complete_snapshot(previous.get('snapshot',''))):
        say(f'Reusing complete cached model at commit {previous["commit"]}.')
        return previous
    result = extract_result(retry(lambda:run_logged([python,str(SCRIPT),'_download',config.model,
                            config.revision,str(Path(config.hf_home)/'hub')],config,'download',config.download_timeout),config.retries))
    save_json(config.state/'model.json',result)
    return result


def process_identity(pid):
    """Do not trust a PID alone across reboots or PID reuse."""
    try:
        stat = (Path('/proc')/str(int(pid))/'stat').read_text()
        fields = stat.rsplit(')',1)[1].split()
        if fields[0] == 'Z':
            return None
        return {'pid':int(pid),'start_ticks':fields[19],
                'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    except (FileNotFoundError,ProcessLookupError,IndexError,ValueError):
        return None


def same_process(record):
    identity = process_identity(record.get('pid',-1))
    return bool(identity and all(record.get(key) == value for key,value in identity.items()))


def request_json(config, route, payload=None, timeout=5):
    headers = {'Content-Type':'application/json'}
    if os.environ.get('VLLM_API_KEY'):
        headers['Authorization'] = 'Bearer '+os.environ['VLLM_API_KEY']
    request = urllib.request.Request(config.url+route,headers=headers,
                                     data=json.dumps(payload).encode() if payload is not None else None)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request,timeout=timeout) as response:
        raw = response.read(1024*1024)
        return json.loads(raw) if raw else {}


def ready(config):
    try:
        result = request_json(config,'/v1/models')
        if not isinstance(result,dict) or not isinstance(result.get('data'),list):
            return False
        return any(isinstance(entry,dict) and entry.get('id') == config.model for entry in result['data'])
    except (OSError,ValueError):
        return False


def port_busy(config):
    family = socket.AF_INET6 if ':' in config.host else socket.AF_INET
    with socket.socket(family,socket.SOCK_STREAM) as sock:
        try:
            sock.bind((config.host,config.port))
            return False
        except OSError:
            return True


def build_server_command(config, python, snapshot):
    return [str(python),'-m','vllm.entrypoints.cli.main','serve',str(snapshot),
            '--served-model-name',config.model,'--host',config.host,'--port',str(config.port),
            '--tensor-parallel-size','1','--max-model-len',str(config.max_model_len),
            '--max-num-seqs',str(config.max_num_seqs),'--gpu-memory-utilization',str(config.gpu_memory_utilization),
            '--kv-cache-dtype','fp8','--enforce-eager','--reasoning-parser','qwen3']


def existing_server(config):
    record = load_json(config.state/'server.json',{})
    if not same_process(record):
        return None
    if record.get('config') != asdict(config) or record.get('cuda_visible_devices') != config.environment()['CUDA_VISIBLE_DEVICES']:
        raise SetupError('Managed server is running with different settings. Run stop using the original STATE_DIR first.')
    return record


def start(config, python, model):
    if existing_server(config):
        if ready(config):
            say(f'Already running: {config.url}/v1'); return
        raise SetupError('Managed server is starting or unhealthy. Inspect status/logs; no duplicate will be started.')
    if port_busy(config):
        raise SetupError(f'Port {config.port} is occupied. Change PORT or stop its owner; no existing process was killed.')
    result = probe(python,config)
    if not result.get('ok'):
        raise SetupError(f'Runtime is not usable: {result.get("error")}. Run setup again.')
    budget = result['total_bytes']*config.gpu_memory_utilization
    if result['free_bytes'] < budget:
        raise SetupError(f'GPU busy: {result["free_bytes"]/GIB:.1f} GiB free; requested budget {budget/GIB:.1f} GiB. Inspect nvidia-smi; unrelated processes were not killed.')
    ensure_build_tools(config,python)
    save_json(config.state/'launch.json',{'command':build_server_command(config,python,model['snapshot'])})
    log_path = config.state/'logs/server.log'
    if log_path.exists():
        os.replace(log_path,log_path.with_name(f'server-{time.time_ns()}.log'))
    with log_path.open('w',encoding='utf-8') as log:
        process = subprocess.Popen([str(python),str(SCRIPT),'_serve',str(config.state)],env=config.environment(python),
                                   stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        record = process_identity(process.pid)
        if not record:
            raise SetupError(f'Server supervisor exited before initialization.\n{tail(log_path)}')
        record.update(config=asdict(config),cuda_visible_devices=config.environment()['CUDA_VISIBLE_DEVICES'])
        save_json(config.state/'server.json',record)
        deadline,next_update = time.monotonic()+config.startup_timeout,0
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise SetupError(f'Server exited (code {process.returncode}). Log: {log_path}\n{tail(log_path,64000)}')
            if ready(config):
                MANAGED_CHILDREN[process.pid] = process
                say(f'READY: {config.url}/v1 ; model={config.model}'); return
            if time.monotonic() >= next_update:
                say(f'Loading model. Log: {log_path}')
                next_update = time.monotonic()+20
            time.sleep(2)
        raise SetupError(f'Server startup timed out after {config.startup_timeout}s.\n{tail(log_path)}')
    except BaseException:
        terminate_spawned(process)
        (config.state/'server.json').unlink(missing_ok=True)
        raise


def serve_worker(state):
    """Own process group contains this supervisor, vLLM and engine workers."""
    spec = load_json(Path(state)/'launch.json')
    stopping = False
    def stop_signal(_signal,_frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM,stop_signal)
    signal.signal(signal.SIGINT,stop_signal)
    child = subprocess.Popen(spec['command'],stdin=subprocess.DEVNULL)
    deadline = None
    while True:
        code = child.poll()
        if code is not None and not stopping:
            say(f'vLLM exited with code {code}; cleaning up its process group.')
            os.killpg(os.getpgrp(),signal.SIGTERM)
        if stopping:
            if deadline is None:
                os.killpg(os.getpgrp(),signal.SIGTERM)
                deadline = time.monotonic()+10
            if time.monotonic() >= deadline:
                os.killpg(os.getpgrp(),signal.SIGKILL)
        time.sleep(0.2)


def stop(config):
    record = load_json(config.state/'server.json',{})
    if not same_process(record):
        say('No live managed server. Stale PID state will not be signalled.')
        (config.state/'server.json').unlink(missing_ok=True); return
    pid = record['pid']
    if os.getpgid(pid) != pid:
        raise SetupError('Managed PID is not its own process group. Refusing to signal unrelated processes.')
    try:
        os.killpg(pid,signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic()+20
    while same_process(record) and time.monotonic() < deadline:
        time.sleep(0.2)
    if same_process(record):
        try:
            os.killpg(pid,signal.SIGKILL)
        except ProcessLookupError:
            pass
    child = MANAGED_CHILDREN.pop(pid,None)
    if child is not None:
        child.wait(timeout=5)
    (config.state/'server.json').unlink(missing_ok=True)
    say('Managed server stopped; runtime and model cache preserved.')


def smoke(config):
    if not ready(config):
        raise SetupError('Expected model is not ready. Inspect status or server.log.')
    response = request_json(config,'/v1/chat/completions',{
        'model':config.model,'messages':[{'role':'user','content':'Reply with exactly OK.'}],
        'max_tokens':64,'temperature':0.7,'chat_template_kwargs':{'enable_thinking':False}},timeout=120)
    try:
        content = response['choices'][0]['message']['content']
    except (KeyError,IndexError,TypeError) as exc:
        raise SetupError('Inference returned an unexpected response shape.') from exc
    if not isinstance(content,str) or not content.strip():
        raise SetupError('Inference returned an empty answer. Inspect server.log.')
    say(f'Inference succeeded: {content.strip()[:200]}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',nargs='?',default='setup',choices=['setup','start','check','status','stop','logs','test','plan'])
    args = parser.parse_args(argv)
    config = Config.from_env()
    if args.command == 'plan':
        print(json.dumps(asdict(config),indent=2)); return 0
    if platform.system() != 'Linux':
        raise SetupError('Run inside the RunPod Linux terminal. Local PowerShell cannot configure the remote GPU.')
    if not (3,10) <= sys.version_info[:2] <= (3,13):
        raise SetupError('Use Python 3.10-3.13; Python 3.12 is recommended.')
    os.umask(0o077)
    ensure_layout(config)
    if args.command == 'logs':
        print(tail(config.state/'logs/server.log')); return 0
    if args.command == 'status':
        record = load_json(config.state/'server.json',{})
        running = same_process(record)
        saved = Config(**record['config']) if running else config
        healthy = running and ready(saved)
        say(f'Managed server: {"READY" if healthy else "STARTING/UNHEALTHY" if running else "STOPPED"}; {saved.url}/v1')
        return 0 if healthy else 1
    if args.command == 'test':
        smoke(config); return 0
    with lock(config.state/'setup.lock'):
        if args.command == 'stop':
            stop(config); return 0
        if args.command in ('setup','start') and existing_server(config):
            if ready(config):
                say(f'Already running: {config.url}/v1'); return 0
            raise SetupError('Managed server is starting or unhealthy. Inspect status/logs; no duplicate will be started.')
        if args.command in ('setup','start') and port_busy(config):
            raise SetupError(f'PORT={config.port} is occupied. No packages or model data were downloaded.')
        if not shutil.which('nvidia-smi'):
            raise SetupError('nvidia-smi missing. Select an NVIDIA GPU RunPod image; host drivers cannot be fixed by this script.')
        gpu = subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.free,driver_version','--format=csv,noheader'],capture_output=True,text=True,timeout=30)
        if gpu.returncode:
            raise SetupError(f'NVIDIA driver is not usable: {redact(gpu.stderr)}')
        say(gpu.stdout.strip())
        if args.command == 'check':
            runtime = load_json(config.state/'runtime.json',{})
            result = probe(runtime.get('python',sys.executable),config)
            print(json.dumps(result,indent=2)); return 0 if result.get('ok') else 1
        if args.command == 'start':
            runtime,model = load_json(config.state/'runtime.json',{}),load_json(config.state/'model.json',{})
            if not runtime.get('python') or not complete_snapshot(model.get('snapshot','')):
                raise SetupError('No complete setup state. Run: bash setup.sh')
            if model.get('model') != config.model or model.get('requested_revision') != config.revision:
                raise SetupError('Model settings changed. Run setup to resolve/download the requested model.')
            start(config,runtime['python'],model)
        else:
            python,result = choose_runtime(config)
            say(f'CUDA verified: PyTorch {result["torch"]}, CUDA {result["cuda"]}.')
            ensure_build_tools(config,python)
            start(config,python,download(config,python))
    return 0


def entrypoint():
    if len(sys.argv)>1 and sys.argv[1]=='_probe':
        probe_worker(sys.argv[2]); return 0
    if len(sys.argv)>1 and sys.argv[1]=='_download':
        download_worker(*sys.argv[2:5]); return 0
    if len(sys.argv)>1 and sys.argv[1]=='_serve':
        serve_worker(sys.argv[2]); return 0
    def interrupted(_signal,_frame):
        raise KeyboardInterrupt
    for name in ('SIGTERM','SIGHUP'):
        if hasattr(signal,name):
            signal.signal(getattr(signal,name),interrupted)
    try:
        return main()
    except KeyboardInterrupt:
        say('Interrupted. Current child operation stopped; cached downloads preserved.'); return 130
    except (SetupError,OSError,subprocess.SubprocessError,ValueError) as exc:
        say(f'ERROR: {type(exc).__name__}: {exc}')
        say(failure_hint(exc))
        return 1


if __name__=='__main__':
    raise SystemExit(entrypoint())
