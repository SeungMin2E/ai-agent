# RunPod Qwen 자동 설정

RTX 5090 한 장에서 `Inferact/Qwen3.8-27B-NVFP4`를 vLLM으로 실행합니다.
환경 검사 → 호환 런타임 확보 → NVFP4 커널 사전 검사 → 모델 다운로드 → 백그라운드 실행 → 실제 짧은 추론까지 자동 처리합니다.

## 개발 PC의 GitHub 자동 업로드

**2026-10-16 사용자 요청으로 ai-agent 자동 업로드를 다시 켰습니다.** 원본 Windows 개발 폴더를 저장한 뒤 60초 동안 추가 변경이 없으면 로컬 테스트를 통과한 변경을 `main`에 자동 커밋합니다. PC가 켜져 있고 로그인 및 인터넷 연결이 필요합니다. 테스트 실패나 GitHub의 같은 파일과 충돌이 있으면 업로드를 보류합니다.

감시 설정과 켜기·끄기·상태 확인 버튼은 개발 PC의 `Deeppoint\github-auto-upload` 폴더에 있습니다. ai-agent와 테스트 사이트 두 폴더만 자동 업로드하며 Windows 로그인 후에도 감시를 시작합니다. RunPod에는 자동으로 배포하지 않으므로 저장소 폴더에서 `git pull --ff-only`로 최신 수정본을 가져오고 실행해야 합니다. RunPod에서 파일을 직접 수정한 경우에는 먼저 변경 내용을 확인하고 보관하세요.

## 실행

**RunPod에 SSH로 접속한 Linux 터미널에서 실행합니다.** 프롬프트 예시는 `root@...:/workspace#`입니다.
로컬 Windows의 `PS C:\...>`에서 실행하는 GPU 설치 명령이 아닙니다.

```bash
cd /workspace
git clone https://github.com/SeungMin2E/ai-agent.git
cd ai-agent
bash setup.sh
```

이미 받았다면 `cd /workspace/ai-agent` 후 `bash setup.sh`만 실행합니다.
`git`이 없는 이미지에서는 Code → Download ZIP으로 받은 파일을 `/workspace/ai-agent`에 풀어도 됩니다.

완료되면 `READY: http://127.0.0.1:8000/v1`이 표시됩니다. `/v1/models`에서 요청 모델을 확인하고 짧은 실제 답변 생성까지 성공해야 준비 완료로 판단합니다.
이후 실제 추론을 다시 확인하려면 다음을 실행합니다.

```bash
bash setup.sh test
```

설치 중에는 SSH 연결을 유지하세요. **READY 이후 서버는 SSH 세션과 분리된 백그라운드 프로세스로 유지됩니다.**
Pod 재시작 후 자동 부팅되지는 않으므로 `bash setup.sh start`를 실행합니다.

## 관리 명령

| 명령 | 동작 |
| --- | --- |
| `bash setup.sh` | 전체 설정·실행. 같은 설정의 관리 서버가 있으면 중복 실행하지 않음 |
| `bash setup.sh check` | GPU·CUDA·vLLM 검사. 패키지 설치·모델 다운로드 없음 |
| `bash setup.sh start` | 저장된 런타임·모델로 실행 |
| `bash setup.sh preflight` | 런타임 확보·빌드 도구 검사·실제 NVFP4 CUTLASS 연산. 모델 다운로드·로딩·서버 실행 없음 |
| `bash setup.sh diagnose` | 버전·빌드 도구·호스트 RAM·컨테이너 RAM 제한·OOM 카운터·원인 로그를 모아서 출력. 패키지 설치·GPU 작업 없음 |
| `bash setup.sh status` | 프로세스와 요청한 모델의 준비 상태 확인 |
| `bash setup.sh logs` | 최근 서버 로그 |
| `bash setup.sh test` | 짧은 실제 추론 요청 |
| `bash setup.sh stop` | 이 스크립트의 서버만 종료. 런타임·캐시 보존 |
| `bash setup.sh plan` | 적용할 설정만 출력 |

실시간 로그는 `tail -f /workspace/ai-agent-state/logs/server.log`로 확인합니다.

## 기본값

| 항목 | 값 |
| --- | --- |
| 모델 | `Inferact/Qwen3.8-27B-NVFP4` |
| vLLM | `0.31.0` |
| 최대 모델 길이 / 동시 요청 | `8192` / `1` |
| GPU 메모리 비율 | `0.90` |
| KV 캐시 / 추론 파서 | `fp8` / `qwen3` |
| CUDA graph | `--enforce-eager`로 비활성 |
| API | `127.0.0.1:8000` |
| 모델 캐시 | `/workspace/hf-cache` |
| 상태·런타임·로그 | `/workspace/ai-agent-state` |
| 준비 제한 | 900초 |
| 첫 답변·초기 CUDA 컴파일 제한 | `WARMUP_TIMEOUT=1800`초 — 모델 준비 제한과 별도 |
| 설치 / 다운로드 제한 | 시도당 3,600초 / 10,800초 |
| NVFP4 사전 검사 제한 | 1,800초 |
| JIT 빌드 동시 작업 | `MAX_JOBS=1` — 호스트 RAM 사용 억제 |
| 일시 오류 재시도 | 최대 3회 |

환경변수로 바꿉니다. 이후 관리 명령에도 같은 값을 적용하세요.

```bash
export PORT=8001
export MAX_MODEL_LEN=4096
bash setup.sh
```

지원 변수: `WORKSPACE`, `STATE_DIR`, `HF_HOME`, `MODEL_ID`, `MODEL_REVISION`, `VLLM_VERSION`, `HOST`, `PORT`,
`MAX_MODEL_LEN`, `MAX_NUM_SEQS`, `GPU_MEMORY_UTILIZATION`, `STARTUP_TIMEOUT`, `WARMUP_TIMEOUT`, `INSTALL_TIMEOUT`, `DOWNLOAD_TIMEOUT`, `JIT_TIMEOUT`, `MAX_JOBS`, `RETRIES`.
`HOST`는 localhost 주소만 허용합니다. `CUDA_VISIBLE_DEVICES`는 지정값을 유지하고, 없으면 GPU 0을 사용합니다.
32GB급 Blackwell GPU·단일 GPU용 프로필이며, 다른 모델은 메모리·아키텍처 호환성을 별도로 검토해야 합니다.

최초 다운로드에서 모델 브랜치를 실제 커밋으로 해석해 기록합니다. 재실행은 기록된 완전한 캐시를 사용하며 자동으로 모델을 갱신하지 않습니다.
갱신하려면 `MODEL_REVISION`에 새 커밋을 지정하세요. 메타데이터·토크나이저·각 가중치 파일의 존재와 비어 있지 않은지를 확인하며 전체 가중치 해시 검사는 하지 않습니다.

## 오류 처리

- Python 3.10~3.13, GPU 드라이버, vLLM 확장 모듈, CUDA 커널 실행, GPU 종류·용량 확인.
- vLLM 0.31의 `vllm._C_stable_libtorch` 확장을 우선 확인하고, 구버전은 `vllm._C`를 사용합니다. 확장 모듈 로드 후 vLLM RMSNorm CUDA 커널의 실행 결과도 검사합니다.
- 실패한 검사 단계와 확인된 Python·vLLM·PyTorch·CUDA 버전은 `STATE_DIR/logs/runtime-check.json`에 기록합니다. 확장 모듈 오류, 라이브러리 불일치, GPU 메모리 부족, 디스크 부족, 포트 충돌은 원인별 조치 안내를 출력합니다.
- FlashInfer의 NVFP4 JIT 빌드에 필요한 `ninja`를 설치하고 실행 환경의 `PATH`에 선택한 Python 런타임의 `bin` 경로를 포함합니다. 기존 런타임을 재사용할 때도 누락된 Ninja를 보완합니다. 별도 시스템 런타임을 재사용하는 경우에는 작은 도구용 가상환경에 설치하여 시스템 패키지를 변경하지 않습니다.
- 모델 다운로드·로딩 전에 Ninja, C++ 컴파일러, `nvcc` 실행을 확인하고 `STATE_DIR/logs/build-tools.json`에 기록합니다. C++ 컴파일러와 CUDA Toolkit이 빠진 이미지라면 개발 도구가 포함된 CUDA 이미지가 필요하다는 안내로 중단합니다.
- 모델 다운로드·로딩 전에 FlashInfer의 실제 NVFP4 양자화 및 CUTLASS 행렬 연산을 128×128 입력으로 검사합니다. 작은 연산도 최초에는 필요한 CUDA 모듈을 컴파일하므로 시간이 걸립니다. 기존 FlashInfer 빌드 캐시는 삭제하지 않습니다.
- 첫 토큰 생성에서 지연 로딩되는 FlashInfer `sampling` 모듈도 사전 검사에서 실제 연산으로 확인합니다. 모델 로딩 전에 컴파일하여 첫 답변의 대기 시간과 RAM 부담을 줄입니다.
- 모델 API 준비 후 첫 답변 검사는 별도 프로세스로 최대 `WARMUP_TIMEOUT`초 기다립니다. 20초마다 대기 상태를 출력하고, 중단·시간 초과 시 검사 프로세스와 이번 실행의 모델 서버를 정리합니다. 일반 `test` 명령의 응답 제한은 120초입니다. 첫 답변 결과는 `logs/warmup.log`에 기록합니다.
- 컴파일 동시 작업을 기본 1개로 제한합니다. 사전 검사 전후의 호스트 RAM 및 cgroup 메모리·OOM 카운터와 결과를 `STATE_DIR/logs/preflight.json`에 기록합니다. `code=137`·`Killed`는 시스템 RAM 관련 가능성을 별도로 안내하며 GPU 문맥 길이 조절로 일괄 안내하지 않습니다.
- 재사용하는 런타임에도 `pip check`를 실행하여 의존성 불일치를 확인합니다. 서버 기동에 실패하면 마지막 요약 오류와 함께 앞쪽 원인 줄을 추출해 `STATE_DIR/logs/startup-failure.json`에 보존합니다.
- 호환되는 기존 환경 재사용. 새 설치는 별도 가상환경에서 수행하여 시스템 PyTorch를 덮어쓰지 않음.
- 별도 런타임 설치 전 20GiB 여유 확인. 모델 다운로드 전 파일 크기·캐시를 대조하여 미다운로드 용량 + 3GiB 확인.
- 네트워크 시간 초과 등 일시 오류만 제한 재시도. 401·403·404와 디스크 부족 등은 즉시 보고.
- 설치·다운로드·서버 단계별 로그. 실패·중단 후 캐시 보존.
- 포트 충돌·GPU 사용 중·중복 실행 감지. 다른 프로세스를 임의 종료하지 않음.
- PID·프로세스 시작 시점·부팅 ID를 함께 비교하여 PID 재사용 후 잘못된 종료 방지.
- 준비 실패·시간 초과·중단 시 이번 실행이 시작한 프로세스 그룹 정리.
- 손상된 상태 JSON과 잘못된 설정을 명시적으로 보고하고 조용히 덮어쓰지 않음.

하드웨어 부족, 호스트 드라이버 문제, upstream 패키지·모델 결함은 별도 조치가 필요할 수 있습니다.
공개 모델의 접근 정책이 바뀌면 환경변수 `HF_TOKEN`이 필요할 수 있습니다. 토큰은 저장소에 넣지 마세요.
상태 JSON에 토큰을 저장하지 않으며, 터미널 오류 출력의 알려진 토큰 값은 가립니다. 원본 로그는 검토 후 공유하세요.
Linux에서는 새 상태 파일·로그를 현재 사용자만 읽도록 생성합니다.

| 상황 | 조치 |
| --- | --- |
| `No module named 'vllm._C'` (예전 설치 스크립트) | 저장소에서 `git pull --ff-only` 후 `bash setup.sh` 재실행. vLLM 0.31의 확장 이름 변경을 반영한 검사 사용 |
| `No such file or directory: 'ninja'` | 저장소에서 `git pull --ff-only` 후 `bash setup.sh` 재실행. Ninja 설치 및 런타임 PATH 보완을 자동 처리 |
| Ninja 빌드 중 `code=137` / `Killed` | CUDA 컴파일러 강제 종료. `MAX_JOBS=1` 유지 후 `bash setup.sh preflight`로 먼저 확인. `diagnose`의 RAM 제한·OOM 카운터 대조 |
| API 기동 후 `Building JIT module sampling` 중 첫 답변 시간 초과 | `git pull --ff-only`로 sampling 사전 검사와 별도 첫 답변 제한을 반영한 수정본 사용. `WARMUP_TIMEOUT=1800` 기본값 유지. 직접 수정한 파일 때문에 pull이 거부되면 해당 변경을 보관하고 충돌 확인 |
| 포트 사용 중 | 기존 vLLM 확인 또는 다른 `PORT` 사용 |
| GPU 메모리 부족 | `nvidia-smi` 확인. 다른 작업 정리 후 `MAX_MODEL_LEN=4096`부터 재시도 |
| venv 생성 실패 | venv/ensurepip가 포함된 Python 3.12 GPU 이미지 사용. Ubuntu는 필요 시 `python3-venv` 설치 |
| HF 401/403 | 접근 권한·토큰 확인 |
| 다운로드 중단 | 같은 설정으로 재실행. 완성된 캐시 재사용 |
| 새 이미지에서 런타임 불일치 | `bash setup.sh` 재실행으로 호환성 확인·별도 설치 |
| `STARTING/UNHEALTHY` | `logs` 확인. 느린 로딩은 `STARTUP_TIMEOUT` 증가. 재설정 전 `stop` |

`/workspace`의 RunPod volume disk는 중지·시작 동안 유지되지만 Pod 삭제 시 사라집니다. Network volume 수명은 별도입니다.

## PC에서 접근

RunPod의 실제 SSH 접속 명령에 아래 포트 전달 옵션을 추가합니다. 필요한 SSH 키 옵션도 유지하세요.

```bash
ssh -L 8000:127.0.0.1:8000 -p <SSH_PORT> root@<RUNPOD_SSH_HOST>
```

터널을 유지한 상태에서 PC의 `http://127.0.0.1:8000/v1`로 접근합니다.

## 검증

```bash
python3 -m unittest discover -s tests -v
bash -n setup.sh
```

외부 다운로드·유료 GPU 없이 설정 오류, 재시도, 캐시·상태 손상, 포트 충돌, PID 재사용 방지를 검사합니다.
Linux에서는 가짜 HTTP 서버로 실행·중복 방지·종료·준비 실패·시간 초과 정리를 확인합니다.
NVFP4 검사 실패 시 모델 다운로드·로딩을 시작하지 않는 순서, 잘못된 연산 결과 거부, CPU RAM 관련 빌드 실패, 원인 로그 보존과 인증정보 마스킹도 검사합니다.
단위 테스트와 CI는 GPU 연산을 대신 실행하지 않습니다. 실제 GPU에서 먼저 `bash setup.sh preflight`를 실행하여 CUDA 빌드·NVFP4 연산을 확인하고, 통과 후 `bash setup.sh`로 모델 기동·답변 생성까지 확인합니다.

## 근거

- [vLLM 0.31 설치·호환성](https://docs.vllm.ai/en/v0.31.0/getting_started/installation/gpu/)
- [Qwen3.8-27B vLLM recipe: 단일 RTX 5090](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)
- [Inferact 모델 파일](https://huggingface.co/Inferact/Qwen3.8-27B-NVFP4/tree/main)
- [RunPod 저장소 수명](https://docs.runpod.io/pods/storage/types)
- [FlashInfer NVFP4 GEMM API](https://docs.flashinfer.ai/generated/flashinfer.gemm.mm_fp4.html)
