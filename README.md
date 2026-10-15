# RunPod Qwen 자동 설정

RTX 5090 한 장에서 `Inferact/Qwen3.8-27B-NVFP4`를 vLLM으로 실행합니다.
환경 검사 → 호환 런타임 재사용 또는 별도 설치 → 모델 다운로드 → 백그라운드 실행 → 준비 확인을 자동 처리합니다.

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

완료되면 `READY: http://127.0.0.1:8000/v1`이 표시됩니다. `/v1/models` 응답에 요청한 모델 이름이 포함돼야 준비 완료로 판단합니다.
실제 추론은 다음으로 확인합니다.

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
| 설치 / 다운로드 제한 | 시도당 3,600초 / 10,800초 |
| 일시 오류 재시도 | 최대 3회 |

환경변수로 바꿉니다. 이후 관리 명령에도 같은 값을 적용하세요.

```bash
export PORT=8001
export MAX_MODEL_LEN=4096
bash setup.sh
```

지원 변수: `WORKSPACE`, `STATE_DIR`, `HF_HOME`, `MODEL_ID`, `MODEL_REVISION`, `VLLM_VERSION`, `HOST`, `PORT`,
`MAX_MODEL_LEN`, `MAX_NUM_SEQS`, `GPU_MEMORY_UTILIZATION`, `STARTUP_TIMEOUT`, `INSTALL_TIMEOUT`, `DOWNLOAD_TIMEOUT`, `RETRIES`.
`HOST`는 localhost 주소만 허용합니다. `CUDA_VISIBLE_DEVICES`는 지정값을 유지하고, 없으면 GPU 0을 사용합니다.
32GB급 Blackwell GPU·단일 GPU용 프로필이며, 다른 모델은 메모리·아키텍처 호환성을 별도로 검토해야 합니다.

최초 다운로드에서 모델 브랜치를 실제 커밋으로 해석해 기록합니다. 재실행은 기록된 완전한 캐시를 사용하며 자동으로 모델을 갱신하지 않습니다.
갱신하려면 `MODEL_REVISION`에 새 커밋을 지정하세요. 메타데이터·토크나이저·각 가중치 파일의 존재와 비어 있지 않은지를 확인하며 전체 가중치 해시 검사는 하지 않습니다.

## 오류 처리

- Python 3.10~3.13, GPU 드라이버, vLLM 확장 모듈, CUDA 커널 실행, GPU 종류·용량 확인.
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
GPU에서 실제 모델 다운로드·기동·추론은 `bash setup.sh`와 `bash setup.sh test`로 확인해야 합니다.

## 근거

- [vLLM 0.31 설치·호환성](https://docs.vllm.ai/en/v0.31.0/getting_started/installation/gpu/)
- [Qwen3.8-27B vLLM recipe: 단일 RTX 5090](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)
- [Inferact 모델 파일](https://huggingface.co/Inferact/Qwen3.8-27B-NVFP4/tree/main)
- [RunPod 저장소 수명](https://docs.runpod.io/pods/storage/types)
