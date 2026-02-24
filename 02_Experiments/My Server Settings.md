# My Server Settings

> UOS HPC cluster / SLURM / gate1

---

## User Limits

| Item | Value |
|------|-------|
| Account | uos |
| Max Running Jobs | **10** |
| Max Submitted Jobs | 20 |
| Max Wall Time | **2일** (2-00:00:00) |

---

## GPU Partitions

| Partition | GPU | GPUs/Node | Nodes | Node Range | CPU/Node | RAM/Node | VRAM/GPU | DPO 사용 |
|-----------|-----|-----------|-------|------------|----------|----------|----------|----------|
| **gpu1** | RTX 3090 | 4 | 14 | n001-014 | 48 | 768 GB | **24 GB** | OOM |
| **gpu2** | A10 | 4 | 11 | n051-061 | 56 | 1 TB | **24 GB** | OOM |
| **gpu3** | A6000 Ada | 4 | 10 | n062-071 | 56 | 1 TB | **48 GB** | OK |
| **gpu4** | A6000 | 4 | 28+1 | n072-100 | 56 | 1 TB | **48 GB** | OK |
| **gpu5** | A6000 | 4 | 6 | n101-106 | 64 | 1 TB | **48 GB** | OK |
| **gpu6** | A10 | 4 | 25 | n015-039 | 48 | 768 GB | **24 GB** | OOM |

> n091만 GPU 3장 (gpu4 파티션)

### DPO 학습 가능 파티션
- **gpu3, gpu4, gpu5** (48GB VRAM) — Qwen2.5-0.5B DPO (bf16, batch=8) 정상 동작
- gpu1, gpu2, gpu6 (24GB VRAM) — CUDA OOM 발생

---

## CPU Partitions

| Partition | Nodes | Node Range | CPU/Node | RAM/Node |
|-----------|-------|------------|----------|----------|
| **cpu1** | 10 | n040-049 | 48 | 768 GB |
| **cpu2** | 10 | n107-116 | 256 | 1 TB |

---

## Partition Limits

- MaxTime: **UNLIMITED** (모든 파티션)
- MaxMemPerNode: **UNLIMITED** (모든 파티션)
- MaxCPUsPerNode: **UNLIMITED** (모든 파티션)
- 실제 제약은 user-level MaxWall (2일)

---

## SLURM 기본 명령어

```bash
# Job 제출 (백그라운드, 터미널 닫아도 OK)
sbatch script.sh

# 파티션 덮어쓰기 (파일 수정 없이)
sbatch --partition=gpu4 script.sh

# Job 상태 확인
squeue -u $USER

# 실시간 모니터링 (2초마다 갱신)
watch -n 2 squeue -u $USER

# Job 취소
scancel JOBID

# 실행 중 시간 변경 (관리자 권한 필요할 수 있음)
scontrol update JobId=JOBID TimeLimit=08:00:00

# 실행 로그 실시간 보기 (RUNNING 상태일 때만)
tail -f path/to/slurm_JOBID.out
```

---

## Notes
- 게이트 노드(gate1)에는 GPU 없음 — `nvidia-smi` 사용 불가
- `sbatch`는 큐 제출 후 즉시 반환, `srun`은 터미널 점유 (끊으면 job 종료)
- MaxJobs=10 제한: DB job 등 다른 작업 고려해서 여유 남겨야 함
