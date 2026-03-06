
import yaml
with open("/gpfs/home1/zye614/projects/DPO_PRISM/configs/dpo_v3_base.yaml") as f:
	cfg = yaml.safe_load(f)


dtype 
bf16 : torch.bfloat16
fp16 : torch.float16
fp32 : torch.float32