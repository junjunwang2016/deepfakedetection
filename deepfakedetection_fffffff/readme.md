单机多卡
torchrun --nproc_per_node=4 train.py \
  --overrides train.ema.enabled=true model.backbone=convnext_base_384_in22ft1k \
              loss.name=focal loss.class_balanced_alpha=true



# 双流：RGB + F3（FAD+LFS），启用 Self-Blend（alpha 融合），DDP 4 卡
torchrun --nproc_per_node=4 train.py \
  --overrides model.type=rgb_f3 model.freq.enabled=true model.freq.mode=Both \
              aug.sbi_enabled=true aug.sbi_prob=0.3 aug.sbi_poisson=false
