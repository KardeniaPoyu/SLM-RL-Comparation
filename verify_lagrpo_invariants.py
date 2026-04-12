"""纯张量自检：长度减法项在组内保持 Σ A = 0（与奖励标准化后的 GRPO 优势一致）。"""
import torch

G = 16
torch.manual_seed(0)
rewards = torch.randn(G)
mean_r = rewards.mean()
std_r = rewards.std() + 1e-4
adv = (rewards - mean_r) / std_r
assert abs(adv.sum().item()) < 1e-5

resp_lengths = torch.randint(20, 200, (G,)).float()
mean_len = resp_lengths.mean().clamp(min=1.0)
rel = (resp_lengths - mean_len) / mean_len
assert abs(rel.sum().item()) < 1e-5

beta_len = 0.15
adv_len = adv - beta_len * rel
assert abs(adv_len.sum().item()) < 1e-5

c = 3.0
clipped = torch.clamp(adv_len, -c, c)
centered = clipped - clipped.mean()
assert abs(centered.sum().item()) < 1e-5

print("LAGRPO invariants OK: zero-sum length term, centered clip.")
