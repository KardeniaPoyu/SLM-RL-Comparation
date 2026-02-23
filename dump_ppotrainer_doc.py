import trl
with open("dump_ppotrainer2.txt", "w", encoding="utf-8") as f:
    f.write(trl.PPOTrainer.__init__.__doc__ or "No docstring for __init__, getting class doc:\n" + trl.PPOTrainer.__doc__ or "No docstring")
