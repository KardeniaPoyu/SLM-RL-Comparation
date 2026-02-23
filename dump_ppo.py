from trl import PPOConfig
import inspect

with open("config_args.txt", "w") as f:
    for k in inspect.signature(PPOConfig).parameters.keys():
        f.write(k + "\n")
