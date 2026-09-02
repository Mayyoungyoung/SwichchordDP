"""蒸馏学生(1/2/4 步)单技能成功率评测。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, torch, h5py
from swdp.distil import ConsistencyStudent
from skills import make_env, SKILLS
from eval_compose import skill_success

DEVICE = "cuda"
ckpt = torch.load(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../results/metaworld/models/dp_pick-place-v3_cd.pt"), map_location=DEVICE)
student = ConsistencyStudent(**ckpt["cfg"], device=DEVICE)
student.load_state_dict(ckpt["model"])
student.eval()
with h5py.File("../../results/metaworld/data/pick-place-v3.h5", "r") as f:
    obs_mean = f["obs_mean"][:]; obs_std = f["obs_std"][:]
    act_mean = f["act_mean"][:]; act_std = f["act_std"][:]
skill_ids = {"reach": 0, "grasp": 1, "lift": 2, "carry": 3, "place": 4}
scene = "pick-place-v3"
PRE = {"reach": [], "grasp": ["reach"], "lift": ["reach", "grasp"],
       "carry": ["reach", "grasp", "lift"],
       "place": ["reach", "grasp", "lift", "carry"]}
N = {"reach": 30, "grasp": 30, "lift": 25, "carry": 30, "place": 25}

for n_steps in [1, 2, 4]:
    print(f"=== student {n_steps}-step ===")
    for name, n in [("reach", 30), ("grasp", 30), ("lift", 25),
                    ("carry", 30), ("place", 25)]:
        succ = []
        for ep in range(8):
            env = make_env(scene, seed=700 + ep)
            obs, _ = env.reset()
            for p in PRE[name]:
                c = SKILLS[scene][p](env)
                for _ in range(30 if p in ("reach", "grasp") else 25):
                    obs, *_ = env.step(c.act(obs))
            s = torch.zeros(1, 5, device=DEVICE)
            s[0, skill_ids[name]] = 1

            def sample(o):
                return student.sample(
                    torch.from_numpy((o - obs_mean) / obs_std).float().to(DEVICE).unsqueeze(0),
                    s, n_steps=n_steps)
            chunk = sample(obs)
            step = 0
            for t in range(n):
                a = np.clip((chunk[0, step].cpu().numpy() * act_std) + act_mean, -1, 1)
                obs, *_ = env.step(a)
                step += 1
                if step >= 8:
                    chunk = sample(obs)
                    step = 0
            succ.append(skill_success(scene, name, env, obs, None))
            env.close()
        print(f"  {name}: {np.mean(succ):.2f}")
