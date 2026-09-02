"""单技能 DP 执行质量验证: 脚本控制器建立前置状态, DP 执行目标技能。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, torch, h5py
from swdp.policy import SkillDP
from skills import make_env, SKILLS
from eval_compose import skill_success

DEVICE = "cuda"
dp = SkillDP.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../results/metaworld/models/dp_pick-place-v3.pt"), DEVICE)
dp.eval()
with h5py.File(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../results/metaworld/data/pick-place-v3.h5"), "r") as f:
    obs_mean = f["obs_mean"][:]; obs_std = f["obs_std"][:]
    act_mean = f["act_mean"][:]; act_std = f["act_std"][:]
skill_ids = {"reach": 0, "grasp": 1, "lift": 2, "carry": 3, "place": 4}
scene = "pick-place-v3"

for name, n in [("reach", 30), ("grasp", 25), ("lift", 25), ("carry", 30), ("place", 20)]:
    succ = []
    for ep in range(8):
        env = make_env(scene, seed=700 + ep)
        obs, _ = env.reset()
        pre = {"reach": [], "grasp": ["reach"], "lift": ["reach", "grasp"],
               "carry": ["reach", "grasp", "lift"],
               "place": ["reach", "grasp", "lift", "carry"]}[name]
        for p in pre:
            c = SKILLS[scene][p](env)
            for _ in range(30 if p == "reach" else 25):
                obs, *_ = env.step(c.act(obs))
        s = torch.zeros(1, 5, device=DEVICE)
        s[0, skill_ids[name]] = 1

        def sample(o):
            return dp.sample(torch.from_numpy((o - obs_mean) / obs_std).float().to(DEVICE).unsqueeze(0), s, n_steps=16)

        chunk = sample(obs)
        step = 0
        for t in range(n):
            a = (chunk[0, step].cpu().numpy() * act_std) + act_mean
            obs, *_ = env.step(a)
            step += 1
            if step >= 8:
                chunk = sample(obs)
                step = 0
        succ.append(skill_success(scene, name, env, obs, None))
        env.close()
    print(f"single-skill {name}: {np.mean(succ):.2f}")
