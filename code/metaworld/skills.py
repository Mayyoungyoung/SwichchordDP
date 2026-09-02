"""Meta-World 场景技能库: 技能控制器、成功判定、数据采集与评测执行器。

场景 A(pick-place-v3): 技能 {reach, grasp, lift, place}(对齐 SWDP.md §7 技能列表)
场景 B(door-open-v3):  技能 {reach, open}

每个技能 = 脚本策略的子阶段控制器(参考 metaworld.policies 源码的解析)。
演示采集时, 先执行前置技能建立状态, 再执行本技能并记录其片段。
"""
import numpy as np
import metaworld


class _FullObsEnv:
    """包装器: metaworld 3.x 的 step() 返回的 obs 会清零 goal 段,
    这里用 env._get_obs() 重新取回全观测, 同时保留底层 info。"""

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self):
        self._env.reset()
        return self._env._get_obs(), {}

    def step(self, a):
        _, rew, term, trunc, info = self._env.step(a)
        return self._env._get_obs(), rew, term, trunc, info

    def close(self):
        self._env.close()


def make_env(scene: str, seed: int = 0):
    """创建并返回场景 env(未 reset)。"""
    ml1 = metaworld.ML1(scene, seed=seed)
    env_cls = ml1.train_classes[scene]
    env = env_cls()
    env.seed(seed)
    task = ml1.train_tasks[seed % len(ml1.train_tasks)]
    env.set_task(task)
    env._partially_observable = False  # 全观测: obs 末尾包含目标位置
    return _FullObsEnv(env)


# ---------------- obs 解析(按场景) ----------------

def parse_pp(obs):
    return {"hand": obs[:3], "grip": obs[3], "puck": obs[4:7],
            "goal": obs[-3:]}


def parse_door(obs):
    return {"hand": obs[:3], "grip": obs[3], "door_handle": obs[4:7],
            "goal": obs[-3:]}


PARSERS = {"pick-place-v3": parse_pp, "door-open-v3": parse_door}


def move(hand, target, p=10.0):
    return p * (target - hand)


# ---------------- 技能控制器 ----------------

class SkillController:
    def __init__(self, env):
        self.env = env

    def act(self, obs):
        raise NotImplementedError

    def done(self, obs):
        return False


class PPReach(SkillController):
    """pick-place: 移动到 puck 上方(夹爪开)。"""

    def act(self, obs):
        o = parse_pp(obs)
        target = o["puck"] + np.array([-0.005, 0, 0.1])
        return np.array([*move(o["hand"], target), 0.0], dtype=np.float32)

    def done(self, obs):
        o = parse_pp(obs)
        return np.linalg.norm(o["hand"][:2] - o["puck"][:2]) < 0.02


class PPGrasp(SkillController):
    """pick-place: 下降到 puck 并闭合夹爪。"""

    def act(self, obs):
        o = parse_pp(obs)
        target = o["puck"] + np.array([-0.005, 0, 0.02])
        grab = 1.0 if np.linalg.norm(o["hand"] - o["puck"]) < 0.08 else 0.0
        return np.array([*move(o["hand"], target), grab], dtype=np.float32)

    def done(self, obs):
        o = parse_pp(obs)
        return o["grip"] < 0.6 and o["puck"][2] < 0.04


class PPLift(SkillController):
    """pick-place: 夹持抬升 puck。"""

    def __init__(self, env, height=0.25):
        super().__init__(env)
        self.height = height

    def act(self, obs):
        o = parse_pp(obs)
        target = o["puck"] + np.array([0.0, 0.0, 0.1])
        target[2] = max(target[2], self.height)
        return np.array([*move(o["hand"], target), 1.0], dtype=np.float32)

    def done(self, obs):
        o = parse_pp(obs)
        return o["puck"][2] > 0.08


class PPPlace(SkillController):
    """pick-place: 平移至 goal 上方 -> 下降到目标高度 -> 松开夹爪。"""

    def act(self, obs):
        o = parse_pp(obs)
        dist_xy = np.linalg.norm(o["hand"][:2] - o["goal"][:2])
        if dist_xy > 0.03:
            # 阶段 1: 平移到 goal 上方(保持夹持)
            grab = 1.0
            target = o["goal"] + np.array([0.0, 0.0, 0.15])
        elif o["hand"][2] - o["goal"][2] > 0.06:
            # 阶段 2: 下降到目标高度(保持夹持)
            grab = 1.0
            target = o["goal"] + np.array([0.0, 0.0, 0.03])
        else:
            # 阶段 3: 松开
            grab = -1.0
            target = o["goal"] + np.array([0.0, 0.0, 0.03])
        return np.array([*move(o["hand"], target), grab], dtype=np.float32)

    def done(self, obs):
        o = parse_pp(obs)
        return (np.linalg.norm(o["puck"][:2] - o["goal"][:2]) < 0.04 and
                o["puck"][2] < 0.06 and o["grip"] > 0.7)


class DoorReach(SkillController):
    """door-open: 移动到门把手前方。"""

    def act(self, obs):
        o = parse_door(obs)
        target = o["door_handle"] + np.array([0.06, 0.02, 0.2])
        return np.array([*move(o["hand"], target, p=25.0), 0.0], dtype=np.float32)

    def done(self, obs):
        o = parse_door(obs)
        return np.linalg.norm(o["hand"] - (o["door_handle"] + np.array([0.06, 0.02, 0.2]))) < 0.06


class DoorOpen(SkillController):
    """door-open: 下降到把手高度后推向把手(开门)。
    对齐官方 SawyerDoorOpenV3Policy: 目标点均基于把手位置向 -x 偏移 0.05。"""

    def act(self, obs):
        o = parse_door(obs)
        handle = o["door_handle"].copy()
        handle[0] -= 0.05
        if np.linalg.norm(o["hand"][:2] - handle[:2]) > 0.12:
            target = handle + np.array([0.06, 0.02, 0.2])
        elif abs(o["hand"][2] - handle[2]) > 0.04:
            target = handle + np.array([0.06, 0.02, 0.0])
        else:
            target = handle
        return np.array([*move(o["hand"], target, p=25.0), 1.0], dtype=np.float32)

    def done(self, obs):
        info = getattr(self.env, "_last_info", None)
        return False


SKILLS = {
    "pick-place-v3": {
        "reach": PPReach, "grasp": PPGrasp, "lift": PPLift, "place": PPPlace,
    },
    "door-open-v3": {"reach": DoorReach, "open": DoorOpen},
}


def get_skill_chain(scene: str, seq):
    """按序列构造技能控制器链。seq 如 ["reach", "grasp", "lift", "place"]。"""
    return [SKILLS[scene][name] for name in seq]
