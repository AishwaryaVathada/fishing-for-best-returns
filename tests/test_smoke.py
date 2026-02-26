import os
import tempfile

from fishery_rl.scripts._common import make_env_fn
from fishery_rl.algorithms.sb3_train import train_sb3, load_sb3_policy
from fishery_rl.safety.verification import verify_policy


def test_train_and_verify_smoke():
    env_fn = make_env_fn(backend="toy", horizon=30, seed=0)
    with tempfile.TemporaryDirectory() as d:
        model_path = train_sb3("ppo", env_fn, total_timesteps=2000, out_dir=d, seed=0, device="cpu", algo_kwargs={})
        policy = load_sb3_policy("ppo", model_path)
        report = verify_policy(env_fn, policy, n_rollouts=3, seed=123)
        assert report.n_rollouts == 3
