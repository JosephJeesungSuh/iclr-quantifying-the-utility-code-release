"""User-simulator GRPO entry point for the pinned upstream SkyRL release."""
import asyncio
import ray
import hydra
from omegaconf import DictConfig
from skyrl_train.utils import initialize_ray
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg
from skyrl_gym.envs import register


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    for name, cls in [("userlm", "UserLMMultiTurnEnv"),
                      ("userlm_baseline", "UserLMBaselineEnv"),
                      ("userlm_baseline_mixture", "UserLMBaselineMixtureEnv")]:
        register(id=name, entry_point=f"examples.userlm_paper.env:{cls}")
    if cfg.pop("userlm_evaluate", False):
        from skyrl_train.entrypoints.main_generate import EvalOnlyEntrypoint
        return asyncio.run(EvalOnlyEntrypoint(cfg).run())
    BasePPOExp(cfg).run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig):
    if cfg.get("userlm_evaluate", False):
        from skyrl_train.utils.utils import validate_generator_cfg
        validate_generator_cfg(cfg)
    else:
        validate_cfg(cfg)
    initialize_ray(cfg)
    print(ray.get(skyrl_entrypoint.remote(cfg)))


if __name__ == "__main__":
    main()
