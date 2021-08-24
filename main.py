import os

import hydra
import omegaconf
import wandb

from src.common.utils import get_best_model
from src.models.cma_es import CMA


def test_agent(cfg):
    agent = hydra.utils.instantiate(cfg.attention_agent,
                                    game=cfg.game,
                                    self_attention=cfg.self_attention,
                                    lstm=cfg.lstm,
                                    log=cfg.log,
                                    _recursive_=False)
    agent.load_model(get_best_model(cfg))
    agent.test_on_screen()
    return 0


def train_agent(cfg):
    wandb.init(**cfg.log.wandb)
    config = wandb.config
    config.sigma0 = cfg.cmaes.sigma0
    config.population_size = cfg.cmaes.popsize

    agent = hydra.utils.instantiate(cfg.attention_agent,
                                    game=cfg.game,
                                    self_attention=cfg.self_attention,
                                    lstm=cfg.lstm,
                                    log=cfg.log,
                                    _recursive_=False)

    if os.path.exists(get_best_model(cfg)):
        agent.load_model(get_best_model(cfg))

    wandb.watch(agent, log='parameters')

    cma = CMA(agent, cfg.cmaes, cfg.log)
    if os.path.exists(cfg.log.log_training + cfg.log.exp_name + '_cma-opt.pkl'):
        cma.train_from_checkpoint(cfg.log.log_training + cfg.log.exp_name + '_cma-opt.pkl')
    else:
        cma.train()


@hydra.main(config_path="conf", config_name="default")
def main(cfg: omegaconf.DictConfig):
    train_agent(cfg)
    # test_agent(cfg)


if __name__ == '__main__':
    main()
