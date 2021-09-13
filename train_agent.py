import os

import hydra
import omegaconf
import wandb

from src.models.cma_es import CMA


@hydra.main(config_path="conf", config_name="config")
def train(cfg: omegaconf.DictConfig):
    cfg = cfg.agent
    wandb.init(**cfg.log.wandb, resume=True)
    config = wandb.config
    config.sigma0 = cfg.cmaes.sigma0
    config.population_size = cfg.cmaes.popsize
    config.surviving_penalty = cfg.attention_agent.surviving_penalty

    agent = hydra.utils.instantiate(cfg.attention_agent,
                                    game=cfg.game,
                                    self_attention=cfg.self_attention,
                                    lstm=cfg.lstm,
                                    log=cfg.log,
                                    _recursive_=False)

    wandb.watch(agent, log='parameters')

    cma = CMA(agent, cfg.cmaes, cfg.log)
    if os.path.exists(cfg.log.log_training + cfg.log.agent_name + '_cma-opt.pkl'):
        cma.train_from_checkpoint(cfg.log.log_training + cfg.log.agent_name + '_cma-opt.pkl')
    else:
        cma.train()


if __name__ == '__main__':
    train()
