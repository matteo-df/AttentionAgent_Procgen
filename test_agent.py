import hydra
import omegaconf

from src.common.utils import get_model


@hydra.main(config_path="conf", config_name="config")
def main(cfg: omegaconf.DictConfig):
    # Selecting the agent config
    agent_cfg = cfg.agent

    # Instantiating the agent
    agent = hydra.utils.instantiate(agent_cfg.attention_agent,
                                    game=agent_cfg.game,
                                    self_attention=agent_cfg.self_attention,
                                    lstm=agent_cfg.lstm,
                                    log=agent_cfg.log,
                                    _recursive_=False)
    # Loading pretrained model
    agent.load_model(get_model(agent_cfg))

    if cfg.save_gif_with_attention_patches:
        agent.save_gif_with_attention_patches()
    if cfg.render:
        agent.play_on_screen(n_episodes=cfg.n_episodes)
    else:
        agent.test(n_episodes=cfg.n_episodes)


if __name__ == '__main__':
    main()
