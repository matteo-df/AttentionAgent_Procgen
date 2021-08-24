import os
import pickle

import cma
import numpy as np
import torch
import wandb


class CMA:
    """CMA algorithm."""
    def __init__(self, agent_model, cma_cfg, log_cfg):

        self.agent = agent_model
        self.log = log_cfg
        self.population = None
        self.cmaes = cma.CMAEvolutionStrategy(x0=self.agent.get_params(), sigma0=cma_cfg.sigma0,
                                              inopts={'popsize': cma_cfg.popsize,
                                                      'randn': cma_cfg.randn})

    def train(self):
        pkl_path = self.log.log_training + self.log.exp_name + '_cma-opt.pkl'
        steps = 0
        with torch.no_grad():
            # while not self.cmaes.stop():
            for _ in range(500):
                rewards = self.agent.evaluate_population(self.get_population())
                self.evolve(rewards)
                wandb.log({"best_reward": np.max(rewards), "mean_reward": np.mean(rewards)})

                if steps % 5 == 0:
                    self.cmaes.disp()
                if steps % 50 == 0:
                    pickle.dump(self.cmaes, open(pkl_path, 'wb'))
                    self.update_agent()
                    self.agent.save_model()
                steps += 1
        pickle.dump(self.cmaes, open(pkl_path, 'wb'))
        print('Agent Optimization with CMA-ES saved in', pkl_path)

    def train_from_checkpoint(self, chekcpoint_file):
        assert os.path.exists(chekcpoint_file), 'Checkpoint file not found!'

        del self.cmaes
        self.cmaes = pickle.load(open(chekcpoint_file, 'rb'))
        print('Agent Optimization with CMA-ES resumed from checkpoint!')
        self.train()

    def get_population(self):
        self.population = np.array(self.cmaes.ask())
        return self.population

    def evolve(self, rewards):
        self.cmaes.tell(self.population, -1 * rewards)

    def get_best_parameters(self):
        return self.cmaes.result.xbest

    def update_agent(self):
        self.agent.set_params(self.get_best_parameters())
