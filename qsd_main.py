from tqdm import tqdm
import matplotlib.pyplot as plt
import threading
import os
import time
import numpy as np
import torch
from torch import nn
from tensorboardX import SummaryWriter
import sys 
sys.path.append('..')
from model.qsd.TD3_agent import TD3_Agent
from model.qsd.flag import build_flag
from model.qsd.replayer_buffer import ReplayBuffer
from model.qsd.model_manager import ModelManager
from model.qsd.utils import *
import matplotlib
matplotlib.use('Agg')
from env.sweep import SweepEnvrionment
from env.place import PlaceEnvrionment
from env.manipulator_agent import ManipulatorAgent
import wandb 

class Population_Trainer(object):
    def __init__(self, env, args):

        # base config
        self.args = args
        self.traj_len = args.traj_len
        self.env = env
        self.population_size = args.population_size
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.env_name = args.env_name
        print("=" * 10, self.state_dim, self.action_dim)
        self.results_dim = self.action_dim + 2
        self.device = args.device

        self.lstm_hidden_dim = args.lstm_hidden_dim 
        self.init_agents()

        # log
        self.prefix = self.agent_pools[0].prefix
        writer_path = os.path.join(self.prefix, f"tb_summary/learner")
        self.writer = SummaryWriter(writer_path)
        self.writer_freq = 100

        # train
        self.total_iter = 200000
        self.max_timesteps = self.env._max_episode_steps * \
            5  # max timesteps each iteration
        self.warmup_timesteps = [self.max_timesteps] * \
            self.population_size  # warm up timesteps
        self.batch_size = args.batch_size
        self.save_freq = args.save_freq
        agent_writers = [agent.writer for agent in self.agent_pools]
        self.model_manager = ModelManager(
            self.agent_pools, self.args, self.writer, agent_writers, self.writer_freq, self.device)
        self.reward_threshold_to_guide = args.reward_threshold_to_guide
        wandb.init(project = 'sweep')
        wandb.run.name = 'qsd_seed0_new2'  

    def init_agents(self):
        kwargs = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "max_action": float(1),
            "lstm_hidden_dim": self.lstm_hidden_dim,
            "args": self.args,
            "device": self.device,
            "env_name":self.env_name
        }

        self.agent_pools = []
        if self.args.policy == "TD3":
            for agent_id in range(self.population_size):
                kwargs["agent_id"] = agent_id
                agent = TD3_Agent(**kwargs)
                self.agent_pools.append(agent)
        else:
            raise NotImplementedError

        self.replay_buffer_pools = []
        for agent_id in range(self.population_size):
            rb = ReplayBuffer(self.state_dim, self.action_dim, self.lstm_hidden_dim,
                              self.results_dim, self.traj_len, self.device)
            self.replay_buffer_pools.append(rb)

    def run(self):
        actor = threading.Thread(target=self.agents_play)
        learner = threading.Thread(target=self.train_agents)
        actor.start()
        learner.start()
        actor.join()
        learner.join()

    def agents_play(self,):
        thread_list = []
        for agent_id in range(self.population_size):
            thread_ = threading.Thread(
                target=self.single_agent_play, args=(agent_id,))
            thread_list.append(thread_)
        for thread_ in thread_list:
            thread_.start()
        for thread_ in thread_list:
            thread_.join()

    def single_agent_play(self, agent_id):
        for iter_ in range(self.total_iter):
            self.agent_pools[agent_id].play_game(iter_,
                                                 self.warmup_timesteps[agent_id],
                                                 self.max_timesteps,
                                                 self.replay_buffer_pools[agent_id],
                                                 self.traj_len)
            self.warmup_timesteps[agent_id] = 0

    def train_agents(self, ):
        while sum(self.warmup_timesteps) > 0:
            time.sleep(0.1)
            continue

        for iter_ in (range(self.total_iter)):
            train_data = self.get_train_data()
            self.model_manager.update(train_data, iter_)
            for agent in self.agent_pools:
                agent.total_it += 1

            if iter_ % self.writer_freq == 0:
                for agent in self.agent_pools:
                    agent.writer.add_scalar(
                        "Reward", agent.running_reward, iter_)

            if iter_ % self.save_freq == 0:
                self.save_population_models(iter_)
            wandb.log({"train avg reward":agent.running_reward}, step=iter_)   

    def get_train_data(self,):
        # behavior_descriptor
        behavior_descriptor = []
        for agent_id in range(self.population_size):
            samples = self.replay_buffer_pools[agent_id].sample_terminate(
                self.batch_size)
            results, not_done = samples[-5], samples[-4]

            results = results[(not_done == 0).squeeze()].mean(0)
            behavior_descriptor.append(results)

        samples_all = []
        for agent_id in range(self.population_size):
            samples = self.replay_buffer_pools[agent_id].sample(
                self.batch_size)
            samples_all.append(samples)

        agent_rewards = [agent.running_reward for agent in self.agent_pools]
        agent_rewards = np.array(agent_rewards)
        best_agent_index = np.argmax(agent_rewards)
        best_reward = agent_rewards[best_agent_index]

        guide_coef = np.exp(- agent_rewards / best_reward)
        is_guided = agent_rewards < (
            best_reward * self.reward_threshold_to_guide)
        guide_coef *= is_guided

        guidance_sample = self.replay_buffer_pools[best_agent_index].sample(
            self.batch_size)

        return behavior_descriptor, samples_all, guidance_sample, guide_coef

    def save_population_models(self, iter_):
        save_path = os.path.join(self.prefix, "model")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = os.path.join(save_path, f"checkpoint_{iter_}.pt")
        torch.save(self.model_manager.models.state_dict(), filename)

    def load_population_models(self, iter_):
        save_path = os.path.join(self.prefix, "model")
        filename = os.path.join(save_path, f"checkpoint_{iter_}.pt")
        self.model_manager.models.load_state_dict(
            torch.load(filename, map_location=self.device))


if __name__ == "__main__":
    args = build_flag()
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    RENDER = False 
    if RENDER:
        MODE = 'window'
        USE_MUJOCO_VIEWER = True
    else: 
        MODE = 'offscreen'
        USE_MUJOCO_VIEWER = False
        
    if args.env_name == 'sweep':
        xml_path = './env/asset/ur5e_new/scene_ur5e_rg2_obj.xml'
        agent = ManipulatorAgent(rel_xml_path=xml_path,VERBOSE=False, USE_MUJOCO_VIEWER=USE_MUJOCO_VIEWER, MODE=MODE)
        agent.reset() # reset
        # Move tables and robot base
        agent.model.body('base_table').pos = np.array([0,0,0.395])
        agent.model.body('front_object_table').pos = np.array([0.38+0.6,0,0])
        agent.model.body('base').pos = np.array([0.18,0,0.79])
        init_pose= np.array([-0.73418, -1.08485, 2.7836, -1.699, 0.8366, 0])
        env = SweepEnvrionment(agent=agent, init_pose=init_pose, RENDER=RENDER, seed=args.seed)
    
    elif args.env_name == 'place':
        xml_path = './env/asset/ur5e_new/scene_ur5e_rg2_obj_place.xml'
        agent = ManipulatorAgent(rel_xml_path=xml_path,VERBOSE=False, USE_MUJOCO_VIEWER=USE_MUJOCO_VIEWER, MODE=MODE)
        agent.reset() # reset
        # Move tables and robot base
        agent.model.body('base_table').pos = np.array([0,0,0.395])
        agent.model.body('front_object_table').pos = np.array([0.38+0.6,0,0])
        agent.model.body('base').pos = np.array([0.18,0,0.79])
        init_pose= np.array([0, -1.57, 1.57, 1.57, 0, 0])
        env = PlaceEnvrionment(agent=agent, init_pose=init_pose, RENDER=RENDER, seed=args.seed)

    population_trainer = Population_Trainer(env, args)
    population_trainer.run()
    """
    How to run:
    python3 qsd_main.py --env 'Reacher-v2' --exp_name Reacher --population_size 10 --save_model --loss_weight_lambda 5 --loss_weight_guide 0 > Reacher_5_0.log
    python3 qsd_main.py --env 'Sweep' --exp_name sweep --population_size 10 --save_model --loss_weight_lambda 5 --loss_weight_guide 0 > sweep_5_0.log

    """
