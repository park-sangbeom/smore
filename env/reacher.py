import numpy as np
import random as rd 
import matplotlib.pyplot as plt 
from gym import utils
from gym.envs.mujoco import mujoco_env
import math 
from mujoco_py import MjViewer
import sys 
sys.path.append("..")
from env.pd_controller import PD_Controller
from env.mujoco.mujoco_utils import * 

class CustomReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, seed=42, RENDER=False): 
        self.RENDER = RENDER
        self.PD = PD_Controller()
        self.goal_quadrant = None 
        self.one_hot = np.zeros((4,))
        self.observation_space = np.zeros((16,))
        self.action_space = np.zeros((2,))
        self._max_episode_steps = 50 
        if self.RENDER:
            self.render_mode = "human"
        else:
            self.render_mode = None 
        # Initalize Mujoco
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, "reacher.xml", 2)       
        # Set seed 
        np.random.seed(seed)
        rd.seed(seed)
        # MjViewer for rendering markers 
        if RENDER:
            self.viewer = MjViewer(self.sim)

    def get_obs_joint(self, obs):
        return np.asarray(obs[:2])

    def step(self, a):
        state = self._get_obs()
        ee_pose = self.get_body_com("fingertip")
        pred_quadrant = self.check_quadrant(ee_pose)
        if pred_quadrant==self.goal_quadrant: 
            reward =2; done = True 
        else: 
            vec = self.get_body_com("fingertip") - self.get_body_com("target")
            distance = np.linalg.norm(vec)
            reward = - distance*5 
            done = False 
        self.do_simulation(a, self.frame_skip)
        if self.render_mode == "human":
            self.render()
        return state, reward, done, dict(reward_task=reward)

    ### Overide Mujoco function ###
    def _get_obs(self):
        theta = self.sim.data.qpos[:2]
        return np.concatenate(
            [self.one_hot,
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")])

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def initialize(self):
        """ Initialize """
        # self.sim.data.qpos[:2] = np.array([0,0])
        """ Random State """
        self.one_hot  = np.zeros((4,))
        self.goal_quadrant = rd.randint(1,4)
        self.one_hot[self.goal_quadrant-1]=1 
        if self.goal_quadrant == 1: 
            self.goal = np.array([0.1, 0.1])
        elif self.goal_quadrant==2: 
            self.goal = np.array([-0.1, 0.1])
        elif self.goal_quadrant==3: 
            self.goal = np.array([-0.1, -0.1])
        elif self.goal_quadrant==4: 
            self.goal = np.array([0.1, -0.1])


    # Check quadrant at the last step 
    def check_quadrant(self, ee_pose):
        if ee_pose[0]>=0 and ee_pose[1]>=0: 
            pred_quadrant=1 
        elif ee_pose[0]<0 and ee_pose[1]>0: 
            pred_quadrant=2
        elif ee_pose[0]<0 and ee_pose[1]<0: 
            pred_quadrant=3
        elif ee_pose[0]>=0 and ee_pose[1]<=0: 
            pred_quadrant=4
        return pred_quadrant

    def reset_model(self):
        self.initialize()
        qpos = (
            self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
            + self.init_qpos
        )
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        self.set_state(qpos, qvel)
        return self._get_obs()

