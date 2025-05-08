import mujoco 
import numpy as np 
import random as rd 
from numpy.random import uniform
import matplotlib.pyplot as plt 
import json 
import sys 
sys.path.append('../')
from env.manipulator_agent import ManipulatorAgent
from env.mujoco.mujoco_utils import * 

class PlaceEnvrionment:
    def __init__(self, agent, init_pose, RENDER=False, seed=0, eval=False):
        self.agent = agent
        self.init_pose = init_pose
        self.action_space = np.zeros(30,) # end-effector x,y,z * 10
        self.observation_space = np.zeros(16) # (obj position x,y *6 + scenarios = 10+6)
        self._max_episode_steps = 1 
        self.one_hot  = np.zeros((6)) # 6 scenarios
        self.target_position = None
        self.max_z = 0
        self.scenario = 0
        self.RENDER = RENDER    
        self.seed = seed
        self.eval = eval 
        # Set seed 
        np.random.seed(self.seed)
        rd.seed(self.seed)
        self.obj_name_lst = ['obj_box_01', #'obj_box_02','obj_box_03',
                             'obj_cylinder_01','obj_cylinder_02','obj_cylinder_03', 'obj_cylinder_04']
        self.target_name = 'target_box_01'

    def reset(self):
        self.agent.reset(q=np.array([0, -1.57, 1.57, 1.57, 1.57, 0]))
        self.initialize()
        obs = self.get_obs()
        return np.array(obs).flatten()
    
    def step(self, action):
        # Scaling 
        action = action.reshape(3,10)/30
        xs,ys,zs = [],[],[]
        x = 0.6; y = 0.; z = 1.
        self.max_z = 0
        # Relative position moving 
        for i in range(self.action_space.shape[0]//3):
            x += action[0,i]; y += action[1,i]; z+=action[2,i]
            x = np.clip(x, 0.6, 0.83); y = np.clip(y, -0.33, 0.33); z = np.clip(z, 0.9, 1.1)
            xs.append([x]); ys.append([y]); zs.append([z])
        self.max_z = z
        _, q_traj=self.agent.move_place_traj(xs=xs,ys=ys,zs=zs, vel=np.radians(10), HZ=500)
        if self.RENDER:
            tick=0
            while self.agent.is_viewer_alive():
                jntadr = self.agent.model.body('target_box_01').jntadr[0]
                qposadr = self.agent.model.jnt_qposadr[jntadr]
                if tick>len(q_traj)-2:
                    pass
                else:
                    self.agent.data.qpos[qposadr:qposadr+3] = self.agent.get_p_body('tcp_link')[:3]+np.array([0,0,-0.07])
                    self.agent.data.qpos[qposadr+3:qposadr+7] = r2quat(rpy2r(np.radians([0, 0, 0])))
                self.agent.step(q_traj[tick,:],ctrl_idxs=self.agent.idxs_forward+[6])
                self.agent.plot_sphere(p=[1.2,0.0, 0.85],r=0.02,rgba=[1,0,0,1],label=f'Time: [{self.agent.tick * 0.002}]')
                for x,y in zip(xs,ys):
                    self.agent.plot_sphere(p=[x[0],y[0], 0.85],r=0.02,rgba=[230/256, 25/256, 75/256, 1])
                self.agent.render(render_every=10)
                tick+=1
                if tick==len(q_traj)-1:
                    break 
        else:
            tick = 0  
            while tick<=len(q_traj)-1:
                jntadr = self.agent.model.body('target_box_01').jntadr[0]
                qposadr = self.agent.model.jnt_qposadr[jntadr]
                if tick>len(q_traj)-2:
                    pass 
                else:
                    self.agent.data.qpos[qposadr:qposadr+3] = self.agent.get_p_body('tcp_link')[:3]+np.array([0,0,-0.07])
                    self.agent.data.qpos[qposadr+3:qposadr+7] = r2quat(rpy2r(np.radians([0, 0, 0])))
                self.agent.step(q_traj[tick,:],ctrl_idxs=self.agent.idxs_forward+[6])
                tick+=1

        obs = self.get_obs()
        reward, done = self.get_reward()
        
        info = None 
        return obs, reward, done, info
    
    def render(self):
        self.agent.render()

    def get_obs(self):
        position_lst = self.get_position()
        obs = np.concatenate([self.one_hot, np.array(position_lst).flatten()])
        return obs
    
    def get_position(self):
        position_lst = []
        for body_name in self.obj_name_lst:
            body_position = self.agent.get_p_body(body_name)
            position_lst.append(body_position[:2])
        return position_lst

    def get_reward(self):
        done=True
        total_reward = 0
        min_dist = 100
        obs_position_lst = self.get_position()
        obs_rotation_lst = self.get_rotation()
        target_position = self.agent.get_p_body(self.target_name)
        for obs_position, obs_rpy in zip(obs_position_lst, obs_rotation_lst):
            if abs(obs_rpy[0])>0.05:
                return 0, done # Fail 
            else:
                dist = self.euclidean_distance(target_position[:2], obs_position)
                if min_dist>dist:
                    min_dist=dist 
        if min_dist < 0.14:
            return 0, done # Fail
        if min_dist>0.2:
            total_reward=0.7
            total_reward+=(1-self.max_z)*3
            
        else: 
            total_reward+=min_dist*2.5
            if total_reward>0.6:
                total_reward=0.6
            total_reward+=(1-self.max_z)*3

        return total_reward, done

    def get_eval_reward(self):
        done=True
        total_reward = 0
        min_dist = 100
        obs_position_lst = self.get_position()
        obs_rotation_lst = self.get_rotation()
        target_position = self.agent.get_p_body(self.target_name)
        for obs_position, obs_rpy in zip(obs_position_lst, obs_rotation_lst):
            if abs(obs_rpy[0])>0.05:
                done = False
                return 0, done # Fail 
            else:
                dist = self.euclidean_distance(target_position[:2], obs_position)
                if min_dist>dist:
                    min_dist=dist 
        if min_dist < 0.1: # Before touch the object
            done = False 
            return 0, done # Fail
        if min_dist>0.2:
            total_reward=0.7
            total_reward+=(1.0-self.max_z)*0.3
        else: 
            total_reward+=min_dist*2.5
            if total_reward>0.6:
                total_reward=0.6
            total_reward+=(1.0-self.max_z)*0.3
        return total_reward, done

    def eval_step(self, action, save_path=None, file_name=None, epoch=0):
        # Scaling 
        action = action.reshape(3,10)/30
        xs,ys,zs = [],[],[]
        eval_xs, eval_ys, eval_zs = [], [], []
        x = 0.6; y = 0.; z = 1.0
        self.max_z = 0
        # Relative position moving 
        for i in range(self.action_space.shape[0]//3):
            x += action[0,i]; y += action[1,i]; z+=action[2,i]
            x = np.clip(x, 0.6, 0.83); y = np.clip(y, -0.33, 0.33); z = np.clip(z, 0.9, 1.1)
            xs.append([x]); ys.append([y]); zs.append([z])
            eval_xs.append(x); eval_ys.append(y); eval_zs.append(z)
        self.max_z = z 
        _, q_traj=self.agent.move_place_traj(xs=xs,ys=ys,zs=zs, vel=np.radians(10), HZ=500)
        if self.RENDER:
            tick=0
            self.agent.forward(q=np.array([0, -1.57, 1.57, 1.57, 1.57, 0]),joint_idxs=self.agent.idxs_forward)
            while self.agent.is_viewer_alive():
                jntadr = self.agent.model.body('target_box_01').jntadr[0]
                qposadr = self.agent.model.jnt_qposadr[jntadr]
                if tick>len(q_traj)-2:
                    pass
                else:
                    self.agent.data.qpos[qposadr:qposadr+3] = self.agent.get_p_body('tcp_link')[:3]+np.array([0,0,-0.07])
                    self.agent.data.qpos[qposadr+3:qposadr+7] = r2quat(rpy2r(np.radians([0, 0, 0])))
                self.agent.step(q_traj[tick,:],ctrl_idxs=self.agent.idxs_forward+[6])
                self.agent.plot_sphere(p=[1.2,0.0, 0.85],r=0.02,rgba=[1,0,0,1],label=f'Time: [{self.agent.tick * 0.002}]')
                for x,y,z in zip(xs,ys,zs):
                    self.agent.plot_sphere(p=[x[0],y[0], z[0]],r=0.02,rgba=[230/256, 25/256, 75/256, 1])
                self.agent.render(render_every=10)
                tick+=1
                if tick==len(q_traj)-1:
                    break 
        else:
            tick = 0  
            while tick<=len(q_traj)-1:
                jntadr = self.agent.model.body('target_box_01').jntadr[0]
                qposadr = self.agent.model.jnt_qposadr[jntadr]
                if tick>len(q_traj)-2:
                    pass
                else:
                    self.agent.data.qpos[qposadr:qposadr+3] = self.agent.get_p_body('tcp_link')[:3]+np.array([0,0,-0.07])
                    self.agent.data.qpos[qposadr+3:qposadr+7] = r2quat(rpy2r(np.radians([0, 0, 0])))
                self.agent.step(q_traj[tick,:],ctrl_idxs=self.agent.idxs_forward+[6])
                tick+=1

        obs = self.get_obs()
        total_reward, done = self.get_eval_reward()
        if save_path is not None:
            # Save action 
            path = save_path+file_name
            with open(path, "a") as f:
                content = {"epoch":epoch,
                            "x":eval_xs,
                            "y":eval_ys,
                            "z":eval_zs,
                            'reward':total_reward,
                            'done':done}
                f.write(json.dumps(content)+'\n')
        info = None
        return obs, total_reward, done, info

    def get_rotation(self):
        rpy_lst = []
        for body_name in self.obj_name_lst:
            body_rotation= self.agent.get_R_body(body_name)
            body_rpy = r2rpy(body_rotation)
            rpy_lst.append(body_rpy)
        return rpy_lst

    def euclidean_distance(self, point1, point2):
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def manual_reset(self, obs_randxs, obs_randys):
        self.agent.reset(q=np.array([0, -1.57, 1.57, 1.57, 1.57, 0]))
        for obj_name, x, y in zip(self.agent.obj_name_lst, obs_randxs, obs_randys):
            jntadr = self.agent.model.body(obj_name).jntadr[0]
            self.agent.model.joint(jntadr).qpos0[:3] = np.array([x,y,0.74])
        self.agent.reset(q=np.array([0, -1.57, 1.57, 1.57, 1.57, 0]))
        obs = self.get_obs()
        return np.array(obs).flatten()

    def initialize(self):
        if self.eval:
            self.scenario = 3 
        else:
            self.scenario = rd.randint(1,6)
        self.one_hot = np.zeros((6))
        capture_pose= np.array([0, -1.57, 1.57, 1.57, 1.57, 0])
        self.one_hot[self.scenario-1]=1 
        if self.scenario==1: # Obs in Quadrant1,4
            obs_randx1,obs_randy1 = uniform(0.7725, 0.83),uniform(0.04,0.19) 
            obs_randx2,obs_randy2 = uniform(0.7725, 0.83),uniform(0.28,0.38) 
            obs_randx3,obs_randy3 = uniform(0.6, 0.6725),uniform(0.04,0.19)
            obs_randx4,obs_randy4 = uniform(0.6, 0.6725),uniform(0.28,0.38) 
            obs_randx5,obs_randy5 = uniform(0.6, 0.83), uniform(-0.02,0.0)

        elif self.scenario==2: # Obs in Quadrant1,2
            obs_randx1,obs_randy1 = uniform(0.7725, 0.83),uniform(0.04,0.19) 
            obs_randx2,obs_randy2 = uniform(0.7725, 0.83),uniform(0.28,0.38) 
            obs_randx3,obs_randy3 = uniform(0.7725, 0.83),uniform(-0.19,-0.04)
            obs_randx4,obs_randy4 = uniform(0.7725, 0.83),uniform(-0.38,-0.28)
            obs_randx5,obs_randy5 = uniform(0.7125, 0.7325),uniform(-0.38,0.38)

        elif self.scenario==3: # Obs in Quadrant1,3
            obs_randx1,obs_randy1 = uniform(0.7725, 0.83),uniform(0.04,0.19) 
            obs_randx2,obs_randy2 = uniform(0.7725, 0.83),uniform(0.28,0.38) 
            obs_randx3,obs_randy3 = uniform(0.6, 0.6725),uniform(-0.19,-0.04)
            obs_randx4,obs_randy4 = uniform(0.6, 0.6725),uniform(-0.38,-0.28) 
            obs_randx5,obs_randy5 = uniform(0.7125, 0.73),uniform(-0.02,0.02)

        elif self.scenario==4: # Obs in Quadrant2,3
            obs_randx1,obs_randy1 = uniform(0.7725, 0.83),uniform(-0.19,-0.04) 
            obs_randx2,obs_randy2 = uniform(0.7725, 0.83),uniform(-0.38,-0.28) 
            obs_randx3,obs_randy3 = uniform(0.6, 0.6725),uniform(-0.19,-0.04)
            obs_randx4,obs_randy4 = uniform(0.6, 0.6725),uniform(-0.38,-0.28) 
            obs_randx5,obs_randy5 = uniform(0.7, 0.83),uniform(0.0,0.02)
            
        elif self.scenario==5: # Obs in Quadrant 4,3
            obs_randx1,obs_randy1 = uniform(0.6, 0.6725),uniform(0.04,0.19) 
            obs_randx2,obs_randy2 = uniform(0.6, 0.6725),uniform(0.28,0.38) 
            obs_randx3,obs_randy3 = uniform(0.6, 0.6725),uniform(-0.19,-0.04)
            obs_randx4,obs_randy4 = uniform(0.6, 0.6725),uniform(-0.38,-0.28)  
            obs_randx5,obs_randy5 = uniform(0.8125, 0.8325),uniform(-0.38,0.38)

        elif self.scenario==6: # Obs in Quadrant 2,4
            obs_randx1,obs_randy1 = uniform(0.7725, 0.83),uniform(-0.19,-0.04)  
            obs_randx2,obs_randy2 = uniform(0.7725, 0.83),uniform(-0.38,-0.28)  
            obs_randx3,obs_randy3 = uniform(0.6, 0.6725),uniform(0.04,0.19)
            obs_randx4,obs_randy4 = uniform(0.6, 0.6725),uniform(0.28,0.38)   
            obs_randx5,obs_randy5 = uniform(0.7125, 0.7325), uniform(-0.02,0.02)

        obs_randxs = [obs_randx1, obs_randx2, obs_randx3, obs_randx4, obs_randx5]
        obs_randys = [obs_randy1, obs_randy2, obs_randy3, obs_randy4, obs_randy5]
        for obj_name, x, y in zip(self.agent.obj_name_lst, obs_randxs, obs_randys):
            jntadr = self.agent.model.body(obj_name).jntadr[0]
            self.agent.model.joint(jntadr).qpos0[:3] = np.array([x,y,0.74])
        self.agent.reset(q=capture_pose)

    def step_traj_3d(self, xs, ys, zs):
        self.max_z = z = zs[-1][0]
        _, q_traj=self.agent.move_place_traj(xs=xs, ys=ys, zs=zs, vel=np.radians(20), HZ=500)
        if self.RENDER:
            tick=0
            self.agent.forward(q=np.array([0, -1.57, 1.57, 1.57, 1.57, 0]),joint_idxs=self.agent.idxs_forward)
            while self.agent.is_viewer_alive():
                jntadr = self.agent.model.body('target_box_01').jntadr[0]
                qposadr = self.agent.model.jnt_qposadr[jntadr]
                if tick>len(q_traj)-2:
                    pass 
                else:
                    self.agent.data.qpos[qposadr:qposadr+3] = self.agent.get_p_body('tcp_link')[:3]+np.array([0,0,-0.07])
                    self.agent.data.qpos[qposadr+3:qposadr+7] = r2quat(rpy2r(np.radians([0, 0, 0])))
                self.agent.step(q_traj[tick,:],ctrl_idxs=self.agent.idxs_forward+[6])
                self.agent.plot_sphere(p=[1.2,0.0, 0.85],r=0.02,rgba=[1,0,0,1],label=f'Time: [{self.agent.tick * 0.002}]')
                for x,y,z in zip(xs,ys,zs):
                    self.agent.plot_sphere(p=[x[0],y[0], z[0]],r=0.02,rgba=[230/256, 25/256, 75/256, 1])
                self.agent.render(render_every=10)
                tick+=1
                if tick==len(q_traj)-1:
                    break 
        else:
            tick = 0  
            self.agent.forward(q=np.array([0, -1.57, 1.57, 1.57, 1.57, 0]),joint_idxs=self.agent.idxs_forward)
            while tick<=len(q_traj)-1:
                jntadr = self.agent.model.body('target_box_01').jntadr[0]
                qposadr = self.agent.model.jnt_qposadr[jntadr]
                if tick>len(q_traj)-2:
                    pass
                else:
                    self.agent.data.qpos[qposadr:qposadr+3] = self.agent.get_p_body('tcp_link')[:3]+np.array([0,0,-0.07])
                    self.agent.data.qpos[qposadr+3:qposadr+7] = r2quat(rpy2r(np.radians([0, 0, 0])))
                self.agent.step(q_traj[tick,:],ctrl_idxs=self.agent.idxs_forward+[6])
                tick+=1

        obs = self.get_obs()
        reward, done = self.get_reward()
        info = None 
        return obs, reward, done, info

    def eval_step_traj_3d(self, xs, ys, zs, save_path=None, file_name=None, epoch=0):
        self.max_z = z = zs[-1][0]
        _, q_traj=self.agent.move_place_traj(xs=xs, ys=ys, zs=zs, vel=np.radians(20), HZ=500)
        if self.RENDER:
            tick=0
            self.agent.forward(q=np.array([0, -1.57, 1.57, 1.57, 1.57, 0]),joint_idxs=self.agent.idxs_forward)

            while self.agent.is_viewer_alive():
                jntadr = self.agent.model.body('target_box_01').jntadr[0]
                qposadr = self.agent.model.jnt_qposadr[jntadr]
                if tick>len(q_traj)-2:
                    pass
                else:
                    self.agent.data.qpos[qposadr:qposadr+3] = self.agent.get_p_body('tcp_link')[:3]+np.array([0,0,-0.07])
                    self.agent.data.qpos[qposadr+3:qposadr+7] = r2quat(rpy2r(np.radians([0, 0, 0])))
                self.agent.step(q_traj[tick,:],ctrl_idxs=self.agent.idxs_forward+[6])
                self.agent.plot_sphere(p=[1.2,0.0, 0.85],r=0.02,rgba=[1,0,0,1],label=f'Time: [{self.agent.tick * 0.002}]')
                for x,y,z in zip(xs,ys,zs):
                    self.agent.plot_sphere(p=[x[0],y[0], z[0]],r=0.02,rgba=[230/256, 25/256, 75/256, 1])
                self.agent.render(render_every=10)
                tick+=1
                if tick==len(q_traj)-1:
                    break 
        else:
            tick = 0  
            self.agent.forward(q=np.array([0, -1.57, 1.57, 1.57, 1.57, 0]),joint_idxs=self.agent.idxs_forward)

            while tick<=len(q_traj)-1:
                jntadr = self.agent.model.body('target_box_01').jntadr[0]
                qposadr = self.agent.model.jnt_qposadr[jntadr]
                if tick>len(q_traj)-2:
                    pass 
                else:
                    self.agent.data.qpos[qposadr:qposadr+3] = self.agent.get_p_body('tcp_link')[:3]+np.array([0,0,-0.07])
                    self.agent.data.qpos[qposadr+3:qposadr+7] = r2quat(rpy2r(np.radians([0, 0, 0])))
                self.agent.step(q_traj[tick,:],ctrl_idxs=self.agent.idxs_forward+[6])
                tick+=1

        obs = self.get_obs()
        reward, done = self.get_eval_reward()
        eval_xs, eval_ys, eval_zs = [], [], []
        for x,y,z in zip(xs,ys,zs):
            eval_xs.append(x[0]); eval_ys.append(y[0]); eval_zs.append(z[0])
        if save_path is not None:
            # Save action 
            path = save_path+file_name
            with open(path, "a") as f:
                content = {"epoch":epoch,
                            "x":eval_xs,
                            "y":eval_ys,
                            "z":eval_zs,
                            'reward':reward,
                            'done':done}
                f.write(json.dumps(content)+'\n')
        info = None
        return obs, reward, done, info

if __name__=="__main__":
    xml_path = '../env/asset/ur5e_new/scene_ur5e_rg2_obj_place.xml'
        
    RENDER = False 
    if RENDER:
        MODE = 'window'
        USE_MUJOCO_VIEWER = True
    else: 
        MODE = 'offscreen'
        USE_MUJOCO_VIEWER = False
        
    agent = ManipulatorAgent(name='UR5',rel_xml_path=xml_path,VERBOSE=False, USE_MUJOCO_VIEWER=USE_MUJOCO_VIEWER, MODE=MODE)
    agent.reset() # reset
    place_init_pose= np.array([0, -1.57, 1.57, 1.57, 0, 0])
    # Move tables and robot base
    agent.model.body('base_table').pos = np.array([0,0,0.395])
    agent.model.body('front_object_table').pos = np.array([0.38+0.6,0,0])
    agent.model.body('base').pos = np.array([0.18,0,0.79])
    print ("[UR5] parsed.")
    env = PlaceEnvrionment(agent=agent, init_pose=place_init_pose, RENDER=RENDER)