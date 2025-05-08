import json
import mujoco 
import numpy as np 
import random as rd 
import matplotlib.pyplot as plt 
import sys 
sys.path.append('../')
from env.manipulator_agent import ManipulatorAgent
from env.mujoco.mujoco_utils import * 

class ReachEnvrionment:
    def __init__(self, agent, init_pose, RENDER=False, seed=0):
        self.agent = agent
        self.init_pose = init_pose
        self.action_space = np.zeros(30,) # end-effector (x,y,z) * 10
        self.observation_space = np.zeros(5) # Target region: 5
        self._max_episode_steps = 1 
        self.one_hot  = np.zeros((5)) # In case of old qsd, and sac, it should be 4 
        self.target_position = None
        self.RENDER = RENDER
        self.seed = seed
        # Set seed 
        np.random.seed(self.seed)
        rd.seed(self.seed)

    def manual_reset(self, _=None, __=None):
        self.agent.reset()
        obs = self.get_obs()
        return np.array(obs).flatten()
    
    def reset(self):
        self.agent.reset()
        self.initialize()
        obs = self.get_obs()
        return np.array(obs).flatten()
    
    def initialize(self):
        self.one_hot  = np.zeros((5))
        scenario = rd.randint(1,5)  # 1~5 target regions
        self.one_hot[scenario-1] = 1
        target_obj_name = f'obj_region_0{scenario}'
        self.target_position = self.agent.get_p_body(target_obj_name)
        self.agent.reset(q=self.init_pose)

    def step(self, action):
        # Scaling 
        action = action.reshape(3,10)/12 # Normalize action space (x,y,z) to 0~1
        xs,ys,zs = [], [], []
        # Offset of (x,y,z) = [ 0.00123988 -0.13599966  1.07999893]
        x = action[0,0]+0.00124; y = action[1,0]-0.136; z = action[2,0]+1.08
        # Relative position moving 
        for i in range(1, self.action_space.shape[0]//3):
            x += action[0,i]; y += action[1,i]; z += action[2,i]
            x = np.clip(x, -1.2, 1.2); y = np.clip(y, -1.2, 1.2); z = np.clip(z, 0.0, 1.2)
            xs.append([x]); ys.append([y]); zs.append([z])

        _, q_traj=self.agent.move_reach_traj(xs=xs,ys=ys,zs=zs, vel=np.radians(10), HZ=500)
        if self.RENDER:
            tick=0
            while self.agent.is_viewer_alive():
                self.agent.step(q_traj[tick,:],ctrl_idxs=self.agent.idxs_forward)
                self.agent.plot_sphere(p=[0.0,0.0, 1.5],r=0.02,rgba=[1,0,0,1],label=f'Time: [{self.agent.tick * 0.002:.3f}]')
                for x,y,z in zip(xs,ys,zs):
                    self.agent.plot_sphere(p=[x[0],y[0],z[0]],r=0.02,rgba=[230/256, 25/256, 75/256, 1])
                self.agent.render(render_every=10)
                tick+=1
                if tick==len(q_traj)-1:
                    break 
        else:
            tick = 0  
            while tick<=len(q_traj)-1:
                self.agent.step(q_traj[tick,:],ctrl_idxs=self.agent.idxs_forward)
                tick+=1

        obs = self.get_obs()
        # xyz_final = np.array([xs[-1], ys[-1], zs[-1]]).flatten()    # Get final (xyz) position
        tcp_xyz = self.agent.get_p_body("tcp_link")
        reward, done = self.get_reward(target_region=obs, tau=tcp_xyz)
        info = None 
        return obs, reward, done, info

    def eval_step(self, action, save_path=None, file_name=None, epoch=0):
        # Scaling 
        action = action.reshape(3,10)/12 # Normalize action space (x,y,z) to 0~1
        xs,ys,zs = [], [], []
        eval_xs, eval_ys, eval_zs = [0.0012], [-0.136], [1.08]
        # Offset of (x,y,z) = [ 0.00123988 -0.13599966  1.07999893]
        x = action[0,0]+0.00124; y = action[1,0]-0.136; z = action[2,0]+1.08
        # Relative position moving 
        for i in range(1, self.action_space.shape[0]//3):
            x += action[0,i]; y += action[1,i]; z += action[2,i]
            x = np.clip(x, -1.2, 1.2); y = np.clip(y, -1.2, 1.2); z = np.clip(z, 0.0, 1.5)
            xs.append([x]); ys.append([y]); zs.append([z])
            eval_xs.append(x); eval_ys.append(y); eval_zs.append(z)
        _, q_traj=self.agent.move_reach_traj(xs=xs,ys=ys,zs=zs, vel=np.radians(10), HZ=500)
        if self.RENDER:
            tick=0
            while self.agent.is_viewer_alive():
                self.agent.step(q_traj[tick,:],ctrl_idxs=self.agent.idxs_forward)
                self.agent.plot_sphere(p=[0.0,0.0, 1.5],r=0.02,rgba=[1,0,0,1],label=f'Time: [{self.agent.tick * 0.002:.3f}]')
                for x,y,z in zip(xs,ys,zs):
                    self.agent.plot_sphere(p=[x[0],y[0],z[0]],r=0.02,rgba=[230/256, 25/256, 75/256, 1])
                self.agent.render(render_every=10)
                tick+=1
                if tick==len(q_traj)-1:
                    break 
        else:
            tick = 0  
            while tick<=len(q_traj)-1:
                self.agent.step(q_traj[tick,:],ctrl_idxs=self.agent.idxs_forward)
                tick+=1

        obs = self.get_obs()
        # xyz_final = np.array([xs[-1], ys[-1], zs[-1]]).flatten()    # Get final (xyz) position
        tcp_xyz = self.agent.get_p_body("tcp_link")
        reward, done = self.get_eval_reward(target_region=obs, tau=tcp_xyz)
        info = None 
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

        return obs, reward, done, info

    def render(self):
        self.agent.render()

    def get_obs(self):
        obs = np.concatenate([self.one_hot])
        return obs
    
    def get_reward(self, target_region=None, tau=None):
        # Get index of filled with 1
        target_idx = np.where(target_region==1)[0][0]+1
        s = self.agent.get_p_body(f'obj_region_0{target_idx}')
        distance = self.euclidean_distance(s, tau)  # get reward based on distance

        if distance<0.15:
            done = True
            return 2, done
        else:
            done = False
        return -distance*10, done

    def get_eval_reward(self, target_region=None, tau=None):
        # Get index of filled with 1
        target_idx = np.where(target_region==1)[0][0]+1
        s = self.agent.get_p_body(f'obj_region_0{target_idx}')
        distance = self.euclidean_distance(s, tau)  # get reward based on distance

        if distance<0.15:
            done = True
            return 2, done
        else:
            done = False
        return -distance*10, done

    def euclidean_distance(self, point1, point2):
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)

if __name__=="__main__":
    xml_path = '../env/asset/ur5e_new/scene_ur5e_rg2_obj.xml'
        
    RENDER = False 
    if RENDER:
        MODE = 'window'
        USE_MUJOCO_VIEWER = True
    else: 
        MODE = 'offscreen'
        USE_MUJOCO_VIEWER = False
        
    agent = ManipulatorAgent(name='UR5',rel_xml_path=xml_path,VERBOSE=False, USE_MUJOCO_VIEWER=USE_MUJOCO_VIEWER, MODE=MODE)
    agent.reset() # reset
    reach_init_pose= np.array([0, -1.57, 0, -1.57, -3.14, 3.14])
    # Move tables and robot base
    agent.model.body('base').pos = np.array([0.0,0,0.0])
    print ("[UR5] parsed.")
    env = ReachEnvrionment(agent=agent, init_pose=reach_init_pose, RENDER=RENDER)