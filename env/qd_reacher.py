import numpy as np 

class QDReacherEnvironment:
    def __init__(self, sim=None, seed=42, PD= None, RENDER=True):
        self.sim    = sim 
        self.RENDER = RENDER
        self.PD     = PD 
        # Set seed 
        np.random.seed(seed)

    def step(self, a, render=False):
        if self.PD is None: # Position Control
            # self.sim.step(ctrl=a, ctrl_idxs=self.sim.rev_joint_idxs)
            for anchor in a:
                self.sim.forward(q=anchor, joint_idxs=self.sim.rev_joint_idxs)
                if self.RENDER or render:
                    self.sim.viewer.render()
        else:
            self.execute_traj(pred_joint_trajectory=a, render=render)
        # Get state 
        state       = self.get_obs() # State
        vec         = self.sim.get_p_body("fingertip") - self.sim.get_p_body("target")
        # print("vec: ", vec)
        # print("finger tip: ", self.sim.get_p_body("fingertip"))
        # print("target: ", self.sim.get_p_body("target"))
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(a).sum()
        reward      = reward_dist #+reward_ctrl 
        if np.linalg.norm(vec)<0.06: done = True 
        else: done = False
        return state, reward, done

    def torque_step(self, a, render=False):
        self.sim.step(ctrl=a, ctrl_idxs=self.sim.rev_joint_idxs)
        if self.RENDER or render:
            self.sim.render()
        # Get state 
        state       = self.get_obs() # State
        vec         = self.sim.get_p_body("fingertip") - self.sim.get_p_body("target")
        # print("vec: ", vec)
        # print("finger tip: ", self.sim.get_p_body("fingertip"))
        # print("target: ", self.sim.get_p_body("target"))
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(a).sum()
        reward      = reward_dist #+reward_ctrl 
        if np.linalg.norm(vec)<0.06: done = True 
        else: done = False
        return state, reward, done
    
    def get_obs(self):
        agent_joint = self.sim.data.qpos[:2] # Reacher Joint0, Joint1 
        return np.concatenate(
            [agent_joint,
            np.cos(agent_joint),
            np.sin(agent_joint),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self.sim.get_p_body("fingertip") - self.sim.get_p_body("target")])

    def execute_traj(self, pred_joint_trajectory, render):
        # Initialize
        self.PD.clear()
        self.sim.data.time = 0 
        for idx, traj_step in enumerate(pred_joint_trajectory):
            if idx==0:
                curr_joint = traj_step
                goal_joint = curr_joint 
            else:
                goal_joint = traj_step 
            diff_joint = curr_joint-goal_joint 
            self.PD.control(diff_joint, self.sim.data.time)
            action     = self.PD.output
            action     = np.clip(action, -1, 1)
            action     = action.flatten()
            self.sim.step(ctrl=action, ctrl_idxs=self.sim.rev_joint_idxs)
            curr_joint = self.sim.data.qpos[:2]
            if self.RENDER or render:
                self.sim.viewer.render()
    
    def reset_model(self):
        self.sim.reset()
        while True:
            goal = np.random.uniform(low=-0.2, high=0.2, size=2)
            if np.linalg.norm(goal) < 0.2 and np.linalg.norm(goal) >0.05:
                break
        self.sim.data.qpos[2:] = goal 
        self.sim.data.qpos[:2] = np.array([0.0, 0.0])
        self.sim.data.qvel[:2] = np.array([0.0, 0.0])
        obs = self.get_obs()
        return obs

    # Check quadrant at the last step 
    # def check_quadrant(self, ee_pose):
    #     if ee_pose[0]>=0 and ee_pose[1]>=0: 
    #         pred_quad_idx=0 
    #     elif ee_pose[0]<0 and ee_pose[1]>0: 
    #         pred_quad_idx=1
    #     elif ee_pose[0]<0 and ee_pose[1]<0: 
    #         pred_quad_idx=2
    #     elif ee_pose[0]>=0 and ee_pose[1]<=0: 
    #         pred_quad_idx=3
    #     return pred_quad_idx

    # def get_reward(self, goal):
    #     ee_pose       = self.sim.get_p_body(body_name='fingertip')
    #     pred_quadrant = self.check_quadrant(ee_pose)
    #     if pred_quadrant==goal:
    #         reward = 2
    #     else: 
    #         target_pose = self.sim.get_p_body(body_name='target')
    #         distance = np.linalg.norm((ee_pose-target_pose))
    #         reward = -distance*5
    #     return reward 