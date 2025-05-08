import gym
import numpy as np
from tqdm import tqdm
# import mujoco_py
import os 
import torch 
import sys 
sys.path.append('..')
from model.diayn.sac import SACAgent
from model.diayn.config import get_params
from model.utils.utils import np2torch, torch2np, get_runname, get_diversity
from env.sweep import SweepEnvrionment
from env.place import PlaceEnvrionment
from env.manipulator_agent import ManipulatorAgent
import json 
import wandb 

def concat_state_latent(s, z_, n):
    z_one_hot = np.zeros(n)
    z_one_hot[z_] = 1
    return np.concatenate([s, z_one_hot])

if __name__ == "__main__":
    params = get_params()
    RENDER = False 
    if RENDER:
        MODE = 'window'
        USE_MUJOCO_VIEWER = True
    else: 
        MODE = 'offscreen'
        USE_MUJOCO_VIEWER = False
    if params['env_name'] == 'sweep':
        xml_path = './env/asset/ur5e_new/scene_ur5e_rg2_obj.xml'
        agent = ManipulatorAgent(rel_xml_path=xml_path,VERBOSE=False, USE_MUJOCO_VIEWER=USE_MUJOCO_VIEWER, MODE=MODE)
        agent.reset() # reset
        # Move tables and robot base
        agent.model.body('base_table').pos = np.array([0,0,0.395])
        agent.model.body('front_object_table').pos = np.array([0.38+0.6,0,0])
        agent.model.body('base').pos = np.array([0.18,0,0.79])
        init_pose= np.array([-0.73418, -1.08485, 2.7836, -1.699, 0.8366, 0])
        env = SweepEnvrionment(agent=agent, init_pose=init_pose, RENDER=RENDER, seed=params['seed'])
    
    elif params['env_name'] == 'place':
        xml_path = './env/asset/ur5e_new/scene_ur5e_rg2_obj_place.xml'
        agent = ManipulatorAgent(rel_xml_path=xml_path,VERBOSE=False, USE_MUJOCO_VIEWER=USE_MUJOCO_VIEWER, MODE=MODE)
        agent.reset() # reset
        # Move tables and robot base
        agent.model.body('base_table').pos = np.array([0,0,0.395])
        agent.model.body('front_object_table').pos = np.array([0.38+0.6,0,0])
        agent.model.body('base').pos = np.array([0.18,0,0.79])
        init_pose= np.array([0, -1.57, 1.57, 1.57, 1.57, 0])
        env = PlaceEnvrionment(agent=agent, init_pose=init_pose, RENDER=RENDER, seed=params['seed'])
    
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    action_bounds = [-1, 1]
    SAVE_WEIGHT_PATH = "./weights/{}/diayn/".format(params['env_name'])+"final_ver_skill30"
    os.makedirs(SAVE_WEIGHT_PATH, exist_ok=True) 
    params.update({"n_states": n_states,
                   "n_actions": n_actions,
                   "action_bounds": action_bounds})
    print("params:", params)

    p_z = np.full(params["n_skills"], 1 / params["n_skills"])
    agent = SACAgent(p_z=p_z, **params)

    if params["do_train"]:
        min_episode = 0
        last_logq_zs = 0
        np.random.seed(params["seed"])
        print("Training from scratch.")

        for episode in tqdm(range(1 + min_episode, params["max_n_episodes"] + 1)):
            z = np.random.choice(params["n_skills"], p=p_z)
            state = env.reset()
            state = concat_state_latent(state, z, params["n_skills"])
            episode_reward = 0
            logq_zses = []

            max_n_steps = 1 #env.spec.max_episode_steps 
            for step in range(1, 1 + max_n_steps):
                action = agent.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                next_state = concat_state_latent(next_state, z, params["n_skills"])
                agent.store(state, z, done, action, next_state)
                logq_zs = agent.train()
                if logq_zs is None:
                    logq_zses.append(last_logq_zs)
                else:
                    logq_zses.append(logq_zs)
                episode_reward += reward
                state = next_state
                if done:
                    break
            if agent.memory.__len__() >= params["batch_size"] and episode%1000==0:
                torch.save(agent.policy_network.state_dict(), SAVE_WEIGHT_PATH+'/pi_{}.pth'.format(episode))
                torch.save(agent.q_value_network1.state_dict(), SAVE_WEIGHT_PATH+'/q1_value_{}.pth'.format(episode))
                torch.save(agent.q_value_network2.state_dict(), SAVE_WEIGHT_PATH+'/q2_value_{}.pth'.format(episode))
                torch.save(agent.value_network.state_dict(), SAVE_WEIGHT_PATH+'/value_{}.pth'.format(episode))
                torch.save(agent.value_target_network.state_dict(), SAVE_WEIGHT_PATH+'/value_target_{}.pth'.format(episode))

    if params["do_eval"]:
        DIV = False 
        file_path = './weights/{}/diayn/'.format(params['env_name'])+'diayn_seed0_final_ver_benchmark2'+'/pi_20000.pth'
        agent.policy_network.load_state_dict(torch.load(file_path, map_location='cpu'))
        save_path = './eval/{}/diayn/'.format(params['env_name'])
        file_name = 'test.json'
        path = save_path+file_name
        os.makedirs(save_path, exist_ok=True) 
        cnt=0
        s = env.reset()
        if params['env_name'] == 'sweep':
            obs_randxys = s[3:].reshape(6,-1)
            obs_randxs, obs_randys = obs_randxys[:,0],obs_randxys[:,1]
        elif params['env_name'] == 'place':
            obs_randxys = s[6:].reshape(5,-1)
            obs_randxs, obs_randys = obs_randxys[:,0],obs_randxys[:,1]    

        for i in range(10):
            if not DIV:
                s = env.reset()
            else:
                env.manual_reset(obs_randxs, obs_randys)
            z = np.random.randint(0,20)
            # if z==11 or z==1 or z==4:
            #     z = np.random.randint(0,20)

            # for z in range(params["n_skills"]):
            s_ = concat_state_latent(s, z, params["n_skills"])
            env.render()
            action = agent.choose_action(s_)
            _, r, done, _ = env.eval_step(action=action,save_path=save_path, file_name=file_name, epoch=i+1)
            # if done:
            #     break
            print(f"skill: {z}, reward:{r}")
            if done: 
                cnt+=1 
        with open(path, "a") as f:
            content = {"SR":cnt/10}
            f.write(json.dumps(content)+'\n')
        print("SR:", cnt/10)

    if params["do_benchmark"]:
        pi_file_path = './weights/place/diayn/'+"diayn_seed0_final_ver"+'/pi_10000.pth'
        q1_value_file_path = './weights/place/diayn/'+'diayn_seed0_final_ver'+'/q1_value_10000.pth'
        q2_value_file_path = './weights/place/diayn/'+'diayn_seed0_final_ver'+'/q2_value_10000.pth'
        value_file_path = './weights/place/diayn/'+'diayn_seed0_final_ver'+'/value_10000.pth'
        value_target_file_path = './weights/place/diayn/'+'diayn_seed0_final_ver'+'/value_target_10000.pth'

        agent.policy_network.load_state_dict(torch.load(pi_file_path, map_location='cpu'))
        agent.q_value_network1.load_state_dict(torch.load(q1_value_file_path, map_location='cpu'))
        agent.q_value_network2.load_state_dict(torch.load(q2_value_file_path, map_location='cpu'))
        agent.value_network.load_state_dict(torch.load(value_file_path, map_location='cpu'))
        agent.value_target_network.load_state_dict(torch.load(value_target_file_path, map_location='cpu'))

        project_name = '{}'.format(params['env_name'])
        runname = '{}_benchmark'.format(params['env_name'])
        wandb.init(project = '{}'.format(params['env_name']))
        wandb.run.name = runname 

        min_episode = 0
        last_logq_zs = 0
        np.random.seed(params["seed"])
        print("Training from scratch.")

        for episode in tqdm(range(1 + min_episode, params["max_n_episodes"] + 1)):
            z = np.random.choice(params["n_skills"], p=p_z)
            state = env.reset()
            state = concat_state_latent(state, z, params["n_skills"])
            episode_reward = 0
            logq_zses = []

            max_n_steps = min(params["max_episode_len"], env.spec.max_episode_steps)
            # max_n_steps = 1
            for step in range(1, 1 + max_n_steps):
                action = agent.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                next_state = concat_state_latent(next_state, z, params["n_skills"])
                agent.benchmark_store(state, z, done, action, next_state, reward)
                agent.benchmark_train()
                episode_reward += reward
                state = next_state

            if agent.benchmark_memory.__len__() >= params["batch_size"] and episode%10==0:
                torch.save(agent.policy_network.state_dict(), './weights/place/diayn/'+'diayn_seed0_final_ver_benchmark2'+'/pi_{}.pth'.format(episode))
                print(f"Episode.{episode} skill: {z}, episode reward:{episode_reward:.1f}")
                # Evaluation 
                eval_cnt = 0
                action_lst = []
                for i in range(10):
                    z = np.random.choice(params["n_skills"], p=p_z)
                    state = env.reset()
                    state = concat_state_latent(state, z, params["n_skills"])

                    action = agent.choose_action(state)
                    next_state, reward, done, _ = env.eval_step(action) 
                    action_lst.append(action)
                    if done: 
                        eval_cnt+=1 
                diversity = get_diversity(action_lst, params['env_name'])
                wandb.log({"SR":eval_cnt/10,
                           "Diversity":diversity}, step=episode)