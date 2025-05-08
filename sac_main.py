import numpy as np 
import torch 
import wandb 
import argparse
import os 
import sys 
sys.path.append('../')
from model.sac.actor_critic import ActorClass, CriticClass, get_target
from model.sac.buffer import ReplayBuffer
from model.utils.utils import np2torch, torch2np, get_runname, get_diversity
import numpy as np
from env.sweep import SweepEnvrionment
from env.place import PlaceEnvrionment
from env.reach import ReachEnvrionment
from env.manipulator_agent import ManipulatorAgent

def main(args):
    # Set random seed 
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
     # Set logger 
    runname = get_runname() if args.runname=='None' else args.runname
    if args.WANDB:
        wandb.init(project = args.pname)
        wandb.run.name = runname   
    # Make a path directory of save weight folder 
    SAVE_WEIGHT_PATH = "./weights/{}/sac/".format(args.env_name)+args.runname
    os.makedirs(SAVE_WEIGHT_PATH, exist_ok=True) 
    
    """ Sweep"""
    RENDER = args.RENDER 
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
        env = SweepEnvrionment(agent=agent, init_pose=init_pose, RENDER=RENDER, seed=args.random_seed)
    
    elif args.env_name == 'place':
        xml_path = './env/asset/ur5e_new/scene_ur5e_rg2_obj_place.xml'
        agent = ManipulatorAgent(rel_xml_path=xml_path,VERBOSE=False, USE_MUJOCO_VIEWER=USE_MUJOCO_VIEWER, MODE=MODE)
        agent.reset() # reset
        # Move tables and robot base
        agent.model.body('base_table').pos = np.array([0,0,0.395])
        agent.model.body('front_object_table').pos = np.array([0.38+0.6,0,0])
        agent.model.body('base').pos = np.array([0.18,0,0.79])
        init_pose= np.array([0, -1.57, 1.57, 1.57, 1.57, 0])
        env = PlaceEnvrionment(agent=agent, init_pose=init_pose, RENDER=RENDER, seed=args.random_seed)

    elif args.env_name == 'reach':
        xml_path = './env/asset/ur5e_new/eval-vis-reach.xml'
        agent = ManipulatorAgent(rel_xml_path=xml_path,VERBOSE=False, USE_MUJOCO_VIEWER=USE_MUJOCO_VIEWER, MODE=MODE)
        agent.reset() # reset
        # Move tables and robot base
        agent.model.body('base').pos = np.array([0.0,0.0,0.0])
        init_pose = np.array([0, -1.57, 0, -1.57, -3.14, 3.14])
        env = ReachEnvrionment(agent=agent, init_pose=init_pose, RENDER=RENDER, seed=args.random_seed)


    q1, q2, q1_target, q2_target = CriticClass(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0]).to(args.device), \
    CriticClass(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0]).to(args.device),\
    CriticClass(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0]).to(args.device), \
    CriticClass(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0]).to(args.device)
    pi     = ActorClass(state_dim=env.observation_space.shape[0],  action_dim=env.action_space.shape[0]).to(args.device)
    buffer = ReplayBuffer(buffer_limit=args.buffer_limit, device=args.device)
    score = 0 
    best_score = 0 
    step = 0 
    for epoch in range(args.total_epochs):
        bestsc = 0; done = False 
        s = env.reset()
        a, log_prob = pi(np2torch(s, device=device))
        # Scaling 
        s_prime,reward,done,_ =env.step(torch2np(a))
        buffer.put((s,torch2np(a), reward, s_prime, done))
        score+=reward 
        bestsc +=reward 
        step+=1; 
        print('epoch: [{}] reward: [{}]'.format(epoch+1,reward))
        if args.WANDB:
            wandb.log({"Training Reward":reward}, step=epoch+1)   

        if bestsc>best_score: best_score=bestsc
        if buffer.size()>1000:
            mini_batch = buffer.sample(args.batch_size)
            td_target = get_target(pi, q1_target, q2_target, mini_batch)
            q1.train_q(td_target, mini_batch)
            q2.train_q(td_target, mini_batch)
            pi.train_p(q1, q2, mini_batch)
            q1.soft_update(q1_target)
            q2.soft_update(q2_target)       

        if (epoch+1)% 10 ==0 or (epoch+1)== args.total_epochs:
            # Eval
            done = False 
            action_lst = []
            total_reward = 0
            cnt = 0
            for i in range(10):
                s = env.reset()
                a, log_prob = pi(np2torch(s, device=device))
                s_prime,reward,done, _ =env.eval_step(torch2np(a))
                action_lst.append(a.detach().numpy())
                total_reward+=reward
                if done == True: 
                    cnt+=1
            diversity = get_diversity(action_lst, args.env_name) 
            print("number of epoch :{}, avg score :{:.4f}, best score :{:.1f}, avg step :{:.1f}, alpha:{:.4f} div:{:.4f}".format(epoch+1, score/step, best_score, step/args.print_interval, pi.log_alpha.exp(),diversity)) 
            torch.save(pi.state_dict(), './weights/{}/sac/'.format(args.env_name)+args.runname+'/p1_{}.pth'.format(epoch+1))
            torch.save(q1.state_dict(), './weights/{}/sac/'.format(args.env_name)+args.runname+'/q1_{}.pth'.format(epoch+1))
            torch.save(q2.state_dict(), './weights/{}/sac/'.format(args.env_name)+args.runname+'/q2_{}.pth'.format(epoch+1))
            if args.WANDB:
                wandb.log({"Eval Reward":reward,
                           "Avg Score":score/step,
                           "Best Score":best_score,
                           "SR":cnt/10,
                           "Eval Reward":total_reward/10,
                           "Diversity":diversity}, step=epoch+1) 
            score = 0
            step = 0

if __name__ == "__main__":
    device = 'cpu' #torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('--runname', type=str, default='final')
    parser.add_argument('--pname', type=str, default='place')
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--WANDB', type=bool, default=True)
    parser.add_argument('--RENDER', type=bool, default=False)
    # Fixed parameters
    parser.add_argument('--device', type=str, default=device)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--buffer_limit', type=int, default=1_000_000)
    parser.add_argument('--print_interval', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--env_name', type=str, default='place')
    parser.add_argument('--total_epochs', type=int, default=10_000)
    args = parser.parse_args()
    main(args)