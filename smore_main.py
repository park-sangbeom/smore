import numpy as np 
import os
import torch 
import torch.nn.functional as F
import wandb
import argparse 
import sys 
sys.path.append('..')
from model.smore.smore import SMORE 
from model.smore.buffer import BufferClass
from model.utils.utils import np2torch, torch2np,get_runname
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
    SAVE_WEIGHT_PATH = "./weights/sweep/smore/"+args.runname
    os.makedirs(SAVE_WEIGHT_PATH, exist_ok=True) 
    xml_path = './env/asset/ur5e_new/scene_ur5e_rg2_obj.xml'
    if args.RENDER:
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
        env = SweepEnvrionment(agent=agent, init_pose=init_pose, RENDER=args.RENDER, seed=args.random_seed)
    
    elif args.env_name == 'place':
        xml_path = './env/asset/ur5e_new/scene_ur5e_rg2_obj_place.xml'
        agent = ManipulatorAgent(rel_xml_path=xml_path,VERBOSE=False, USE_MUJOCO_VIEWER=USE_MUJOCO_VIEWER, MODE=MODE)
        agent.reset() # reset
        # Move tables and robot base
        agent.model.body('base_table').pos = np.array([0,0,0.395])
        agent.model.body('front_object_table').pos = np.array([0.38+0.6,0,0])
        agent.model.body('base').pos = np.array([0.18,0,0.79])
        init_pose= np.array([0, -1.57, 1.57, 1.57, 1.57, 0])
        env = PlaceEnvrionment(agent=agent, init_pose=init_pose, RENDER=args.RENDER, seed=args.random_seed)

    elif args.env_name == 'reach':
        xml_path = './env/asset/ur5e_new/eval-vis-reach.xml'
        agent = ManipulatorAgent(rel_xml_path=xml_path,VERBOSE=False, USE_MUJOCO_VIEWER=USE_MUJOCO_VIEWER, MODE=MODE)
        agent.reset() # reset
        # Move tables and robot base
        agent.model.body('base').pos = np.array([0.0,0.0,0.0])
        init_pose = np.array([0, -1.57, 0, -1.57, -3.14, 3.14])
        env = ReachEnvrionment(agent=agent, init_pose=init_pose, RENDER=args.RENDER, seed=args.random_seed)
    buffer    = BufferClass(xdim=args.xdim, cdim=args.cdim, buffer_limit=args.buffer_limit, device=args.device)
    smore      = SMORE(xdim=args.xdim, cdim=args.cdim, zdim=args.zdim, hdims=args.hdims, n_components=args.n_components, lr=args.lr, device=args.device)
    traj_cnt  = 0
    path = 'test_buffer.json'
    for epoch in range(args.total_epochs):
        obs    = env.reset()
        if args.env_name=='sweep':
            obs_randxys = obs[3:].reshape(6,-1)
            obs_randxs, obs_randys = obs_randxys[:,0],obs_randxys[:,1]

        elif args.env_name=='place':
            obs_randxys = obs[6:].reshape(5,-1)
            obs_randxs, obs_randys = obs_randxys[:,0],obs_randxys[:,1]
        c = env.one_hot.copy()
        # Epsgrdy
        one_to_zero = 1.0-(epoch/(args.total_epochs-1))
        exploration_rate = 0.8*one_to_zero 
        if (np.random.rand()>exploration_rate): # Exploitation
            z = torch.randn(size=(1, args.zdim)).to(args.device)
            # z = z/torch.norm(z)
            # z = F.normalize(z, p=2, dim=1)
            if args.env_name=='sweep':
                xs, ys, _ = smore.exploit(c=np2torch(c, device=args.device).reshape(-1,args.cdim))
                # Execute the trajectory
                _, reward, _, _ = env.step_traj(xs=xs, ys=ys)
                _, anchor = smore.grp.get_anchors_from_traj(xs=xs, ys=ys, n_anchor=10)
            elif args.env_name=='place':
                xs,ys,zs,_ = smore.exploit_3d(c=np2torch(c, device=args.device).reshape(-1,args.cdim))
                # xs,ys,zs = dlpg.exploit_place(c=np2torch(c, device=args.device).reshape(-1,args.cdim), z=z)
                # Execute the trajectory
                _, reward, _, _ = env.step_traj_3d(xs=xs, ys=ys, zs=zs)
                # anchor = smore.grp.get_anchors_from_traj_3d(xs=xs, ys=ys, zs=zs, n_anchor=10)
                _, anchor = smore.grp.get_anchors_from_traj(xs=xs, ys=ys, n_anchor=10)
            buffer.store(x=anchor.reshape(-1), c=c, reward=reward)
            avg_reward = reward 
            max_reward = reward
            traj_cnt +=1 
            # with open(path, "a") as f:
            #     content = {'traj':traj_cnt, "x":anchor.reshape(-1).tolist(), "c":c.tolist(), "reward":reward}
            #     f.write(json.dumps(content)+'\n')
        else: # Exploration
            total_reward = 0 
            max_reward = -10
            if args.env_name=='reach' or args.env_name=='place':
                xs_lst, ys_lst, zs_lst = smore.random_explore_3d(n_sample=10, scenario=env.scenario)
                for xs, ys, zs in zip(xs_lst, ys_lst, zs_lst):
                    traj_cnt +=1
                    _, reward, _, _ = env.step_traj_3d(xs=xs, ys=ys, zs=zs)
                    if reward>max_reward:
                        max_reward = reward   
                    # anchor = smore.grp.get_anchors_from_traj_3d(xs=xs, ys=ys, zs=zs, n_anchor=10)
                    _, anchor = smore.grp.get_anchors_from_traj(xs=xs, ys=ys, n_anchor=10)
                    buffer.store(x=anchor.reshape(-1), c=c, reward=reward)
                    total_reward+=reward
                    env.manual_reset(obs_randxs, obs_randys)
                    # with open(path, "a") as f:
                    #     content = {'traj':traj_cnt, "x":anchor.reshape(-1).tolist(), "c":c.tolist(), "reward":reward}
                    #     f.write(json.dumps(content)+'\n')

            else:
                xs, ys_lst = smore.random_explore(n_sample=10)
                for ys in ys_lst:
                    traj_cnt +=1 
                    _, reward, _, _ = env.step_traj(xs=xs, ys=ys)
                    if reward>max_reward:
                        max_reward = reward
                    _, anchor = smore.grp.get_anchors_from_traj(xs=xs, ys=ys, n_anchor=10)
                    buffer.store(x=anchor.reshape(-1), c=c, reward=reward)
                    total_reward+=reward
                    env.manual_reset(obs_randxs, obs_randys)
            avg_reward = total_reward/args.n_sample
        print("[Epoch:{}][Reward:{}][Max Reward:{}]".format(epoch+1, avg_reward, max_reward))  
        if args.WANDB:
            wandb.log({"train avg reward":avg_reward}, step=epoch+1)   

        if traj_cnt>500:
            traj_cnt=0
        # if (epoch+1)%500==0 and (epoch+1)>1000:
            unscaled_loss_recon_sum=0;unscaled_loss_kld_sum=0; loss_recon_sum=0;loss_kl_sum=0;n_batch_sum=0;train_unscaled_reward_sum=0
            for iter in range(args.MAXITER):
                batch = buffer.sample_batch(sample_method=args.sample_method,
                                            batch_size=args.batch_size)
                mean_unscaled_kld_loss, mean_unscaled_recon_loss, kld_loss, recon_loss, total_loss = smore.update(batch)
                loss_recon_sum = loss_recon_sum + args.batch_size*recon_loss
                unscaled_loss_recon_sum = unscaled_loss_recon_sum+args.batch_size*mean_unscaled_recon_loss
                unscaled_loss_kld_sum = unscaled_loss_kld_sum+args.batch_size*mean_unscaled_kld_loss
                loss_kl_sum    = loss_kl_sum + args.batch_size*kld_loss
                n_batch_sum    = n_batch_sum + args.batch_size   
                unscaled_reward_batch_np = torch2np(batch["reward"])
                train_unscaled_reward_sum +=np.average(unscaled_reward_batch_np) 
                
                n_batch_sum    = n_batch_sum + args.batch_size   
            # Average loss during train
            loss_recon_avg, loss_kl_avg, unscaled_loss_recon_avg = (loss_recon_sum/n_batch_sum),(loss_kl_sum/n_batch_sum), (unscaled_loss_recon_sum/n_batch_sum)            
            print ("[%d/%d] SMORE updated. Reward: [%.3f] Total loss:[%.3f] (recon:[%.3f] kl:[%.3f])"%
            (epoch+1,args.total_epochs, train_unscaled_reward_sum/n_batch_sum,loss_recon_avg+loss_kl_avg,loss_recon_avg,loss_kl_avg))
            # Evaluation
            with torch.no_grad():
                eval_total_reward = 0
                eval_cnt = 0
                for i in range(10):
                    obs = env.reset()
                    c =  env.one_hot.copy()
                    if args.env_name=='reach' or args.env_name=='place':
                        xs,ys,zs,_ = smore.exploit_3d(c=np2torch(c, device=args.device).reshape(-1,args.cdim))
                        # xs,ys,zs = dlpg.exploit_place(c=np2torch(c, device=args.device).reshape(-1,args.cdim), z=z)
                        _, reward, done, _ = env.eval_step_traj_3d(xs=xs, ys=ys, zs=zs)

                    elif args.env_name=='sweep':
                        xs, ys, _ = smore.exploit(c=np2torch(c, device=args.device).reshape(-1,args.cdim))
                        # Execute the trajectory
                        _, reward, done, _ = env.eval_step_traj(xs=xs, ys=ys)
                    eval_total_reward+=reward 
                    if done:
                        eval_cnt+=1
                if args.WANDB:
                    wandb.log({
                    "eval avg reward":eval_total_reward/10,
                    "SR":eval_cnt/10}, step=epoch+1)   
        # if (epoch+1)%50==0 or (epoch+1)==(args.total_epochs-1):
            torch.save(smore.state_dict(),SAVE_WEIGHT_PATH+"/smore_{}.pth".format(epoch+1))
            print("WEIGHT SAVED.")

if __name__=="__main__":
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='place')
    parser.add_argument('--runname', type=str, default='final')
    parser.add_argument('--WANDB', type=bool, default=True)
    parser.add_argument('--pname', type=str, default='place')
    parser.add_argument('--sample_method', type=str, default='laqdpp')
    parser.add_argument('--spectral_norm', type=bool, default=False)
    parser.add_argument('--xdim', type=int, default=20) # 30
    parser.add_argument('--cdim', type=int, default=6) # Sweep: 3/Place: 6
    parser.add_argument('--zdim', type=int, default=5)
    parser.add_argument("--hdims", nargs="+", default=[128, 128]) 
    parser.add_argument("--n_components", type=int, default=10)   
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--total_epochs', type=int, default=15000)
    parser.add_argument('--MAXITER', type=int, default=40)
    parser.add_argument('--recon_gain', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.01)
    parser.add_argument('--RENDER', type=bool, default=False)
    parser.add_argument('--n_sample', type=int, default=10)
    parser.add_argument('--buffer_limit', type=int, default=1000)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--device', type=str, default=device)
    args = parser.parse_args()

    main(args)