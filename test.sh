cd ..  #到主目录
source env_isaaclab/bin/activate #激活虚拟环境

cd swarm_rl/scripts/
python3 visualize_enemy.py --task FAST-Quadcopter-Bodyrate --num_envs 1 --verbosity DEBUG

# python3 -u check_env_numeric.py --headless --steps 1200 --vm_set 2.2 #测试脚本，自检跑分脚本

# 追踪可视化测试脚本
# cd ~/swarm_rl
# python3 scripts/demo_intercept_controller.py --steps 1400 --vm_set 2.2 --k_yaw 2.0 --k_pitch 2.0

#使用rsl_rl库下的ppo算法进行训练
#cd ~/swarm_rl/reinforcement_learning/rsl_rl
#python3 train.py --task FAST-Quadcopter-Bodyrate --num_envs 5000
#quadcopter_rsl_rl_ppo_cfg.py #在该py文件中修改ppo的参数

#btop   直观地展示 CPU、内存、磁盘、网络、进程等资源的实时状态
#nvtop  GPU 状态查看器

#查看训练的东西有没有收敛
#python -m tensorboard.main --logdir ../reinforcement_learning/rsl_rl/outputs/rsl_rl/FAST-Quadcopter-Bodyrate/flowline


#play刚才训练的部分
#python3 play.py --task FAST-Quadcopter-Bodyrate --checkpoint /home/wcw/swarm_rl/reinforcement_learning/rsl_rl/outputs/rsl_rl/FAST-Quadcopter-Bodyrate/flowline/2025-09-01_13-18-03/model_1000.pt


