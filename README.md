# Welcome to Swarm-Interception！
首先系统是**ubuntu22.04，isaac lab4.50,ROS2humble**
# 1.第一步安装ubuntu22.04
教程如下：
```
https://www.bilibili.com/video/BV1Cc41127B9/?spm_id_from=333.337.search-card.all.click&vd_source=ab91ad8b781b309917e3e7c162e2e67d
```
# 2.第二步安装ROS2humble
教程如下：
```
https://www.bilibili.com/video/BV14p4y1j7wE/?spm_id_from=333.337.search-card.all.click&vd_source=ab91ad8b781b309917e3e7c162e2e67d
```
# 3.第三步安装isaac lab
进入官方的说明文档
```
https://isaac-sim.github.io/IsaacLab/v2.1.0/source/setup/installation/pip_installation.html
```
进入local installation下的pip installation

<img width="306" height="429" alt="image" src="https://github.com/user-attachments/assets/e4352252-e2e5-4664-adc0-f0895d6b5911" />

之后按照文档的流程安装isaac sim与isaac lab
```
python3.10 -m venv env_isaaclab
source env_isaaclab/bin/activate
pip install --upgrade pip
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
```
检查isaac sim是否安装成功：
```
isaacsim 									  			# Check that the simulator runs as expected
```
```
isaacsim isaacsim.exp.full.kit 				# It’s also possible to run with a specific experience file
```
安装isaac lab
```
git clone https://github.com/isaac-sim/IsaacLab.git
```
```
sudo apt install cmake build-essential
```
```
./isaaclab.sh --install
```
检查isaac lab是否安装成功
```
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
```
## 2.第四步下载swarm_rl
```
git clone https://github.com/wangchengwei-wcw/swarm_rl.git
```
克隆子仓库内容
```
git submodule init
```
```
git submodule update
```
**attention!** 
必须进入工作目录下， cd进入 /home/wcw/swarm_rl/envs/lib/IsaacLab，运行isaaclab.sh该脚本，目的是安装该项目自带的isaac lab，之前从官网下载的isaac lab可以删掉。
```
./isaaclab.sh --install
```
将skrl更换为魔改的skrl
```
pip install -e /home/wcw/swarm_rl/reinforcement_learning/lib
```
## 3.训练你自己的agent
确保source了ros2
```
source /opt/ros/humble/setup.zsh # or setup.bash
```
如果你想要训练自己的ai，那么需要先激活虚拟环境
```
source env_isaaclab/bin/activate
```
之后进入到目录中去
```
cd ~/swarm_rl/reinforcement_learning/skrl
```
```
python3 train.py --task FAST-Intercept-Swarm-Distributed --num_envs 500 --algorithm IPPO --headless
```
回放刚才训练的内容
```
python3 play.py --task FAST-Intercept-Swarm-Distributed --algorithm IPPO --checkpoint /home/wcw/swarm_rl/reinforcement_learning/skrl/outputs/skrl/FAST-Intercept-Swarm-Distributed/flowline/2025-09-01_13-18-03/model_1000.pt
```
打开tensorboard，可以查看训练曲线
```
python -m tensorboard.main --logdir outputs/skrl/FAST-Intercept-Swarm-Distributed/flowline/
```
