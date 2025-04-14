## 使用 Conda 安装 PyTorch和相关库
```bash
# 创建新环境
conda create -n pytorch_env python=3.10

# 激活环境
conda activate pytorch_env

# 安装 PyTorch 全家桶
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```
## 使用 Pip 安装依赖
```bash
# 安装其他依赖
pip install -r requirements.txt
```