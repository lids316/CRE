---

# NNI 使用手册

## 目录
1. **NNI 简介**
2. **安装与配置**
3. **快速入门**
4. **实验配置详解**
5. **高级功能**
6. **注意事项与常见问题**

---

## 1. NNI 简介
NNI 是微软开源的自动化机器学习（AutoML）工具，专注于帮助用户高效地进行超参数调优（HPO）、神经网络架构搜索（NAS）、模型压缩等任务。其主要特点包括：
- 支持多种调优算法（Random Search、TPE、贝叶斯优化、进化算法等）。
- 分布式部署，支持本地、远程服务器和 Kubernetes。
- 可视化监控实验进展。
- 灵活的 API 接口，兼容 PyTorch、TensorFlow 等主流框架。

---

## 2. 安装与配置

### 2.1 安装
通过 `pip` 安装：
```bash
pip install nni
```

### 2.2 验证安装
```bash
nnictl --version
```
若输出版本号（如 `2.10`），则安装成功。

---

## 3. 快速入门

### 3.1 实验流程
1. **定义搜索空间**：指定需要调优的超参数范围。
2. **修改训练代码**：插入 NNI 的 API 以接收超参数并返回结果。
3. **配置实验**：编写 YAML 文件定义调优算法、最大 Trial 数等。
4. **启动实验**：通过命令行启动并监控实验。

### 3.2 示例：MNIST 分类任务调参

#### 步骤 1：创建搜索空间文件 `search_space.json`
```json
{
    "lr": {"_type": "loguniform", "_value": [0.0001, 0.1]},
    "batch_size": {"_type": "choice", "_value": [32, 64, 128]}
}
```

#### 步骤 2：修改训练代码（`train.py`）
```python
import nni

params = {
    'lr': 0.001,
    'batch_size': 32,
}
# 从 NNI 获取超参数
params = nni.get_next_parameter()

# 训练逻辑
def train():
    # ... （加载数据、定义模型、优化器等）
    for epoch in range(10):
        loss = model.train(params['batch_size'])
        accuracy = evaluate()
        nni.report_intermediate_result(accuracy)  # 报告中间结果
    nni.report_final_result(accuracy)  # 报告最终结果

if __name__ == '__main__':
    train()
```

#### 步骤 3：创建实验配置文件 `config.yml`
```yaml
authorName: YourName
experimentName: mnist_demo
trialConcurrency: 2  # 并行 Trial 数
maxExecDuration: 1h  # 最大运行时间
maxTrialNum: 10      # 最大 Trial 数
trainingServicePlatform: local  # 运行环境（支持 remote, kubernetes）
searchSpaceFile: search_space.json
useAnnotation: false
tuner:
  name: TPE          # 调优算法（可选：Random、SMAC、Evolution 等）
  classArgs:
    optimize_mode: maximize
trial:
  command: python train.py
  codeDir: .
  gpuNum: 0          # 指定 GPU 数量
```

#### 步骤 4：启动实验
```bash
nnictl create --config config.yml
```

#### 步骤 5：监控实验结果
访问命令行输出的 Web UI URL（如 `http://localhost:8080`），查看实时结果、超参数分布和 Trial 详情。

---

## 4. 实验配置详解

### 4.1 搜索空间（Search Space）
- **支持的类型**：`choice`, `randint`, `uniform`, `quniform`, `loguniform`, `normal` 等。
- **示例**：
  ```json
  {
      "dropout_rate": {"_type": "uniform", "_value": [0.5, 0.9]},
      "hidden_size": {"_type": "choice", "_value": [64, 128, 256]}
  }
  ```

### 4.2 实验配置（`config.yml`）
- **关键参数**：
    - `trialConcurrency`: 并行运行的 Trial 数量。
    - `maxTrialNum`: 最大 Trial 总数。
    - `tuner`: 调优算法（如 `TPE`, `Random`, `SMAC`）。
    - `assessor`: 提前终止策略（如 `Medianstop`）。

### 4.3 训练代码适配
- **关键 API**：
    - `nni.get_next_parameter()`: 获取超参数。
    - `nni.report_intermediate_result(metric)`: 报告中间指标（如每个 epoch 的准确率）。
    - `nni.report_final_result(metric)`: 报告最终指标。

---

## 5. 高级功能

### 5.1 自定义调优算法
- 继承 `nni.tuner.Tuner` 类，实现 `generate_parameters` 和 `receive_trial_result` 方法。
- 在 `config.yml` 中指定自定义 Tuner 路径。

### 5.2 分布式部署
- **远程服务器**：在 `config.yml` 中设置 `trainingServicePlatform: remote`，并配置机器列表。
- **Kubernetes**：使用 `trainingServicePlatform: kubernetes`，提供 Kubernetes 集群配置。

### 5.3 模型压缩与 NAS
- **模型压缩**：使用 `nni.compression` 对模型进行剪枝、量化。
- **神经网络架构搜索**：通过 `nni.nas` 定义搜索空间并探索最佳架构。

---

## 6. 注意事项与常见问题

### 6.1 注意事项
- **代码兼容性**：确保训练代码能在不同超参数下独立运行。
- **资源管理**：合理设置 `trialConcurrency` 避免资源耗尽。
- **实验恢复**：使用 `nnictl resume EXPERIMENT_ID` 恢复中断的实验。
- **超参数搜索空间**：避免范围过大或无效组合。

### 6.2 常见问题
- **Q：实验无法启动**
    - 检查 `config.yml` 格式是否正确，端口是否被占用。
- **Q：Trial 未返回结果**
    - 确保代码中调用了 `report_final_result`。
- **Q：调优效果不佳**
    - 更换 Tuner 算法或调整搜索空间范围。

---


## 7. NNI 常用命令

NNI 的核心管理工具是 `nnictl` 命令行工具，用于创建、监控、停止实验等操作。以下是常用命令及示例：

---

### **7.1 实验管理命令**

#### 1. 创建并启动实验
```bash
# 指定配置文件启动实验（默认端口8080）
nnictl create --config config.yml

# 指定端口和配置文件
nnictl create --config config.yml --port 8888
```

#### 2. 查看所有运行中的实验
```bash
nnictl experiment list
```
输出示例：
```
+----+--------------+---------+----------+---------+-------------+----------+------------+
| ID |     Name     | Status  | Start Time | End Time | Port | Platform |    Tags    |
+----+--------------+---------+----------+---------+-------------+----------+------------+
| 1  | mnist_demo   | RUNNING | 10:00:00 | -        | 8080 | local    | autoML    |
+----+--------------+---------+----------+---------+-------------+----------+------------+
```

#### 3. 停止实验
```bash
# 停止指定实验（通过实验ID）
nnictl stop 1

# 停止所有实验
nnictl stop --all
```

#### 4. 恢复中断的实验
```bash
nnictl resume EXPERIMENT_ID
```

#### 5. 更新实验配置
```bash
# 修改 config.yml 后更新实验（需实验处于停止状态）
nnictl update --config new_config.yml
```

---

### **7.2 监控与调试命令**

#### 1. 查看实验状态
```bash
nnictl status [EXPERIMENT_ID]
```
若未指定 ID，默认显示最新实验的状态。

#### 2. 查看 Trial 日志
```bash
# 查看某个 Trial 的日志
nnictl trial log TRIAL_ID

# 实时追踪日志输出
nnictl trial log TRIAL_ID --tail
```

#### 3. 打开 Web UI
```bash
# 直接打开浏览器访问 Web 界面
nnictl webui url
```

#### 4. 查看 GPU 资源使用
```bash
# 显示实验中各 Trial 的 GPU 占用情况
nnictl experiment monitor GPU
```

---

### **7.3 数据导出与管理**

#### 1. 导出实验结果
```bash
# 导出所有 Trial 的指标数据为 CSV
nnictl experiment export --type csv --file output.csv
```

#### 2. 查看最佳超参数
```bash
# 显示当前实验的最佳 Trial 结果
nnictl experiment export --best
```

---

### **7.4 平台管理命令（远程/Kubernetes）**

#### 1. 管理远程机器
```bash
# 添加远程服务器到 NNI 配置
nnictl machine add --host 192.168.1.100 --username user --password pass

# 移除已注册的机器
nnictl machine remove --host 192.168.1.100
```

#### 2. Kubernetes 相关操作
```bash
# 部署 NNI 实验到 Kubernetes 集群（需提前配置 kubeconfig）
nnictl create --config k8s_config.yml --platform kubernetes
```

---

### **7.5 其他实用命令**

#### 1. 查看 NNI 版本
```bash
nnictl --version
```

#### 2. 查看帮助文档
```bash
# 查看全局帮助
nnictl --help

# 查看子命令帮助（如 create）
nnictl create --help
```

#### 3. 清空所有实验记录
```bash
# 删除所有已完成的实验元数据
nnictl experiment delete --all
```

---

## **命令使用示例场景**

### 场景 1：启动实验并实时监控
```bash
# 启动实验
nnictl create --config config.yml --port 8888

# 查看状态
nnictl status

# 打开 Web 监控页面
nnictl webui url

# 停止实验
nnictl stop 1
```

### 场景 2：调试失败的 Trial
```bash
# 查看所有 Trial 列表
nnictl trial list

# 查看失败 Trial 的日志
nnictl trial log failed_trial_id
```

---

## **注意事项**
- **实验 ID**：每个实验启动后会分配唯一 ID，可通过 `nnictl experiment list` 查看。
- **端口冲突**：若默认端口 8080 被占用，使用 `--port` 指定其他端口。
- **权限问题**：远程或 Kubernetes 操作需确保网络和权限配置正确。
---

