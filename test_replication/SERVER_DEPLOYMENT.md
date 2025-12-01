# 服务器部署指南

## ✅ 代码兼容性检查

经过检查，代码**完全可以在服务器上运行**：

### ✅ 优点
1. **无硬编码路径**：所有路径都是相对路径（`data/train`, `outputs/`）
2. **自动设备检测**：自动检测 CUDA/CPU，无需手动配置
3. **跨平台兼容**：使用标准 Python 库，支持 Linux/Windows/macOS
4. **依赖清晰**：所有依赖都在 `requirements.txt` 中

### ⚠️ 需要注意
1. **数据文件**：需要确保 `data/` 目录存在并包含训练数据
2. **GPU 支持**：如果有 GPU，会自动使用（推荐）
3. **网络连接**：首次运行需要下载 BERT 模型（约 400MB）

## 服务器要求

### 最低配置（CPU 训练）
- **CPU**: 4 核以上
- **内存**: 8GB 以上
- **存储**: 20GB 以上（模型 + 数据）
- **训练时间**: 约 20-30 小时（10 epochs）

### 推荐配置（GPU 训练）
- **GPU**: NVIDIA GPU，支持 CUDA（至少 6GB 显存）
  - 推荐：RTX 3060/3070, V100, A100
- **CPU**: 4 核以上
- **内存**: 16GB 以上
- **存储**: 30GB 以上
- **训练时间**: 约 5-7 小时（10 epochs）

### 云服务器推荐

#### 1. 阿里云 / 腾讯云
- **GPU 实例**: g5/g6 系列
- **价格**: 约 ¥5-15/小时
- **推荐**: 按需付费，训练完成后释放

#### 2. AutoDL / 恒源云
- **GPU**: RTX 3090, A100
- **价格**: 约 ¥1-3/小时
- **推荐**: 性价比高，适合学生

#### 3. Google Colab Pro
- **GPU**: T4/V100
- **价格**: $10/月
- **限制**: 有使用时长限制

## 部署步骤

### 1. 准备服务器环境

```bash
# 1. 连接到服务器
ssh user@your-server-ip

# 2. 安装 Python 3.8+（如果还没有）
python3 --version  # 检查版本

# 3. 安装 uv（推荐）或使用 pip
curl -LsSf https://astral.sh/uv/install.sh | sh
# 或者
pip3 install --user uv
```

### 2. 上传代码和数据

```bash
# 方式1: 使用 git（推荐）
git clone <your-repo-url>
cd test_replication

# 方式2: 使用 scp 上传
scp -r test_replication/ user@server:/path/to/destination/
```

### 3. 准备数据

```bash
# 确保 data/ 目录存在并包含数据文件
ls data/
# 应该看到: train, valid, test

# 如果没有数据，需要先准备数据文件
# 格式：每行 "word punctuation"
```

### 4. 安装依赖

```bash
# 使用 uv（推荐）
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# 或使用 pip
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 5. 检查 GPU（如果有）

```bash
# 检查 CUDA 是否可用
python3 -c "import torch; print(torch.cuda.is_available())"
# 应该输出: True

# 检查 GPU 信息
nvidia-smi
```

### 6. 运行训练

```bash
# 激活环境
source .venv/bin/activate

# 方式1: 使用 nohup 后台运行（推荐）
nohup python run_10_epochs_comparison.py > training.log 2>&1 &

# 方式2: 使用 screen（推荐，可以随时查看）
screen -S training
python run_10_epochs_comparison.py
# 按 Ctrl+A 然后 D 退出 screen
# 重新连接: screen -r training

# 方式3: 使用 tmux
tmux new -s training
python run_10_epochs_comparison.py
# 按 Ctrl+B 然后 D 退出
# 重新连接: tmux attach -t training
```

### 7. 监控训练进度

```bash
# 查看日志
tail -f baseline_training_*.log
tail -f adapter_training_*.log

# 或使用状态检查脚本
python check_training_status.py

# 查看 GPU 使用情况
watch -n 1 nvidia-smi
```

### 8. 下载结果（训练完成后）

```bash
# 方式1: 使用 scp
scp -r user@server:/path/to/test_replication/outputs/ ./

# 方式2: 使用 rsync（推荐，支持断点续传）
rsync -avz --progress user@server:/path/to/test_replication/outputs/ ./outputs/
```

## 常见问题

### Q1: 如何确保训练不会中断？

**A**: 使用 `screen` 或 `tmux`，即使 SSH 断开连接，训练也会继续。

```bash
# 使用 screen
screen -S training
python run_10_epochs_comparison.py
# 按 Ctrl+A 然后 D 退出

# 重新连接
screen -r training
```

### Q2: 如何节省服务器成本？

**A**: 
1. 使用 Adapter 模型（训练更快，显存占用更少）
2. 减少训练数据量（调整 `MAX_TRAIN_LINES`）
3. 使用按需付费，训练完成后立即释放
4. 考虑使用 AutoDL/恒源云等性价比高的平台

### Q3: 显存不足怎么办？

**A**:
1. 减小 batch size: `--batch-size 20`
2. 使用 Adapter 模型（显存占用更少）
3. 禁用混合精度: `--no-amp`（不推荐）
4. 使用梯度累积（需要修改代码）

### Q4: 如何加快训练速度？

**A**:
1. 使用 GPU（最重要）
2. 使用混合精度训练（默认启用）
3. 使用 Adapter 模型
4. 增加 batch size（如果显存允许）

### Q5: 服务器上没有 GPU 怎么办？

**A**: 
- 代码会自动使用 CPU，但训练会很慢（约 20-30 小时）
- 建议租用带 GPU 的服务器，或使用 Google Colab

## 成本估算

### 训练时间估算
- **基线模型（10 epochs）**: 3.5-4 小时
- **Adapter 模型（10 epochs）**: 2.5-3 小时
- **总计**: 约 6-7 小时

### 服务器成本（示例）
- **AutoDL RTX 3090**: ¥2/小时 × 7小时 = ¥14
- **阿里云 GPU**: ¥8/小时 × 7小时 = ¥56
- **Google Colab Pro**: $10/月（不限时长，但有每日限制）

## 快速部署脚本

创建 `deploy_to_server.sh`:

```bash
#!/bin/bash
# 服务器快速部署脚本

echo "=== 服务器部署脚本 ==="

# 1. 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 Python3"
    exit 1
fi

# 2. 创建虚拟环境
echo "创建虚拟环境..."
python3 -m venv .venv
source .venv/bin/activate

# 3. 安装依赖
echo "安装依赖..."
pip install -r requirements.txt

# 4. 检查数据
if [ ! -d "data" ] || [ ! -f "data/train" ]; then
    echo "警告: data/ 目录不存在或数据文件缺失"
    echo "请确保 data/train, data/valid, data/test 存在"
fi

# 5. 检查 GPU
echo "检查 GPU..."
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"

echo "=== 部署完成 ==="
echo "运行训练: python run_10_epochs_comparison.py"
```

## 推荐服务器配置

### 预算有限（¥20-50）
- **AutoDL RTX 3090**: ¥2/小时
- **训练时间**: 约 7 小时
- **总成本**: 约 ¥14

### 标准配置（¥50-100）
- **阿里云 GPU 实例**: ¥8/小时
- **训练时间**: 约 7 小时
- **总成本**: 约 ¥56

### 高性能（¥100+）
- **A100 GPU**: ¥15-20/小时
- **训练时间**: 约 4-5 小时（更快）
- **总成本**: 约 ¥60-100

## 总结

✅ **代码完全可以在服务器上运行**
✅ **无需修改代码，直接使用**
✅ **推荐使用 GPU 服务器，训练速度快 5-10 倍**
✅ **使用 screen/tmux 确保训练不中断**
✅ **成本可控，约 ¥14-100 即可完成训练**

建议使用 **AutoDL** 或 **恒源云**，性价比最高！

