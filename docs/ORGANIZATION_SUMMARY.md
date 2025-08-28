# DAMP Project Organization Summary

## 🎯 整理完成！

项目文件夹和代码已经成功整理完成，现在具有清晰的结构和命名规范。

## 📁 新的项目结构

```
DAMP/
├── 🚀 run.py                          # 主入口点
├── 📁 src/                            # 核心源代码
├── 📁 scripts/                        # 主要脚本
├── 📁 test/                           # 测试文件
├── 📁 visualization/                  # 可视化工具
├── 📁 docs/                           # 文档
├── 📁 dataset/                        # 数据文件
├── 📁 models/                         # 训练好的模型
├── 📁 results/                        # 生成结果
├── 📁 plots/                          # 生成的图片
├── 📁 logs/                           # 训练日志
├── ⚙️ config.py                       # 配置
├── 📦 requirements.txt                # 依赖
└── 📋 pyproject.toml                  # 项目元数据
```

## ✅ 整理成果

1. **文件归类**: 所有文件按功能分类到相应目录
2. **命名规范**: 统一的命名约定
3. **测试组织**: 测试文件集中在 `test/` 目录
4. **可视化整理**: 可视化工具集中在 `visualization/` 目录
5. **文档整理**: 文档集中在 `docs/` 目录
6. **脚本整理**: 主要脚本集中在 `scripts/` 目录
7. **主入口点**: 创建了统一的 `run.py` 入口点

## 🚀 使用方法

```bash
# 主程序
python run.py --epochs 50 --num_sequences 100

# 运行测试
python test/run_tests.py

# 生成可视化
python visualization/advanced_plotting.py
```

---

**整理状态**: ✅ 完成  
**最后更新**: 2024-08-24 