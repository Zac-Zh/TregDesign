#!/usr/bin/env python3
import os
import sys
import subprocess


def setup():
    """设置项目环境并安装依赖项"""
    print("设置HLA-DQ8-胰岛素特异性Treg TCR筛选环境")

    # 创建必要的目录
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 安装依赖项
    print("安装所需依赖项...")
    requirements = [
        "scanpy>=1.9.3",
        "pandas>=1.5.3",
        "numpy>=1.23.5",
        "matplotlib>=3.7.1",
        "seaborn>=0.12.2",
        "biopython>=1.81",
        "torch>=2.0.0",
        "scikit-learn>=1.2.2",
        "requests>=2.28.1"
    ]

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + requirements)
        print("依赖项安装成功")
    except subprocess.CalledProcessError:
        print("依赖项安装失败，请检查网络连接和权限")
        return False

    print("\n环境设置完成! 您可以通过以下步骤运行程序:")
    print("1. 运行主程序: python main.py")
    print("   - 添加--skip-download跳过数据下载 (如果已有数据)")
    print("   - 添加--data-dir和--output-dir自定义目录")
    print("\n有关更多选项，请运行: python main.py --help")

    return True


if __name__ == "__main__":
    setup()