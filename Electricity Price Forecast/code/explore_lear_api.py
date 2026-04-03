"""
探索 LEAR 模型的 API
"""

import sys
sys.path.append('..')

from epftoolbox.models import LEAR
import inspect

print("=" * 80)
print("LEAR 模型 API 探索")
print("=" * 80)

# 获取 LEAR 类的签名
print("\n1. LEAR 类初始化参数:")
sig = inspect.signature(LEAR.__init__)
for param_name, param in sig.parameters.items():
    if param_name != 'self':
        default = param.default if param.default != inspect.Parameter.empty else '无默认值'
        print(f"  - {param_name}: {default}")

# 获取 LEAR 类的方法
print("\n2. LEAR 类方法:")
for name, method in inspect.getmembers(LEAR, predicate=inspect.isfunction):
    if not name.startswith('_'):
        print(f"  - {name}")

# 尝试创建一个实例并查看其属性
print("\n3. 创建 LEAR 实例:")
try:
    # 使用默认参数创建
    lear = LEAR()
    print("  ✓ 默认参数创建成功")
    
    # 查看实例属性
    print("\n4. LEAR 实例属性:")
    for attr in dir(lear):
        if not attr.startswith('_'):
            print(f"  - {attr}")
    
except Exception as e:
    print(f"  ✗ 创建失败: {e}")

print("\n" + "=" * 80)
