import pandas as pd
import numpy as np
from datetime import datetime
import os

# 文件路径
market_boundary_file = "../Hubei Province Energy Dataset/00-市场边界信息/市场边界信息总表.xlsx"
avg_clearing_price_file = "../Hubei Province Energy Dataset/平均出清价格(2024-04-16至2025-03-28)-24点平铺.xlsx"
output_file = "../data/raw/市场边界_出清价格总表.csv"

print("开始数据聚合处理...")

# 读取数据
try:
    print("1. 读取市场边界信息总表...")
    market_boundary_df = pd.read_excel(market_boundary_file)
    print(f"   市场边界信息数据形状: {market_boundary_df.shape}")
    
    print("2. 读取平均出清价格数据...")
    avg_price_df = pd.read_excel(avg_clearing_price_file)
    print(f"   平均出清价格数据形状: {avg_price_df.shape}")
    
except Exception as e:
    print(f"读取数据时出错: {e}")
    exit(1)

# 数据预处理
print("3. 数据预处理...")

# 检查日期格式并转换为datetime
def convert_date(date_str):
    """将日期字符串转换为datetime对象"""
    if isinstance(date_str, str):
        try:
            return pd.to_datetime(date_str)
        except:
            return pd.NaT
    return date_str

# 处理市场边界信息数据
market_boundary_df['日期'] = market_boundary_df['日期'].apply(convert_date)
market_boundary_df = market_boundary_df.dropna(subset=['日期'])

# 处理平均出清价格数据
avg_price_df['日期'] = avg_price_df['日期'].apply(convert_date)
avg_price_df = avg_price_df.dropna(subset=['日期'])

# 创建时间戳列用于合并（确保日期格式统一为 YYYY-MM-DD）
market_boundary_df['时间戳'] = market_boundary_df['日期'].dt.strftime('%Y-%m-%d') + '_' + market_boundary_df['时段']
avg_price_df['时间戳'] = avg_price_df['日期'].dt.strftime('%Y-%m-%d') + '_' + avg_price_df['时段']

print(f"   预处理后市场边界信息数据形状: {market_boundary_df.shape}")
print(f"   预处理后平均出清价格数据形状: {avg_price_df.shape}")

# 创建完整的时间序列（补充缺失的时间点）
print("3.5. 创建完整时间序列并检测缺失时间点...")

# 获取日期范围
date_range = pd.date_range(
    start=min(market_boundary_df['日期'].min(), avg_price_df['日期'].min()),
    end=max(market_boundary_df['日期'].max(), avg_price_df['日期'].max()),
    freq='D'
)

# 创建24个时段（格式与原始数据一致：HH:00-(H+1):00，其中23:00-24:00）
hours = [f"{h:02d}:00-{h+1:02d}:00" if h < 23 else "23:00-24:00" for h in range(24)]

# 创建完整的时间戳列表
complete_timestamps = []
for date in date_range:
    for hour in hours:
        complete_timestamps.append(f"{date.strftime('%Y-%m-%d')}_{hour}")

# 创建完整时间序列的DataFrame
complete_df = pd.DataFrame({'时间戳': complete_timestamps})
complete_df['日期'] = complete_df['时间戳'].str.extract(r'(\d{4}-\d{2}-\d{2})')
complete_df['日期'] = pd.to_datetime(complete_df['日期'])
complete_df['时段'] = complete_df['时间戳'].str.extract(r'_(.+)')

print(f"   完整时间序列应有数据点: {len(complete_df)}")
print(f"   日期范围: {date_range.min().strftime('%Y-%m-%d')} 到 {date_range.max().strftime('%Y-%m-%d')}")
print(f"   总天数: {len(date_range)}")

# 数据聚合
print("4. 数据聚合...")

# 首先将两个数据集分别与完整时间序列合并（外连接），补充缺失时间点
market_boundary_full = pd.merge(
    complete_df[['时间戳', '日期', '时段']],
    market_boundary_df,
    on='时间戳',
    how='left',
    suffixes=('', '_orig')
)
# 使用原始日期和时段列（如果存在）
if '日期_orig' in market_boundary_full.columns:
    market_boundary_full['日期'] = market_boundary_full['日期'].fillna(market_boundary_full['日期_orig'])
    market_boundary_full['时段'] = market_boundary_full['时段'].fillna(market_boundary_full['时段_orig'])
    market_boundary_full = market_boundary_full.drop(columns=['日期_orig', '时段_orig'])

# 合并出清价格数据
merged_df = pd.merge(
    market_boundary_full,
    avg_price_df[['时间戳', '平均出清价格-日前（元/MWh）', '平均出清价格-实时（元/MWh）']],
    on='时间戳',
    how='left'
)

# 统计缺失时间点
existing_market = market_boundary_full['时间戳'].isin(market_boundary_df['时间戳']).sum()
existing_price = merged_df['时间戳'].isin(avg_price_df['时间戳']).sum()
print(f"   市场边界信息存在时间点: {existing_market}/{len(complete_df)}")
print(f"   市场边界信息缺失时间点: {len(complete_df) - existing_market}")
print(f"   出清价格存在时间点: {existing_price}/{len(complete_df)}")
print(f"   出清价格缺失时间点: {len(complete_df) - existing_price}")

print(f"   合并后数据形状: {merged_df.shape}")

# 检查数据质量
print("5. 数据质量检查...")
print(f"   合并后数据行数: {len(merged_df)}")
print(f"   日期范围: {merged_df['日期'].min()} 到 {merged_df['日期'].max()}")
print(f"   时段数量: {merged_df['时段'].nunique()}")

# 检查缺失值
missing_data = merged_df.isnull().sum()
print("\n缺失值统计:")
total_missing_cols = 0
for col, missing_count in missing_data.items():
    if missing_count > 0:
        print(f"   {col}: {missing_count} 个缺失值 ({missing_count/len(merged_df)*100:.2f}%)")
        total_missing_cols += 1
if total_missing_cols == 0:
    print("   无缺失值")

# 添加时间特征
print("6. 添加时间特征...")

# 提取日期特征（使用 .dt 访问器前需要确保日期列是 datetime 类型）
# 对于缺失的日期，相关特征将自动为 NaN
merged_df['年'] = merged_df['日期'].dt.year
merged_df['月'] = merged_df['日期'].dt.month
merged_df['日'] = merged_df['日期'].dt.day
merged_df['星期'] = merged_df['日期'].dt.dayofweek + 1  # dayofweek从0开始(周一)，加1后符合中国习惯(1=周一)
merged_df['是否周末'] = merged_df['星期'].isin([6, 7]).astype(int)  # 周末对应6(周六)和7(周日)
merged_df['季度'] = merged_df['日期'].dt.quarter

# 提取时段特征（小时）- 安全处理缺失值
merged_df['小时'] = merged_df['时段'].str.extract(r'(\d+):')
# 转换为数值类型，无法转换的（包括NaN）将变为NaN
merged_df['小时'] = pd.to_numeric(merged_df['小时'], errors='coerce')
# 安全地进行比较（使用 fillna(False) 处理 NaN 值）
merged_df['是否高峰时段'] = ((merged_df['小时'] >= 8) & (merged_df['小时'] <= 20)).fillna(False).astype(int)
merged_df['是否夜间'] = ((merged_df['小时'] >= 22) | (merged_df['小时'] <= 6)).fillna(False).astype(int)

# 重新排列列顺序，使时间相关列在前
cols_order = ['时间戳', '日期', '时段', '年', '月', '日', '星期', '是否周末', '季度', '小时', '是否高峰时段', '是否夜间']
other_cols = [col for col in merged_df.columns if col not in cols_order]
final_cols = cols_order + other_cols
merged_df = merged_df[final_cols]

# 导出数据
print("8. 导出聚合数据...")
merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"   数据已导出到: {output_file}")

# 生成数据摘要
print("\n=== 数据聚合完成 ===")
print(f"最终数据形状: {merged_df.shape}")
print(f"数据列数: {len(merged_df.columns)}")
print(f"时间范围: {merged_df['日期'].min().strftime('%Y-%m-%d')} 到 {merged_df['日期'].max().strftime('%Y-%m-%d')}")
print(f"总天数: {merged_df['日期'].nunique()}")
print(f"总小时数: {len(merged_df)}")

print("\n数据列信息:")
for i, col in enumerate(merged_df.columns, 1):
    dtype = merged_df[col].dtype
    print(f"{i:2d}. {col} ({dtype})")

print(f"\n数据文件大小: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")