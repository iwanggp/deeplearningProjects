"""
solve_offline.py — 真机离线手眼标定求解

输入:
  images/img_NN.png   — capture.py 保存的照片
  poses.csv           — 人工从示教器抄录的法兰位姿
  intrinsics.npz      — capture.py 自动保存的相机内参

输出:
  T_flange_cam.npy    — 手眼矩阵(4x4)
  逐帧离群检测报告     — 自动定位抄错的行
"""
import csv
import os
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as Rsc

# ───── 参数(只需改这一行) ────────────────────────────────
SQUARE = 0.025   # !! 改成你卡尺实测的方格边长(m),打印机缩放会导致实际值≠设计值

COLS, ROWS = 11, 8   # 内角点数(方格阵列12x9 -> 内角点11x8)

# ───── 工具函数 ──────────────────────────────────────────
def make_T(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t; return T

# ───── 加载内参 ──────────────────────────────────────────
intr = np.load("intrinsics.npz")
K    = np.array([[intr["fx"], 0,          intr["cx"]],
                 [0,          intr["fy"], intr["cy"]],
                 [0,          0,          1.0       ]])
dist = np.asarray(intr["dist"], float)
print(f"内参: fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")

# ───── 棋盘格三维角点 ────────────────────────────────────
objp = np.zeros((ROWS * COLS, 3), np.float32)
objp[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2) * SQUARE

# ───── 读取 poses.csv ────────────────────────────────────
rows = []
with open("poses.csv") as f:
    for r in csv.reader(f):
        if r and not r[0].strip().startswith("#"):
            rows.append(r)

print(f"poses.csv: {len(rows)} 行数据")

# ───── 逐帧处理 ──────────────────────────────────────────
R_g2b, t_g2b = [], []   # T_base_flange
R_t2c, t_t2c = [], []   # T_cam_board
used_idx      = []

for r in rows:
    idx = int(r[0])
    path = f"images/img_{idx:02d}.png"
    img  = cv2.imread(path)

    if img is None:
        print(f"[跳过] {path} 不存在"); continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCornersSB(gray, (COLS, ROWS))
    if not ok:
        print(f"[跳过] img_{idx:02d}: 角点检测失败"); continue

    ok2, rvec, tvec = cv2.solvePnP(objp, corners, K, dist)
    if not ok2:
        print(f"[跳过] img_{idx:02d}: PnP失败"); continue

    # UR 位姿: [X_mm, Y_mm, Z_mm, RX, RY, RZ(旋转矢量,弧度)]
    t_g2b.append(np.array(r[1:4], float) / 1000.0)   # mm -> m
    R_g2b.append(Rsc.from_rotvec(np.array(r[4:7], float)).as_matrix())
    R_t2c.append(cv2.Rodrigues(rvec)[0])
    t_t2c.append(tvec.ravel())
    used_idx.append(idx)

print(f"有效帧: {len(used_idx)}/{len(rows)}  -> {used_idx}")
assert len(used_idx) >= 10, "有效帧不足10,补拍或检查角点检测失败原因"

# ───── 求解手眼矩阵 ──────────────────────────────────────
R_c2g, t_c2g = cv2.calibrateHandEye(
    R_g2b, t_g2b, R_t2c, t_t2c,
    method=cv2.CALIB_HAND_EYE_TSAI)

T_fc = make_T(R_c2g, t_c2g.ravel())
np.save("T_flange_cam.npy", T_fc)

print(f"\nT_flange_cam =\n{np.round(T_fc, 5)}")
print(f"\n平移量(mm): {np.round(T_fc[:3,3]*1000, 2)}")

# ───── 固定点散布验证 + 逐帧离群检测 ────────────────────
pos = []
for Rg, tg, Rt, tt in zip(R_g2b, t_g2b, R_t2c, t_t2c):
    T_bf = make_T(Rg, tg)
    T_cb = make_T(Rt, tt)
    pos.append((T_bf @ T_fc @ T_cb)[:3, 3])
pos = np.array(pos)
dev = np.linalg.norm(pos - pos.mean(0), axis=1) * 1000   # mm

median_dev = np.median(dev)
threshold  = max(3 * median_dev + 1.0, 3.0)   # 动态阈值

print(f"\n固定点散布: std={np.round(pos.std(0)*1000,2)} mm  max={dev.max():.2f} mm")
print(f"板原点(基座系,mm) = {np.round(pos.mean(0)*1000, 1)}  <- 探针触点测试目标坐标\n")
print("逐帧偏差:")
any_outlier = False
for i, d in zip(used_idx, dev):
    flag = ""
    if d > threshold:
        flag = "  <-- ⚠️ 离群!检查该行抄数"
        any_outlier = True
    print(f"  帧 {i:02d}: {d:6.2f} mm{flag}")

if not any_outlier:
    print("\n无离群帧 ✅")

# ───── 最终判定 ──────────────────────────────────────────
print("\n" + "="*50)
if dev.max() <= 3.0 and not any_outlier:
    print("✅ 标定合格(散布 ≤ 3mm,无离群)")
elif dev.max() <= 5.0 and not any_outlier:
    print("⚠️  散布 3~5mm,可用但建议重标")
else:
    print("❌ 标定不合格")
    if any_outlier:
        print("   -> 先处理上面标记的离群帧,改正抄数后重跑")
    else:
        print("   -> 检查:SQUARE是否用卡尺实测值 / 板是否平整 / TCP是否置零")
