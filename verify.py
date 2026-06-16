"""
verify.py — 手眼标定验证

流程:
  1. 标定板固定在桌上不动
  2. 按【f】freedrive → 移动机械臂到新位置 → 按【f】锁定
  3. 角点变绿后按【空格】→ 记录板子原点在基座系坐标
  4. 重复5次以上（位置要有变化）
  5. 按【q】退出，显示散布误差报告
"""
import os
import platform
import numpy as np

if platform.system() == "Darwin":
    os.environ.setdefault("OPENCV_OPENCL_DEVICE", "disabled")

import cv2
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as Rsc

cv2.ocl.setUseOpenCL(False)

# ── 配置 ──────────────────────────────────────────────────────────────────
UR_IP      = "169.254.174.10"
COLS, ROWS = 11, 8
SQUARE     = 0.010       # !! 改成卡尺实测的方格边长(m)
W, H, FPS  = 1280, 720, 30
CORNER_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# ── 加载标定结果 ──────────────────────────────────────────────────────────
T_flange_cam = np.load("T_flange_cam.npy")
intr = np.load("intrinsics.npz")
K    = np.array([[intr["fx"], 0,          intr["cx"]],
                 [0,          intr["fy"], intr["cy"]],
                 [0,          0,          1.0       ]])
dist = np.asarray(intr["dist"], float)
print(f"T_flange_cam 已加载")
print(f"内参: fx={K[0,0]:.1f}  fy={K[1,1]:.1f}")

# ── 棋盘格3D角点（板子坐标系） ────────────────────────────────────────────
objp = np.zeros((ROWS * COLS, 3), np.float32)
objp[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2) * SQUARE


# ── 角点检测 ──────────────────────────────────────────────────────────────
def detect_corners(gray):
    ok, corners = cv2.findChessboardCornersSB(gray, (COLS, ROWS))
    if ok:
        return True, corners
    ok, corners = cv2.findChessboardCorners(gray, (COLS, ROWS))
    if ok:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CORNER_CRITERIA)
        return True, corners
    return False, None


# ── 连接 UR5 ──────────────────────────────────────────────────────────────
rtde_r = None
rtde_c = None

try:
    import rtde_receive
    import rtde_control

    rtde_r = rtde_receive.RTDEReceiveInterface(UR_IP)
    if not rtde_r.isConnected():
        raise RuntimeError("receive 连接失败")

    try:
        rtde_c = rtde_control.RTDEControlInterface(UR_IP)
        rtde_c.setTcp([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        print(f"✅ UR5 已连接  按【f】切换 freedrive")
    except Exception as ce:
        rtde_c = None
        print(f"   控制接口不可用({ce})，用示教器移动机械臂")

except ImportError:
    print("⚠️  未安装 ur-rtde")
    rtde_r = None
except Exception as e:
    print(f"⚠️  UR5 连接失败: {e}")
    rtde_r = None


def get_T_base_flange():
    p = rtde_r.getActualTCPPose()          # TCP=0，即法兰位姿，单位:米/弧度
    R = Rsc.from_rotvec(p[3:6]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = p[:3]
    return T


# ── 连接相机 ──────────────────────────────────────────────────────────────
pipe = rs.pipeline()
cfg  = rs.config()
cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
profile = pipe.start(cfg)

print("相机预热中...")
for _ in range(60):
    pipe.wait_for_frames()
print("就绪\n")

print("【f】freedrive开/关  【空格】记录当前位置  【q】退出显示报告\n")

# ── 主循环 ────────────────────────────────────────────────────────────────
results      = []    # 每次记录的板原点坐标(基座系,mm)
freedrive_on = False

WIN = "verify  [f:freedrive / 空格:记录 / q:退出]"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, 1280, 720)

try:
    while True:
        frames      = pipe.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        img  = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ok, corners = detect_corners(gray)
        vis = img.copy()

        if ok:
            cv2.drawChessboardCorners(vis, (COLS, ROWS), corners, True)
            cv2.putText(vis, f"角点OK  已记录:{len(results)}次  ← 空格记录",
                        (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(vis, f"未检到角点  已记录:{len(results)}次",
                        (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        if freedrive_on:
            cv2.putText(vis, "FREEDRIVE  按f锁定",
                        (W - 320, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

        cv2.imshow(WIN, vis)
        k = cv2.waitKey(10) & 0xFF

        if k == ord('q'):
            break

        if k == ord('f') and rtde_c:
            if not freedrive_on:
                rtde_c.teachMode()
                freedrive_on = True
                print("🟠 freedrive ON  — 可手推机械臂")
            else:
                rtde_c.endTeachMode()
                freedrive_on = False
                print("🔒 freedrive OFF — 机械臂已锁定")

        if k == ord(' ') and ok and not freedrive_on:
            if not rtde_r:
                print("⚠️  未连接 UR5，无法读取位姿")
                continue

            ok2, rvec, tvec = cv2.solvePnP(objp, corners, K, dist)
            if not ok2:
                print("solvePnP 失败，跳过")
                continue

            T_cam_board = np.eye(4)
            T_cam_board[:3, :3] = cv2.Rodrigues(rvec)[0]
            T_cam_board[:3, 3]  = tvec.ravel()

            # base ← flange ← cam ← board
            T_base_board = get_T_base_flange() @ T_flange_cam @ T_cam_board
            origin_mm    = T_base_board[:3, 3] * 1000

            results.append(origin_mm)
            print(f"记录 {len(results):02d}: 板原点 = "
                  f"({origin_mm[0]:.1f}, {origin_mm[1]:.1f}, {origin_mm[2]:.1f}) mm")

finally:
    if freedrive_on and rtde_c:
        rtde_c.endTeachMode()
    pipe.stop()
    if rtde_c:
        rtde_c.disconnect()
    if rtde_r:
        rtde_r.disconnect()
    cv2.destroyAllWindows()

# ── 误差报告 ──────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
if len(results) < 3:
    print("记录不足3次，无法统计")
else:
    pts      = np.array(results)
    mean     = pts.mean(axis=0)
    devs     = np.linalg.norm(pts - mean, axis=1)
    threshold = max(3 * np.median(devs) + 1.0, 3.0)

    print(f"验证报告  共 {len(results)} 次")
    print(f"{'='*50}")
    print(f"板原点均值: ({mean[0]:.1f}, {mean[1]:.1f}, {mean[2]:.1f}) mm")
    print(f"std       : {np.round(pts.std(axis=0), 2)} mm")
    print(f"最大偏差   : {devs.max():.2f} mm\n")

    for i, (p, d) in enumerate(zip(results, devs)):
        flag = "  ⚠️  离群" if d > threshold else ""
        print(f"  {i+1:02d}: ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}) mm  偏差={d:.2f}mm{flag}")

    print()
    if devs.max() <= 3.0:
        print("✅ 标定合格 (最大偏差 ≤ 3mm)")
    elif devs.max() <= 5.0:
        print("⚠️  标定可用，建议重标 (最大偏差 3~5mm)")
    else:
        print("❌ 标定不合格，需重新采集 (最大偏差 > 5mm)")
