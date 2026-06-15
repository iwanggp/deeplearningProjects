"""
capture.py — D435i + UR5 手眼标定采集

流程:
  1. 按【f】进入 freedrive → 用手推机械臂到合适位置
  2. 再按【f】退出 freedrive → 机械臂锁定
  3. 实时查看角点检测结果(绿色=检到 / 红色=未检到+原因)
  4. 角点OK后按【空格】→ 自动保存图片(原图+角点图) + 从机械臂读取位姿写入 poses.csv
  5. 重复直到采够帧数(建议15~20帧),按 q 退出

支持断点续拍(自动接续编号)
"""
import os
import platform
import csv
import numpy as np

if platform.system() == "Darwin":
    os.environ.setdefault("OPENCV_OPENCL_DEVICE", "disabled")

import cv2
import pyrealsense2 as rs

cv2.ocl.setUseOpenCL(False)

# ── 配置 ──────────────────────────────────────────────────────────────────
UR_IP  = "169.254.174.10"  # UR5 实际 IP
COLS, ROWS = 11, 8
W, H, FPS  = 1280, 720, 30
CORNER_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# ── 角点检测 ──────────────────────────────────────────────────────────────
def detect_corners(gray):
    ok, corners = cv2.findChessboardCornersSB(gray, (COLS, ROWS))
    if ok:
        return True, corners, "SB"
    ok, corners = cv2.findChessboardCorners(gray, (COLS, ROWS))
    if ok:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CORNER_CRITERIA)
        return True, corners, "classic"
    return False, None, None


def diagnose(gray):
    mean, std = gray.mean(), gray.std()
    if mean < 50:  return f"过暗(亮度={mean:.0f})"
    if mean > 210: return f"过曝(亮度={mean:.0f})"
    if std  < 20:  return f"对比度不足(std={std:.0f})"
    return "棋盘格超出画面或角度过大"


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
        print(f"   TCP    : 已置零 ✓  按【f】切换 freedrive")
    except Exception as ce:
        rtde_c = None
        print(f"   控制接口不可用({ce})")
        print(f"   → 本地模式: 用示教器移动机械臂即可,位姿仍自动读取")

    print(f"✅ UR5 已连接({UR_IP})  位姿自动读取 ✓")
    print(f"   坐标系 : Base — RTDE 协议固定")
    print(f"   姿态格式: Rotation Vector(弧度) — RTDE 协议固定")

except ImportError:
    print("⚠️  未安装 ur-rtde(pip install ur-rtde)")
    print("   退回手动模式,需在示教器手动将 TCP 配置全部设为 0")
    rtde_r = None
except Exception as e:
    print(f"⚠️  UR5 连接失败: {e}")
    print("   退回手动模式,需在示教器手动将 TCP 配置全部设为 0")
    rtde_r = None


def read_pose():
    """读取当前法兰位姿(TCP已置零,getActualTCPPose即法兰位姿)。"""
    p = rtde_r.getActualTCPPose()   # [x,y,z,rx,ry,rz] 米/弧度
    return [p[0]*1000, p[1]*1000, p[2]*1000, p[3], p[4], p[5]]


# ── 连接相机 ──────────────────────────────────────────────────────────────
pipe = rs.pipeline()
cfg  = rs.config()
cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
profile = pipe.start(cfg)

print("相机预热中(自动曝光稳定)...")
for _ in range(90):
    pipe.wait_for_frames()

for sensor in profile.get_device().query_sensors():
    if "RGB" in sensor.get_info(rs.camera_info.name):
        cur_exp = sensor.get_option(rs.option.exposure)
        sensor.set_option(rs.option.enable_auto_exposure, 0)
        sensor.set_option(rs.option.exposure, cur_exp)
        print(f"曝光已锁定  exposure={cur_exp:.0f}")

# 保存内参
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
np.savez("intrinsics.npz",
         fx=intr.fx, fy=intr.fy, cx=intr.ppx, cy=intr.ppy,
         dist=np.array(intr.coeffs, float))
print(f"内参已保存  fx={intr.fx:.1f}  fy={intr.fy:.1f}")

# ── 初始化文件 ────────────────────────────────────────────────────────────
os.makedirs("images", exist_ok=True)

idx = 1
while os.path.exists(f"images/img_{idx:02d}.png"):
    idx += 1
print(f"从第 {idx} 张开始")

if not os.path.exists("poses.csv"):
    with open("poses.csv", "w", newline="") as f:
        csv.writer(f).writerow(["# 序号", "X_mm", "Y_mm", "Z_mm", "RX", "RY", "RZ"])

print("\n【f】freedrive开/关  【空格】保存  【q】退出\n")

# ── 主循环 ────────────────────────────────────────────────────────────────
freedrive_on = False
cv2.namedWindow("handeye  [f:freedrive / 空格:保存 / q:退出]", cv2.WINDOW_NORMAL)
cv2.resizeWindow("handeye  [f:freedrive / 空格:保存 / q:退出]", 1280, 720)

try:
    while True:
        frames = pipe.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        img  = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ok, corners, method = detect_corners(gray)
        vis = img.copy()

        if ok:
            cv2.drawChessboardCorners(vis, (COLS, ROWS), corners, True)
            cv2.putText(vis, f"[{idx}] 角点OK [{method}]  ← 空格保存",
                        (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(vis, f"[{idx}] {diagnose(gray)}",
                        (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # 右上角：已保存帧数 + freedrive 状态
        cv2.putText(vis, f"已保存: {idx-1} 帧",
                    (W - 230, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
        if freedrive_on:
            cv2.putText(vis, "FREEDRIVE  按f锁定",
                        (W - 320, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

        cv2.imshow("handeye  [f:freedrive / 空格:保存 / q:退出]", vis)
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
            cv2.imwrite(f"images/img_{idx:02d}.png", img)
            cv2.imwrite(f"images/img_{idx:02d}_corners.png", vis)

            if rtde_r:
                pose = read_pose()
                with open("poses.csv", "a", newline="") as f:
                    csv.writer(f).writerow([idx] + [f"{v:.4f}" for v in pose])
                print(f"✅ 帧 {idx:02d}  "
                      f"({pose[0]:.1f}, {pose[1]:.1f}, {pose[2]:.1f}) mm  "
                      f"RV=({pose[3]:.4f}, {pose[4]:.4f}, {pose[5]:.4f})")
            else:
                print(f"✅ 帧 {idx:02d} 已保存 → 请在 poses.csv 填入第{idx}行位姿")

            idx += 1

finally:
    if freedrive_on and rtde_c:
        rtde_c.endTeachMode()
    pipe.stop()
    if rtde_c:
        rtde_c.disconnect()
    if rtde_r:
        rtde_r.disconnect()
    cv2.destroyAllWindows()
    print(f"\n退出,共保存 {idx-1} 张")
    if idx - 1 >= 10:
        print("可运行:  python solve_offline.py")
