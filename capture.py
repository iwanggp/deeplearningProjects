"""
capture.py — D435i + UR5 手眼标定采集

流程:
  1. 手动移动机械臂到合适位置
  2. 实时查看角点检测结果(绿色=检到 / 红色=未检到+原因)
  3. 角点OK后按【空格】→ 自动保存图片 + 从机械臂读取位姿写入 poses.csv
  4. 重复直到采够帧数(建议15~20帧),按 q 退出

支持断点续拍(自动接续编号)
"""
import os
import csv
import numpy as np
import cv2
import pyrealsense2 as rs

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

try:
    import rtde_receive
    import rtde_control

    rtde_r = rtde_receive.RTDEReceiveInterface(UR_IP)
    if not rtde_r.isConnected():
        raise RuntimeError("receive 连接失败")

    # setTcp 设置 TCP 偏移量为零,使 getActualTCPPose() 返回法兰位姿
    # setTcp 不抛异常即代表成功,设置在断开后仍持续生效
    rtde_c = rtde_control.RTDEControlInterface(UR_IP)
    rtde_c.setTcp([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    rtde_c.disconnect()

    print(f"✅ UR5 已连接({UR_IP})")
    print(f"   TCP    : 已置零 ✓  (setTcp 成功)")
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

print("相机预热中...")
for _ in range(60):
    pipe.wait_for_frames()

for sensor in profile.get_device().query_sensors():
    if "RGB" in sensor.get_info(rs.camera_info.name):
        sensor.set_option(rs.option.enable_auto_exposure, 0)
        print("曝光已锁定")

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

print("\n移动机械臂 → 角点变绿 → 按【空格】保存 → 移动到下一个位置")
print("按 q 退出\n")

# ── 主循环 ────────────────────────────────────────────────────────────────
try:
    while True:
        img  = np.asanyarray(pipe.wait_for_frames().get_color_frame().get_data())
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

        # 右上角显示已保存帧数
        cv2.putText(vis, f"已保存: {idx-1} 帧",
                    (W - 230, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

        cv2.imshow("handeye  [空格保存 / q退出]", vis)
        k = cv2.waitKey(1) & 0xFF

        if k == ord('q'):
            break

        if k == ord(' ') and ok:
            cv2.imwrite(f"images/img_{idx:02d}.png", img)

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
    pipe.stop()
    if rtde_r:
        rtde_r.disconnect()
    cv2.destroyAllWindows()
    print(f"\n退出,共保存 {idx-1} 张")
    if idx - 1 >= 10:
        print("可运行:  python solve_offline.py")
