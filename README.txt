【真机标定文件说明】

流程:
  1. 打印 checker_12x9.png (缩放100%实际大小)
  2. 卡尺实测方格边长,填入 solve_offline.py 的 SQUARE
  3. 修改 capture.py 顶部的 UR_IP 为机械臂实际 IP
  4. python capture.py
       手动移动机械臂 → 画面角点变绿 → 按空格保存图片+位姿
       重复15~20次,按 q 退出
  5. python solve_offline.py → 求解并验证

依赖:
  pip install pyrealsense2 opencv-python numpy scipy
  pip install ur-rtde      # 自动读取位姿 + 自动置零 TCP(强烈推荐)

启动时自动验证并打印:
  [TCP]       capture.py 启动时调用 setTcp([0,0,0,0,0,0]) 自动置零
  [坐标系]    RTDE 协议固定返回 Base(基座系),无需配置
  [姿态格式]  RTDE 协议固定返回 Rotation Vector(弧度),无需配置

  → 三条约束只有 TCP 需要处理,其余 RTDE 自动满足
  → 示教器上的坐标系/姿态显示设置只影响屏幕显示,不影响 RTDE 读值

未安装 ur-rtde 时(手动模式):
  TCP 须在示教器手动确认: 安装 → TCP配置 → 全部设为0
  坐标系和姿态格式无需操心(poses.csv 由程序直接填写)
