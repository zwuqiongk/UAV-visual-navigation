import time
import math
import sys
import cv2
import os
import datetime
import UE4CtrlAPI

ue = UE4CtrlAPI.UE4CtrlAPI()
import VisionCaptureApi  # 视觉捕获API
import PX4MavCtrlV4 as PX4MavCtrl

# 创建视觉捕获API实例
vis = VisionCaptureApi.VisionCaptureApi()

# 创建风电机组布局
# 中间一列
ue.sendUE4Pos2Ground(10, 113, 0, [0+50, 0+5, 0], [0, 0, 0])  # 创建ID=1的风电机组
ue.sendUE4Pos2Ground(2, 113, 0, [100+50, -15, 0], [0, 0, 0])  # 创建ID=2的风电机组
ue.sendUE4Pos2Ground(3, 113, 0, [200+50, 15, 0], [0, 0, 0])  # 创建ID=3的风电机组
# 左边一列
ue.sendUE4Pos2Ground(4, 113, 0, [0+50, 100, 0], [0, 0, 0])  # 创建ID=4的风电机组
ue.sendUE4Pos2Ground(5, 113, 0, [100+50, 115, 0], [0, 0, 0])  # 创建ID=5的风电机组
ue.sendUE4Pos2Ground(6, 113, 0, [200+50, 85, 0], [0, 0, 0])  # 创建ID=6的风电机组
# 右边一列
ue.sendUE4Pos2Ground(7, 113, 0, [0+50, -100, 0], [0, 0, 0])  # 创建ID=7的风电机组
ue.sendUE4Pos2Ground(8, 113, 0, [100+50, -85, 0], [0, 0, 0])  # 创建ID=8的风电机组
ue.sendUE4Pos2Ground(9, 113, 0, [200+50, -115, 0], [0, 0, 0])  # 创建ID=9的风电机组

# 创建MAVLink通信实例，UDP发送端口为20100
mav = PX4MavCtrl.PX4MavCtrler(20100)

# 设置UE4窗口分辨率为720x405（窗口模式）
ue.sendUE4Cmd('r.setres 720x405w', 0)
# 设置UE4最大刷新频率为30FPS（也是图像捕获频率）
ue.sendUE4Cmd('t.MaxFPS 30', 0)

time.sleep(2)  # 等待2秒确保设置生效

# 初始化点质量模型，设置初始位置和姿态
mav.initPointMassModel( -8.086, [0, 0, 0])

time.sleep(2)

# VisionCaptureApi配置函数
vis.jsonLoad()  # 加载Config.json中的传感器配置文件

isSuss = vis.sendReqToUE4()  # 向RflySim3D发送图像捕获请求并验证
if not isSuss:  # 如果请求失败则退出程序
    print("错误：无法连接到RflySim3D，请检查配置")
    sys.exit(0)
vis.startImgCap(True)  # 开启图像捕获，并启用共享内存图像转发

time.sleep(1)  # 等待1秒确保图像捕获启动

# 创建视频保存目录和文件
video_dir = "simulation_videos"
if not os.path.exists(video_dir):
    os.makedirs(video_dir)

# 生成带时间戳的视频文件名
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
video_filename = os.path.join(video_dir, f"simulation_{timestamp}.avi")

# 设置视频编码器和参数
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 30  # 帧率
frame_size = (640, 480)  # 帧大小，与UE4窗口分辨率一致

# 创建VideoWriter对象
out = cv2.VideoWriter(video_filename, fourcc, fps, frame_size)

print(f"开始录制视频，保存路径: {video_filename}")

# 初始化飞行控制
print("无人机起飞...")
mav.SendVelNED(0, 0, -2, 0)  # 以2m/s速度起飞

# 飞行状态跟踪
start_time = time.time()
flight_phase = "takeoff"
takeoff_duration = 7  # 起飞阶段持续时间
cruise_duration = 50  # 巡航阶段持续时间

# 时间控制变量
lastTime = time.time()
timeInterval = 1 / 30.0  # 约33ms，对应30FPS

print("开始仿真，按'q'键可提前退出...")

# 主循环
while True:
    # 计算下一帧的理想时间
    lastTime = lastTime + timeInterval
    # 计算需要休眠的时间
    sleepTime = lastTime - time.time()
    # 如果需要休眠则休眠
    if sleepTime > 0:
        time.sleep(sleepTime)
    else:
        # 如果落后于计划时间则重置计时
        lastTime = time.time()

    # 检查飞行状态转换
    current_time = time.time() - start_time
    if flight_phase == "takeoff" and current_time >= takeoff_duration:
        print("无人机向北飞行...")
        mav.SendVelNED(3, 0, 0, 0)  # 以3m/s速度向北飞行
        flight_phase = "cruise"
    elif flight_phase == "cruise" and current_time >= takeoff_duration + cruise_duration:
        print("结束仿真...")
        mav.EndPointMassModel()
        break

    # 如果有图像数据，处理图像
    if vis.hasData[0]:
        # 获取当前帧
        frame = vis.Img[0]

        # 调整帧大小以匹配视频设置
        if frame.shape[1] != frame_size[0] or frame.shape[0] != frame_size[1]:
            frame = cv2.resize(frame, frame_size)

        # 写入视频文件
        out.write(frame)

        # 显示图像
        cv2.imshow('Simulation View', frame)

        # 检查按键输入
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("用户请求提前终止...")
            break

# 释放资源
out.release()
cv2.destroyAllWindows()
print(f"视频已保存至: {video_filename}")
print("仿真完成!")