import time
import math
import sys
import cv2
import os
import datetime
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
import UE4CtrlAPI
import VisionCaptureApi
import PX4MavCtrlV4 as PX4MavCtrl

# 初始化API
ue = UE4CtrlAPI.UE4CtrlAPI()
vis = VisionCaptureApi.VisionCaptureApi()
mav = PX4MavCtrl.PX4MavCtrler(20100)

# 发送消息给RflySim3D，让其将当前收到的飞机数据转发出来，回传到组播地址224.0.0.10的20006端口
ue.sendUE4Cmd('RflyReqVehicleData 1')
time.sleep(0.5)
# 注：只有飞机位置发生改变时，才会将位置数据传出，因此本语句要放在最前面，确保后续创建的物体（Python一次性创建）都能被传出

# Python开始飞机数据的监听，数据存储在inReqUpdateVect列表（是否更新标志），和inReqVect列表（碰撞数据）中
ue.initUE4MsgRec()
time.sleep(2)
# 注意：监听语句应该放到sendUE4系列语句之前，不然无法捕获创建的障碍物
class WindTurbineTracker:
    def __init__(self):
        # 风电机组真实位置（世界坐标系）
        self.turbine_positions = {
            2: [50, 5, 0], 3: [150, -10, 0], 4: [250, 20, 0],
            5: [50, 105, 0], 6: [150, 115, 0], 7: [250, 85, 0],
            8: [50, -95, 0], 9: [150, -80, 0], 10: [250, -110, 0]
        }

        # 跟踪状态管理
        self.track_history = {}  # track_id -> {'copter_id': x, 'frames': [], 'bind_time': None}
        self.current_bindings = {}  # track_id -> copter_id
        self.binding_stats = {
            'rebind_delays': [],
            'bind_durations': [],
            'wrong_bindings': 0,
            'total_bindings': 0
        }

        # 性能指标记录
        self.performance_metrics = {
            'frame_count': 0,
            'detection_history': defaultdict(list)
        }

        # 相机参数（根据文档2调整）
        self.camera_params = {
            'focal_length': 320,
            'image_width': 640,
            'image_height': 480,
            'principal_point': (320, 240)
        }

        # 无人机初始位置
        self.uav_initial_pos = [-8.086, 0, 0]

    def eul2rot(self, theta):
        """欧拉角转旋转矩阵"""
        R_x = np.array([
            [1, 0, 0],
            [0, math.cos(theta[0]), math.sin(theta[0])],
            [0, -math.sin(theta[0]), math.cos(theta[0])]
        ])

        R_y = np.array([
            [math.cos(theta[1]), 0, -math.sin(theta[1])],
            [0, 1, 0],
            [math.sin(theta[1]), 0, math.cos(theta[1])]
        ])

        R_z = np.array([
            [math.cos(theta[2]), math.sin(theta[2]), 0],
            [-math.sin(theta[2]), math.cos(theta[2]), 0],
            [0, 0, 1]
        ])

        return np.dot(R_x, np.dot(R_y, R_z))

    def pixel_to_world(self, bbox_center, bbox_size, uav_position, uav_orientation,
                                 known_object_width=11.361):
        """
        基于针孔相机模型的正确坐标转换
        known_object_width: 风电机组塔筒的大致宽度（米），可根据实际情况调整
        """
        center_x, center_y = bbox_center
        bbox_width, bbox_height = bbox_size

        # 使用相似三角形原理计算距离
        # 距离 = (焦距 × 实际物体宽度) / 图像中的像素宽度
        if bbox_width > 0:
            distance = (self.camera_params['focal_length'] * known_object_width) / bbox_width
        else:
            distance = 50  # 默认距离

        # 归一化像素坐标（相机坐标系）
        x_normalized = (center_x - self.camera_params['principal_point'][0]) / self.camera_params['focal_length']
        y_normalized = (center_y - self.camera_params['principal_point'][1]) / self.camera_params['focal_length']

        # 相机坐标系中的3D坐标（Z轴向前）
        # 注意：OpenCV相机坐标系通常是Z向前，Y向下，X向右
        x_cam = x_normalized * distance
        y_cam = y_normalized * distance
        z_cam = distance

        # 构建齐次坐标
        point_camera = np.array([z_cam, x_cam, y_cam])

        # 获取旋转矩阵（从相机到世界）
        rotation_matrix = self.eul2rot(uav_orientation)

        # 相机坐标系到世界坐标系的转换
        # 注意：这里需要考虑无人机姿态的影响
        point_world = np.dot(rotation_matrix, point_camera) + np.array(uav_position) + np.array([0.03,0,0.15])

        return point_world

    def find_nearest_turbine(self, world_position):
        """找到距离最近的风电机组"""
        min_distance = float('inf')
        nearest_id = None

        for copter_id, turbine_pos in self.turbine_positions.items():
            distance = np.linalg.norm(np.array(world_position) - np.array(turbine_pos))
            if distance < min_distance:
                min_distance = distance
                nearest_id = copter_id

        return nearest_id, min_distance

    def update_tracking(self, track_id, bbox, uav_position, uav_orientation):
        """更新跟踪状态"""
        # 计算边界框中心和大小的像素坐标
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1

        # 转换为世界坐标
        world_pos = self.pixel_to_world(
            (center_x, center_y), (width, height),
            uav_position, uav_orientation
        )

        # 找到最近的风电机组
        nearest_id, distance = self.find_nearest_turbine(world_pos)

        # 距离阈值判断（避免过远匹配）
        binding_threshold = 50  # 米
        if distance > binding_threshold:
            return None, distance

        current_time = self.performance_metrics['frame_count']

        # 更新跟踪历史
        if track_id not in self.track_history:
            self.track_history[track_id] = {
                'copter_id': nearest_id,
                'frames': [current_time],
                'bind_time': current_time,
                'last_bind_time': current_time
            }
        else:
            self.track_history[track_id]['frames'].append(current_time)

            # 检查是否需要重新绑定
            if self.track_history[track_id]['copter_id'] != nearest_id:
                # 记录重绑定延迟
                rebind_delay = current_time - self.track_history[track_id]['last_bind_time']
                self.binding_stats['rebind_delays'].append(rebind_delay)

                # 更新绑定
                self.track_history[track_id]['copter_id'] = nearest_id
                self.track_history[track_id]['last_bind_time'] = current_time

                # 检查是否为错误绑定（基于历史记录）
                if len(self.track_history[track_id]['frames']) > 10:
                    self.binding_stats['wrong_bindings'] += 1

        # 更新当前绑定
        self.current_bindings[track_id] = nearest_id
        self.binding_stats['total_bindings'] += 1

        return nearest_id, distance

    def calculate_binding_duration_corrected(self, max_gap_frames=3):
        """
        计算平均绑定时间（考虑绑定中断阈值）
        max_gap_frames: 最大允许的帧间隔，超过此间隔认为绑定中断
        """
        binding_durations = []

        for track_id, history in self.track_history.items():
            if len(history['frames']) < 2:
                continue  # 需要至少2帧才能计算持续时间

            frames = sorted(history['frames'])  # 确保帧号有序
            copter_id = history['copter_id']

            # 查找连续的绑定段
            binding_segments = self._find_continuous_binding_segments(
                frames, copter_id, max_gap_frames
            )

            # 计算每个绑定段的持续时间
            for segment in binding_segments:
                if len(segment) > 1:  # 至少2帧才有持续时间
                    duration = segment[-1] - segment[0]
                    binding_durations.append(duration)

        if binding_durations:
            return np.mean(binding_durations)
        else:
            return 0

    def _find_continuous_binding_segments(self, frames, copter_id, max_gap):
        """
        在帧序列中查找连续的绑定段
        返回: 列表的列表，每个子列表是一个连续绑定段
        """
        if not frames:
            return []

        segments = []
        current_segment = [frames[0]]

        for i in range(1, len(frames)):
            gap = frames[i] - frames[i - 1]

            if gap <= max_gap:
                # 帧间隔在阈值内，属于同一绑定段
                current_segment.append(frames[i])
            else:
                # 帧间隔超过阈值，开始新的绑定段
                if len(current_segment) > 1:  # 只保存有持续时间的段
                    segments.append(current_segment)
                current_segment = [frames[i]]

        # 添加最后一个段
        if len(current_segment) > 1:
            segments.append(current_segment)

        return segments
    def calculate_metrics(self):
        """计算性能指标"""
        metrics = {}

        # 平均重绑定延迟
        if self.binding_stats['rebind_delays']:
            metrics['avg_rebind_delay'] = np.mean(self.binding_stats['rebind_delays'])
        else:
            metrics['avg_rebind_delay'] = 0

        # 平均绑定时间
        metrics['avg_bind_duration'] = self.calculate_binding_duration_corrected()


        # 错误绑定率
        if self.binding_stats['total_bindings'] > 0:
            metrics['wrong_binding_rate'] = (self.binding_stats['wrong_bindings'] /
                                             self.binding_stats['total_bindings'] * 100)
        else:
            metrics['wrong_binding_rate'] = 0

        return metrics

    def visualize_results(self, frame, tracks, metrics, uav_position):
        """可视化结果显示"""
        # 绘制检测框和跟踪信息
        for track in tracks:
            if hasattr(track, 'boxes'):
                bbox = track.boxes.xyxy[0].cpu().numpy()
                track_id = int(track.boxes.id[0]) if track.boxes.id is not None else 0

                x1, y1, x2, y2 = map(int, bbox)

                # 获取绑定信息
                copter_id = self.current_bindings.get(track_id, 'Unknown')
                color = (0, 255, 0) if copter_id != 'Unknown' else (0, 0, 255)

                # 绘制边界框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # 显示跟踪ID和绑定的风电机组ID
                label = f"Track{track_id}->Turbine{copter_id}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 显示性能指标
        y_offset = 30
        for metric_name, metric_value in metrics.items():
            text = f"{metric_name}: {metric_value:.2f}"
            cv2.putText(frame, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25

        # 显示无人机位置
        pos_text = f"UAV Position: ({uav_position[0]:.1f}, {uav_position[1]:.1f})"
        cv2.putText(frame, pos_text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame


def main():
    # 创建风电机组布局（同文档1）
    turbine_positions = [
        [50, 5, 0], [150, -10, 0], [250, 20, 0],  # 中间列
        [50, 105, 0], [150, 115, 0], [250, 85, 0],  # 左边列
        [50, -95, 0], [150, -80, 0], [250, -110, 0]  # 右边列
    ]

    for i, pos in enumerate(turbine_positions, 1):
        ue.sendUE4Pos2Ground(i+1, 113, 0, pos, [0, 0, 0])

    # 设置UE4窗口
    ue.sendUE4Cmd('r.setres 720x405w', 0)
    ue.sendUE4Cmd('t.MaxFPS 30', 0)
    time.sleep(2)

    # 初始化无人机模型
    mav.initPointMassModel(-8.086, [0, 0, 0])
    time.sleep(2)
    a=ue.getUE4Pos(1)
    # 配置视觉捕获
    vis.jsonLoad()
    isSuss = vis.sendReqToUE4()
    if not isSuss:
        print("错误：无法连接到RflySim3D")
        sys.exit(0)
    vis.startImgCap(True)
    time.sleep(1)

    # 加载YOLO模型（使用BoT-SORT跟踪器）
    model = YOLO(r"E:\paper\YOLOv8-DeepSORT-Object-Tracking-main\ultralytics-main\runs\train\yolo12s2\weights\best.pt")

    # 创建跟踪器实例
    tracker = WindTurbineTracker()

    # 创建视频保存
    video_dir = "tracking_results"
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = os.path.join(video_dir, f"tracking_{timestamp}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_filename, fourcc, 30, (640, 480))

    print("开始风电机组跟踪...")

    # 飞行控制（同文档1）
    print("无人机起飞...")
    mav.SendVelNED(0, 0, -2, 0)

    # 飞行状态跟踪
    start_time = time.time()
    flight_phase = "takeoff"
    takeoff_duration = 7
    cruise_duration = 50

    # 时间控制
    lastTime = time.time()
    timeInterval = 1 / 30.0

    print("开始仿真跟踪，按'q'键退出...")

    try:
        while True:
            # 时间控制
            lastTime += timeInterval
            sleepTime = lastTime - time.time()
            if sleepTime > 0:
                time.sleep(sleepTime)
            else:
                lastTime = time.time()

            # 飞行状态转换
            current_time = time.time() - start_time
            if flight_phase == "takeoff" and current_time >= takeoff_duration:
                print("无人机向北飞行...")
                mav.SendVelNED(3, 0, 0, 0)
                flight_phase = "cruise"
            elif flight_phase == "cruise" and current_time >= takeoff_duration + cruise_duration:
                print("结束仿真...")
                mav.EndPointMassModel()
                break

            # 获取图像帧
            if vis.hasData[0]:
                frame = vis.Img[0]
                frame_resized = cv2.resize(frame, (640, 480))

                # 更新帧计数
                tracker.performance_metrics['frame_count'] += 1

                # 获取无人机当前位置和姿态（简化处理）
                uav_position = ue.getUE4Pos(1)[:3]
                uav_orientation = [0, 0, 0]  # 简化处理，实际应从MAVLink获取

                # YOLO检测与跟踪
                results = model.track(
                    source=frame_resized,
                    conf=0.3,
                    iou=0.5,
                    persist=True,
                    tracker="botsort.yaml"  # 使用BoT-SORT跟踪器
                )

                # 处理跟踪结果
                if results and len(results) > 0:
                    tracks = results[0]

                    # 更新跟踪状态
                    for track in tracks:
                        if hasattr(track, 'boxes') and track.boxes.id is not None:
                            bbox = track.boxes.xyxy[0].cpu().numpy()
                            track_id = int(track.boxes.id[0])

                            # 更新跟踪绑定
                            copter_id, distance = tracker.update_tracking(
                                track_id, bbox, uav_position, uav_orientation
                            )

                # 计算当前性能指标
                metrics = tracker.calculate_metrics()

                # 可视化结果
                visualized_frame = tracker.visualize_results(
                    frame_resized.copy(),
                    tracks if 'tracks' in locals() else [],
                    metrics,
                    uav_position
                )

                # 保存和显示视频
                out.write(visualized_frame)
                cv2.imshow('Wind Turbine Tracking', visualized_frame)

                # 打印性能指标（每30帧）
                if tracker.performance_metrics['frame_count'] % 30 == 0:
                    print(f"\n=== 第{tracker.performance_metrics['frame_count']}帧性能指标 ===")
                    for metric_name, metric_value in metrics.items():
                        print(f"{metric_name}: {metric_value:.2f}")
                    print("当前绑定:", tracker.current_bindings)

                # 退出检查
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("用户中断仿真...")
    except Exception as e:
        print(f"发生错误: {e}")

    finally:
        # 最终性能报告
        final_metrics = tracker.calculate_metrics()
        print("\n=== 最终性能报告 ===")
        for metric_name, metric_value in final_metrics.items():
            print(f"{metric_name}: {metric_value:.2f}")

        # 保存性能数据
        metrics_filename = os.path.join(video_dir, f"metrics_{timestamp}.txt")
        with open(metrics_filename, 'w') as f:
            for metric_name, metric_value in final_metrics.items():
                f.write(f"{metric_name}: {metric_value:.2f}\n")

        # 释放资源
        out.release()
        cv2.destroyAllWindows()
        print(f"跟踪结果已保存至: {video_filename}")
        print("仿真完成!")


if __name__ == "__main__":
    main()