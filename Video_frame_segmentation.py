import cv2
import os


def video_to_frames(video_path, output_dir):
    # 创建输出文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("错误：无法打开视频文件")
        return

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"视频信息：")
    print(f" - 帧率: {fps:.2f} FPS")
    print(f" - 总帧数: {total_frames}")
    print(f" - 时长: {duration:.2f} 秒")

    # 读取并保存每一帧
    frame_count = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # 生成文件名（按帧数编号，前面补零）
        filename = f"frame_{frame_count:06d}.jpeg"
        filepath = os.path.join(output_dir, filename)

        # 保存为JPEG格式
        cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        frame_count += 1

        # 显示进度（每100帧打印一次）
        if frame_count % 100 == 0:
            print(f"已处理 {frame_count}/{total_frames} 帧...")

    # 释放资源
    cap.release()

    print(f"完成！共提取 {frame_count} 帧图像")
    print(f"图像保存至: {output_dir}")


if __name__ == "__main__":
    # 视频文件路径
    video_file = r"E:\paper\YOLOv8-DeepSORT-Object-Tracking-main\output1\tracking_result_20251113_151750.mp4"

    # 输出文件夹路径
    output_folder = "video_frames"

    # 执行转换
    video_to_frames(video_file, output_folder)