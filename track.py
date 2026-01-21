from ultralytics import YOLO

# Configure the tracking parameters and run the tracker
model = YOLO(r"E:\paper\YOLOv8-DeepSORT-Object-Tracking-main\ultralytics-main\runs\train\yolo12s2\weights\best.pt")
#results = model.track(source=r"E:\paper\YOLOv8-DeepSORT-Object-Tracking-main\ultralytics-main\4.mp4", conf=0.3, iou=0.5, show=True)
results = model.track(source=r"E:\paper\YOLOv8-DeepSORT-Object-Tracking-main\rflysim\simulation_videos\simulation_20251017_213032.avi", conf=0.3, iou=0.5, show=True,save=True, tracker="bytetrack.yaml")  # with ByteTrack
