# 导入必要的库
import random  # 随机数生成
import os  # 操作系统接口
from turtle import shape  # 从turtle库导入shape（可能未使用）
import cv2  # OpenCV计算机视觉库
import time  # 时间控制
import math  # 数学函数
import sys  # 系统相关功能
import datetime  # 日期和时间处理
import numpy as np  # 数值计算库

# 导入RflySim专用API
import VisionCaptureApi  # 视觉捕获API
import UE4CtrlAPI  # UE4控制API

ue = UE4CtrlAPI.UE4CtrlAPI()  # 创建UE4控制接口实例

vis = VisionCaptureApi.VisionCaptureApi()  # 创建视觉捕获API实例
#ue.sendUE4Pos2Ground(1, 113, 0, [0, 0, 0], [0, 0, 0])  # 创建ID=1的飞行器
# ue.sendUE4Pos2Ground(2, 113, 0, [100, 0, 0], [0, 0, 0])  # 创建ID=1的飞行器
# ue.sendUE4Pos2Ground(3, 113, 0, [200, 0, 0], [0, 0, 0])  # 创建ID=1的飞行器
# ue.sendUE4Pos2Ground(4, 113, 0, [300, 0, 0], [0, 0, 0])  # 创建ID=1的飞行器
# ue.sendUE4Pos2Ground(5, 113, 0, [0, 100, 0], [0, 0, 0])  # 创建ID=1的飞行器
# ue.sendUE4Pos2Ground(6, 113, 0, [100, 100, 0], [0, 0, 0])  # 创建ID=1的飞行器
# ue.sendUE4Pos2Ground(7, 0, 0, [-1, 0, 0], [0, 0, 0])  # 创建ID=1的飞行器
# 切换地图到"GrassLands"（草地场景）
ue.sendUE4Cmd('RflyChangeMapbyName GrassLands')
time.sleep(5)  # 等待5秒确保地图加载完成

# 设置UE4窗口分辨率为720x405（窗口模式）
ue.sendUE4Cmd('r.setres 720x405w', 0)
# 设置UE4最大刷新频率为30FPS（也是图像捕获频率）
ue.sendUE4Cmd('t.MaxFPS 30', 0)

time.sleep(2)  # 等待2秒确保设置生效

# 发送位置命令创建一个飞行器（模拟相机）
# 初始位置在地面上方1.5米（草地地图的地面高度为-8.086）
PosInit = [0, 0, -8.086-12]  # 北0米，东0米，高度-12.086米（地面下4米）
angCopterE = [0, 0, 0]  # 欧拉角：滚转0°，俯仰0°，偏航0°
ue.sendUE4Pos(1, 0, 0, PosInit, angCopterE)  # 创建ID=1的飞行器
time.sleep(2)  # 等待2秒确保飞行器创建完成

# VisionCaptureApi配置函数
vis.jsonLoad()  # 加载Config.json中的传感器配置文件

isSuss = vis.sendReqToUE4()  # 向RflySim3D发送图像捕获请求并验证
if not isSuss:  # 如果请求失败则退出程序
    sys.exit(0)
vis.startImgCap(True)  # 开启图像捕获，并启用共享内存图像转发

time.sleep(1)  # 等待1秒确保图像捕获启动

# 发送位置命令创建一个棋盘格（ID=100）
# 棋盘格位于飞行器前方1米处，偏航角90度（面向飞行器）
InitTargePos = [50, 0, -8.086]  # 目标初始位置
InitTargeAng = [0, 0, 0]  # 目标初始角度（偏航90度）
ue.sendUE4Pos(100, 113, 0, InitTargePos, InitTargeAng)  # 创建ID=100的棋盘格目标

# 相机标定后的内参矩阵
intrMatrix = np.matrix([[320, 0, 0],
                        [0, 320, 0],
                        [320, 240, 1]]).T  # 转置后的内参矩阵
copterCenterHeight = 0.15  # 无人机中心高度
cameraPosForUav = [0.03, 0, 0]  # 相机相对于无人机的位置
pic_w = 640  # 图像宽度
pic_h = 480  # 图像高度
focal = 320  # 焦距（像素）
px = pic_w / 2  # 主点x坐标（图像中心）
py = pic_h / 2  # 主点y坐标（图像中心）

time.sleep(1)  # 等待1秒确保目标创建完成


def eul2rot(theta):
    '''
    欧拉角转旋转矩阵，与世界坐标系方向、欧拉角旋转顺序、旋转正方向的定义有关
    '''
    # 创建绕X轴旋转的矩阵
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), math.sin(theta[0])],
                    [0, -math.sin(theta[0]), math.cos(theta[0])]
                    ])

    # 创建绕Y轴旋转的矩阵
    R_y = np.array([[math.cos(theta[1]), 0, -math.sin(theta[1])],
                    [0, 1, 0],
                    [math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    # 创建绕Z轴旋转的矩阵
    R_z = np.array([[math.cos(theta[2]), math.sin(theta[2]), 0],
                    [-math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    # 组合旋转矩阵（按Z-Y-X顺序）
    R = np.dot(R_x, np.dot(R_y, R_z))

    return R


def getExternalMatrix(uavPos, cameraPosForUav, eurTheta):
    '''
    获取外参矩阵，转换为齐次坐标形式并输出
    '''
    # 计算旋转矩阵
    rotMatrix = eul2rot(np.array(eurTheta))
    # 计算平移向量
    t = -np.dot(rotMatrix, (np.array(uavPos) +
                            np.array(cameraPosForUav)).reshape(-1, 1))
    # 创建齐次坐标的最后一行
    f = np.array([0, 0, 0, 1])
    # 组合旋转和平移
    rot_t = np.hstack((rotMatrix, t))
    # 组合成完整的齐次变换矩阵
    ext_matrix = np.vstack((rot_t, f))
    return ext_matrix


def getUav9Point(uavPos, copterCenterHeight, uavAng, UAVh=0.185, UAVw=0.185, eurTheta=[0, 0, 0]):
    '''
    通过输入UAV的坐标以及相关信息，获取九个空间点的坐标
    '''
    # 计算无人机中心点
    center = np.array(uavPos)

    # 返回无人机9个关键点的3D坐标（中心点+8个边界点）
    return [
        # 底风轮
        center + np.transpose(np.dot(np.linalg.inv(eul2rot(uavAng)),
                                     np.transpose(np.array([-0.682, 1.598, -4.641])))),
        # 右风轮
        center + np.transpose(np.dot(np.linalg.inv(eul2rot(uavAng)),
                                     np.transpose(np.array([-0.636, 4.875, -15.869])))),
        # 左风轮
        center + np.transpose(np.dot(np.linalg.inv(eul2rot(uavAng)),
                                     np.transpose(np.array([-0.604, -6.486, -13.068])))),
        # 中心
        center + np.transpose(np.dot(np.linalg.inv(eul2rot(uavAng)),
                                     np.transpose(np.array([-0.931, 0, -11.2])))),
        # 中心后
        center + np.transpose(np.dot(np.linalg.inv(eul2rot(uavAng)),
                                     np.transpose(np.array([1.168, 0, -11.2])))),
        # 中心前
        center + np.transpose(np.dot(np.linalg.inv(eul2rot(uavAng)),
                                     np.transpose(np.array([-1.05, -1.42, -10.919])))),
        # 原点
        center + np.transpose(np.dot(np.linalg.inv(eul2rot(uavAng)),
                                     np.transpose(np.array([0, 0, 0])))),

    ]


def get_image_size(knownWidth, focalLength, distance):
    '''
    根据已知宽度、焦距和距离计算图像中的尺寸
    '''
    return (knownWidth * focalLength) / distance


def get_pic_situation(relative_pos, px, py, focalLength):
    '''
    计算点在图像中的像素坐标
    '''
    # 计算x方向的等效焦距
    fx = (focalLength / relative_pos[0]) * \
         np.linalg.norm([relative_pos[0], relative_pos[2]])
    # 计算y方向的等效焦距
    fy = (focalLength / relative_pos[0]) * \
         np.linalg.norm([relative_pos[0], relative_pos[1]])
    # 计算x坐标
    cx = get_image_size(relative_pos[1], fx, np.linalg.norm(
        [relative_pos[0], relative_pos[2]])) + px
    # 计算y坐标
    cy = get_image_size(relative_pos[2], fy, np.linalg.norm(
        [relative_pos[0], relative_pos[1]])) + py

    return [round(cx), round(cy)]  # 返回整数像素坐标


def relative_position(sit_a, sit_b):
    '''
    计算两个位置之间的相对位置
    '''
    a = [sit_a[0] - sit_b[0], sit_a[1] - sit_b[1], sit_a[2] - sit_b[2]]
    return a


def get_distance(relative_pos):
    '''
    计算相对位置的距离
    '''
    distance = np.linalg.norm(relative_pos)
    return distance


def xyxy2xywh(x1, x2):
    '''
    从(x1,y1,x2,y2)格式转换为(center_x, center_y, width, height)格式
    '''
    cent = [(x1[0] + x2[0]) / 2, (x1[1] + x2[1]) / 2]  # 计算中心点
    wh = [x2[0] - x1[0], x2[1] - x1[1]]  # 计算宽度和高度
    return cent, wh


def xywh2yolo(c, wh):
    '''
    转换为YOLO格式：归一化的(center_x, center_y, width, height)
    '''
    y1 = round(c[0] / pic_w, 8)  # 归一化中心x坐标
    y2 = round(c[1] / pic_h, 8)  # 归一化中心y坐标
    y3 = round(wh[0] / pic_w, 8)  # 归一化宽度
    y4 = round(wh[1] / pic_h, 8)  # 归一化高度
    return y1, y2, y3, y4


def text_create(name, msg):
    '''
    创建文本文件并写入内容
    '''
    desktop_path = labels  # 文件存放路径
    full_path = desktop_path + name + '.txt'  # 完整文件路径
    file = open(full_path, 'w')  # 打开文件
    file.write(msg)  # 写入内容
    file.close()  # 关闭文件


# 以当前日期和时间创建文件夹，准备写入新图片
path_prefix = sys.path[0]  # 当前工作路径
path_dir = os.path.join(
    path_prefix, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))  # 创建带时间戳的目录名
os.makedirs(path_dir)  # 创建目录
print("path_dir: {}".format(path_dir))
path_img = os.path.join(path_dir, "images")  # 图像子目录
labels = os.path.join(path_dir, "labels")  # 标签子目录
os.makedirs(path_img)  # 创建图像目录
os.makedirs(labels)  # 创建标签目录

print(path_img)  # 打印图像目录路径

# 初始化计时和计数变量
startTime = time.time()  # 程序开始时间
lastTime = time.time()  # 上一次循环时间
timeInterval = 0.1  # 时间间隔（0.1秒，10Hz）
cnt = 0  # 图像计数器
num = 0  # 循环计数器
shape_num = 7  # 目标自凸点个数以及自身原点
pic_sit = [[0 for col in range(2)] for row in range(shape_num)]  # 存储图像坐标的数组

# 计算相机位置
camere_pos = PosInit + \
             np.transpose(np.dot(np.linalg.inv(
                 eul2rot(angCopterE)), np.transpose(cameraPosForUav)))  # 观测飞机的相机位姿是固定的

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

    num = num + 1  # 增加循环计数器

    # 每5个循环（0.5秒）执行一次
    if num % 5 == 0:
        # 随机生成目标位置和角度
        TargePos = [InitTargePos[0] + random.randint(-60, -30) , InitTargePos[1] +
                    random.randint(-6, 6) , InitTargePos[2] + random.randint(0, 100) / 200.0]
        TargeAng = [InitTargeAng[0] + random.randint(-50, 50) / 50.0 * 20 / 180.0 * math.pi,
                    InitTargeAng[1] + random.randint(
                        -50, 50) / 50.0 * 20 / 180.0 * math.pi,
                    InitTargeAng[2] + random.randint(-50, 50) / 50.0 * 30 / 180.0 * math.pi]

        # 发送位置命令更新目标
        ue.sendUE4Pos(100, 113, 0, TargePos, TargeAng, -1)
        # 计算目标的9个关键点
        points = getUav9Point(TargePos, copterCenterHeight, TargeAng)
        time.sleep(0.2)  # 等待0.2秒确保目标更新
        i = 0  # 点计数器

        # 如果有图像数据，显示图像
        if vis.hasData[0]:
            img = vis.Img[0]  # 获取图像
            cv2.imshow("pic1", img)  # 显示图像
            cv2.waitKey(1)  # 等待1ms保持窗口响应

        # 遍历所有点，计算它们在图像中的位置
        for t_point in points:
            # 计算相对位置并旋转到相机坐标系
            relative_pos = np.dot(eul2rot(angCopterE), np.transpose(
                relative_position(t_point, camere_pos)))
            # 计算点在图像中的像素坐标
            pos_p = get_pic_situation(
                np.transpose(relative_pos), px, py, focal)
            pic_sit[i] = pos_p  # 存储坐标
            i = i + 1  # 增加点计数器
            print(pos_p)  # 打印坐标

        pic_sit = np.array(pic_sit)  # 转换为numpy数组

        # 计算边界框的左上和右下点
        x1y1 = [min(pic_sit[:, 0]), min(pic_sit[:, 1])]  # 左上点
        x2y2 = [max(pic_sit[:, 0]), max(pic_sit[:, 1])]  # 右下点

        # 检查目标是否在图像内
        if (x1y1[0] + x2y2[0]) / 2 > 0 and (x1y1[1] + x2y2[1]) / 2 > 0 and (x1y1[0] + x2y2[0]) / 2 < pic_w and (
                x1y1[1] + x2y2[1]) / 2 < pic_h:
            target = 1  # 目标在图像内
        else:
            target = 0  # 目标不在图像内

        # 调整边界框确保在图像范围内
        if target == 1 and x1y1[0] < 0:
            x1y1[0] = 0
        if target == 1 and x1y1[1] < 0:
            x1y1[1] = 0
        if target == 1 and x2y2[0] > pic_w:
            x2y2[0] = pic_w
        if target == 1 and x2y2[1] > pic_h:
            x2y2[1] = pic_h

        # 如果有图像数据且目标在图像内
        if vis.hasData[0] and target == 1:
            img1 = vis.Img[0]  # 获取图像
            # 创建带边界框的可视化图像
            img_show = vis.Img[0].copy()
            cv2.rectangle(img_show, (int(x1y1[0]), int(x1y1[1])),
                          (int(x2y2[0]), int(x2y2[1])), (0, 0, 255))  # 绘制红色边界框
            cv2.imshow("pic1", img_show)  # 显示带边界框的图像

            # 保存图像和标注
            cv2.imwrite(path_img + '/' + "{}.jpg".format(cnt), img1)  # 保存图像
            cent, wh = xyxy2xywh(x1y1, x2y2)  # 计算中心点和宽高
            y1, y2, y3, y4 = xywh2yolo(cent, wh)  # 转换为YOLO格式
            # 创建标注文件
            full_path = labels + '/' + '{}.txt'.format(cnt)
            file = open(full_path, 'w')
            # 写入YOLO格式标注（类别0，中心x，中心y，宽度，高度）
            file.writelines(['0', ' ', str(y1), ' ', str(
                y2), ' ', str(y3), ' ', str(y4)])
            file.close()

        cnt += 1  # 增加图像计数器

    # 按q键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break