"""
1. kd_z=30 可以乘0.1*dt 或者0.2*dt

2. 查看镜头转动的角度:
    print(robot.getFromDef('roll_jointParameters').getField('position').getSFFloat())
    print(robot.getFromDef('pitch_jointParameters').getField('position').getSFFloat())

3. robot.getTime()  # This function returns the current simulation time in seconds.

4. robot.step(100)  暂停函数(ms)

5. 获取UVA_yaw的速率： actual_state.yaw_rate = gyro.getValues()[2]

6.     camera_pitch_motor = robot.getDevice('camera pitch')
    camera_roll_motor = robot.getDevice('camera roll')
    camera_roll_motor.setPosition(-(imu.getRollPitchYaw()[0] + gyro.getValues()[0] * dt))
    camera_pitch_motor.setPosition(-(imu.getRollPitchYaw()[1] + gyro.getValues()[1] * dt))

7. 使能键盘
    key_bord = robot.getKeyboard()
    key_bord.enable(timestep)

8. 输出接收的数据   print(f"Received data: {data_received}")

"""
import json
import os
import struct
import time
from collections import deque

import cupy as cp
from math import sin
from math import cos

from matplotlib import pyplot as plt

from controller import Supervisor
from pid_controller import *
from utility import webots_save_image, get_position
from utility import webots_get_image
from DemoModel import NewModel
from NewModel import Model
from matplotlib import font_manager

# create robot instance
robot = Supervisor()

# the timestep of 3D world
timestep = int(robot.getBasicTimeStep())

# Initialize motors
motors = []
for i in range(1, 5):
    motors.append(robot.getDevice('m{0}_motor'.format(i)))
    motors[-1].setPosition(float('inf'))
    if i % 2:
        motors[-1].setVelocity(-1.0)
    else:
        motors[-1].setVelocity(1.0)

# initialize camera node
camera_pitch_node = robot.getFromDef('CAMERA_PITCH_MOTOR')
camera_roll_node = robot.getFromDef('CAMERA_ROLL_MOTOR')

# initialize sensor
imu = robot.getDevice('inertial_unit')
imu.enable(timestep)
gps = robot.getDevice('gps')
gps.enable(timestep)
gyro = robot.getDevice('gyro')
gyro.enable(timestep)
camera = robot.getDevice('camera')
camera.enable(timestep)
receiver = robot.getDevice('receiver')
receiver.enable(timestep)
receiver.setChannel(1)


# get time differential
dt = timestep / 1000

# get target node
target_node = robot.getFromDef('target')

# true value for PID control
actual_state = drone_state()

# initialize variables
past_drone_x = 0.
past_drone_y = 0.
past_drone_z = 0.
past_target_x = 0.
past_target_y = 0.
past_target_z = 0.

# get image width and height
img_width = camera.getWidth()
img_height = camera.getHeight()

# center of the image
i0, j0 = img_height // 2, img_width // 2

# desired value
desired_altitude = 1

# receiver data
data_received = None

# Initialize PID gains.
gains_pid = gains_pid_t(1, 0.5, 0.5, 0.1, 2, 0.5, 10, 5, 30)

# initialize struct for motor power
motor_power = motor_power()

# model
model = Model()

# circular list
target_i_list = deque(maxlen=5)
target_j_list = deque(maxlen=5)
ave_target_i_list = deque(maxlen=5)
ave_target_j_list = deque(maxlen=5)

# 初始化 Matplotlib 实时图形
plt.ion()  # 开启交互模式

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决坐标轴负号显示问题

# 创建一个包含三个子图的画布 (3行1列的布局)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 14))  # 3行1列布局

# 调整子图之间的间距，hspace 控制垂直间距
plt.subplots_adjust(hspace=0.5)

# 初始化数据
time_data = []  # 时间数据
x1_data = []
y1_data = []
x2_data = []
y2_data = []
y2_single_data = []
y2_const_data = []
y3_data_1 = []
y3_data_2 = []
y3_data_3 = []

# 初始化图形对象
line1_1, = ax1.plot([], [], 'r-', label='center_point')  # 子图1: 第一个二维坐标系
line1_2, = ax1.plot([], [], 'b-', label='model')  # 子图1: 第二个二维坐标系
line2_1, = ax2.plot([], [], 'g-', label='drone')  # 子图2: 单独数据沿时间展示
line2_2, = ax2.plot([], [], 'b-', label='target')  # 子图2: 单独数据沿时间展示
# line3_1, = ax3.plot([], [], 'm-', label='PID')  # 子图3: 第一个数据
line3_2, = ax3.plot([], [], 'c-', label='drone')  # 子图3: 第二个数据
line3_3, = ax3.plot([], [], 'g-', label='target')  # 子图3: 第二个数据
# 设置子图的标题和标签
ax1.set_title('Subplot 1: 像素值')
ax2.set_title('Subplot 2: Y轴位置轨迹')
ax3.set_title('Subplot 3: Y轴速度轨迹')

ax1.set_xlabel('Time (s)')
ax1.set_ylabel('pixel value')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Y-axis Position (m)')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Y-axis Velocity (m/s)')

# 开启图例
ax1.legend()
ax2.legend()
ax3.legend()

# 开启交互模式
plt.ion()
plt.show()

# 初始化仿真时间
time = 0

# receive model data
my_list = [None for i in range(8)]

num = 0.
desired_vy = 0.

vy_numbers = []
vz_numbers = []
data = []



"""altitude hold"""
while robot.step(timestep) != -1:

    # altitude control
    actual_state.altitude = gps.getValues()[2]
    desired_altitude = UAV_height_controller(actual_state, gains_pid, dt, motor_power, desired_altitude, img_height, 0)
    setting_motor_speed(motors, motor_power)

    if robot.getTime() >= 50:
        break

"""update circular list data"""
a, b = 0, 0
while robot.step(timestep) != -1:

    # recever data
    if receiver.getQueueLength() > 0:
        # Get the received message
        message = receiver.getBytes()

        # Unpack the floating point numbers from the message
        data_received = struct.unpack('fff', message)

        # Clear the receiver queue for the next message
        receiver.nextPacket()

    # altitude control
    actual_state.altitude = gps.getValues()[2]
    desired_altitude = UAV_height_controller(actual_state, gains_pid, dt, motor_power, desired_altitude, img_height, 0)
    setting_motor_speed(motors, motor_power)

    grayImg = webots_get_image(camera, img_height, img_width)
    grayImg_cupy = cp.asarray(grayImg)
    model.run1(grayImg_cupy, 0, 0, my_list)
    target_i_pixel, target_j_pixel = my_list[0][0], my_list[0][1]

    if [target_i_pixel, target_j_pixel] != [-1, -1]:

        target_j_list.append(target_j_pixel)
        target_i_list.append(target_i_pixel)
        if a <= len(target_j_list):
            a += 1
            continue

        ave_target_i_list.append(sum(target_i_list) / len(target_i_list))
        ave_target_j_list.append(sum(target_j_list) / len(target_j_list))
        if b <= len(ave_target_j_list):
            b += 1
            continue
        else:
            break

print('追踪开始...')
while robot.step(timestep) != -1:

    '''compute control variable'''
    # get measurements
    target_x, target_y, target_z = target_node.getPosition()
    drone_x, drone_y, drone_z = gps.getValues()
    drone_roll, drone_pitch, drone_yaw = imu.getRollPitchYaw()
    gyro_x, gyro_y, gyro_z = gyro.getValues()

    # get velocities under body-fixed coordinate system
    drone_vx = (drone_x - past_drone_x) / dt
    drone_vy = (drone_y - past_drone_y) / dt
    drone_vz = (drone_z - past_drone_z) / dt
    cos_yaw, sin_yaw = cos(drone_yaw), sin(drone_yaw)
    actual_state.vy = float(-drone_vx * sin_yaw + drone_vy * cos_yaw)
    actual_state.vx = float(drone_vx * cos_yaw + drone_vy * sin_yaw)

    actual_state.roll = drone_roll
    actual_state.pitch = drone_pitch
    actual_state.altitude = drone_z

    num += 1
    if num == 1:
        actual_state.vy = 0
        drone_vz = 0

    """save image"""
    # if 0 <= num <= 3000:
    #     vy_numbers.append(actual_state.vy)
    #     vz_numbers.append(actual_state.vz)
    #     a = -actual_state.vy if actual_state.vy < 0 else 0  # 背景向左运动
    #     b = actual_state.vy if actual_state.vy > 0 else 0  # 背景向右运动
    #     c = -actual_state.vz if actual_state.vz < 0 else 0  # 背景向上运动
    #     d = actual_state.vz if actual_state.vz > 0 else 0  # 背景向下运动
    #     e = gyro_x
    #     f = gyro_y
    #     g = gyro_z
    #
    #     data.append([a, b, c, d, e, f, g])
    #     # webots_save_image(camera, './data/videos/original_rgb_image')
    # else:
    #     uav_path = './uav_val.json'
    #     if os.path.exists(uav_path):  # 检查文件是否存在
    #         raise FileExistsError(f"Error: The file '{uav_path}' already exists.")
    #     with open(uav_path, 'w') as f:
    #         json.dump(data, f)
    #     print(f"Data saved to {uav_path}")

    ''''model process'''
    grayImg = webots_get_image(camera, img_height, img_width)
    grayImg_cupy = cp.asarray(grayImg)
    model.run1(grayImg_cupy, actual_state.vy, drone_vz, my_list)
    target_i_pixel, target_j_pixel = my_list[0][0], my_list[0][1]

    '''motion control'''
    if [target_i_pixel, target_j_pixel] != [-1, -1] and np.abs(target_i_pixel - target_i_list[-1]) < 10 and np.abs(target_j_pixel - target_j_list[-1]) < 10:
        # update y-axis data
        target_i_list.append(target_i_pixel)
        ave_target_i_list.append(sum(target_i_list) / len(target_i_list))

        # update x-axis data
        target_j_list.append(target_j_pixel)
        ave_target_j_list.append(sum(target_j_list) / len(target_j_list))

    # altitude control
    vertical_pixel_error = i0 - ave_target_i_list[-1]
    desired_altitude = UAV_height_controller(actual_state, gains_pid, dt, motor_power, desired_altitude, img_height,
                                             vertical_pixel_error)

    # roll control
    horizontal_pixel_error = j0 - ave_target_j_list[-1]
    desired_vy = UAV_roll_controller(actual_state, gains_pid, dt, motor_power, horizontal_pixel_error, img_width)
    print('pos', ave_target_i_list[-1], ave_target_j_list[-1])
    # motor control
    setting_motor_speed(motors, motor_power)

    # camera gimbal control
    camera_roll_node.setJointPosition(-(imu.getRollPitchYaw()[0] + gyro.getValues()[0] * dt), 1)
    camera_pitch_node.setJointPosition(-(imu.getRollPitchYaw()[1] + gyro.getValues()[1] * dt), 1)

    '''receiver data'''
    if receiver.getQueueLength() > 0:
        # Get the received message
        message = receiver.getBytes()

        # Unpack the floating point numbers from the message
        data_received = struct.unpack('fff', message)

        # Clear the receiver queue for the next message
        receiver.nextPacket()

    '''show label'''
    # VUA的速度在3D场景坐标系下解算
    robot.setLabel(1, 'drone_vx: ' + '{:0.2f}'.format(actual_state.vx), 0.75, 0.03, 0.07, 0x000080, 0, "Arial")
    robot.setLabel(2, 'drone_vy: ' + '{:0.2f}'.format(actual_state.vy), 0.75, 0.06, 0.07, 0x000080, 0, "Arial")
    robot.setLabel(3, 'drone_vz: ' + '{:0.2f}'.format(drone_vz), 0.75, 0.09, 0.07, 0x000080, 0, "Arial")
    robot.setLabel(4, 'global position of   UVA: (' + '{:0.2f} '.format(drone_x) + '{:0.2f} '.format(drone_y) +
                   '{:0.2f})'.format(drone_z), 0.5, 0.12, 0.067, 0xFF0000, 0, "Arial")

    '''update figure'''
    time_data.append(time)

    y1_data.append(i0)
    y2_data.append(ave_target_j_list[-1])

    y2_single_data.append(drone_x)
    y2_const_data.append(target_x)

    # y3_data_1.append(desired_vy)
    y3_data_2.append(actual_state.vy)
    y3_data_3.append(data_received[1])

    # 更新子图1: (x1, y1) 和 (x2, y2)
    line1_1.set_xdata(time_data)
    line1_1.set_ydata(y1_data)
    line1_2.set_xdata(time_data)
    line1_2.set_ydata(y2_data)
    ax1.relim()
    ax1.autoscale_view()

    # 更新子图2: 单独数据沿时间展示
    line2_1.set_xdata(time_data)
    line2_1.set_ydata(y2_single_data)
    line2_2.set_xdata(time_data)
    line2_2.set_ydata(y2_const_data)
    ax2.relim()
    ax2.autoscale_view()

    # 更新子图3: 两个数据沿时间展示
    #line3_1.set_xdata(time_data)
    #line3_1.set_ydata(y3_data_1)
    line3_2.set_xdata(time_data)
    line3_2.set_ydata(y3_data_2)
    line3_3.set_xdata(time_data)
    line3_3.set_ydata(y3_data_3)
    ax3.relim()
    ax3.autoscale_view()

    # 更新画布
    fig.canvas.draw()
    fig.canvas.flush_events()

    # 增加时间
    time += dt  # 假设时间步长是毫秒

    '''update global variable'''
    past_drone_x = drone_x
    past_drone_y = drone_y
    past_drone_z = drone_z
    past_target_x = target_x
    past_target_y = target_y
    past_target_z = target_z

print('追踪结束...')
