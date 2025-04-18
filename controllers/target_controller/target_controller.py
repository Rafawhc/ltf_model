from math import cos, sin
from controller import Emitter
from controller import Supervisor
import struct

# create the Robot instance.
robot = Supervisor()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# get target node
target_node = robot.getFromDef('target')
translation_field = target_node.getField('translation')

# initialize sensor
imu = robot.getDevice('imu')
imu.enable(timestep)
emitter = robot.getDevice('emitter')
emitter.setChannel(1)

past_x_global = target_node.getPosition()[0]
past_y_global = target_node.getPosition()[1]
past_z_global = target_node.getPosition()[2]
dt = timestep / 1000


def set_target_velocity(desired_vx, desired_vy, desired_vz, past_x_global, past_y_global, past_z_global):
    """通过设置目标速度来解算出当前时刻目标的位置
    """
    desired_x = desired_vx * dt + past_x_global
    desired_y = desired_vy * dt + past_y_global
    desired_z = desired_vz * dt + past_z_global

    return desired_x, desired_y, desired_z


print('开始执行1...')
robot.step(50000)

start_time = robot.getTime()
desired_vx, desired_vy, desired_vz = -0.2, 0, 0

while robot.step(timestep) != -1:

    # get measurements and body fixed velocities
    x_global, y_global, z_global = target_node.getPosition()
    vx_global = (x_global - past_x_global) / dt
    vy_global = (y_global - past_y_global) / dt
    vz_global = (z_global - past_z_global) / dt
    actualYaw = imu.getRollPitchYaw()[2]
    cosyaw, sinyaw = cos(actualYaw), sin(actualYaw)
    vy = float(-vx_global * sinyaw + vy_global * cosyaw)
    vx = float(vx_global * cosyaw + vy_global * sinyaw)
    vz = float(z_global - past_z_global) / dt

    # set velocity
    # if robot.getTime() - start_time > 1:
    #     desired_vy += -0.1
    #     desired_vz += 0.1
    #     start_time = robot.getTime()

    # send message
    # Prepare floating point numbers to send
    data_to_send = [vx, vy, vz]  # Example floating-point numbers
    # Pack the floating point numbers into bytes
    message = struct.pack('fff', *data_to_send)
    # Send the message
    emitter.send(message)

    # update target location
    pos_x, pos_y, pos_z = set_target_velocity(desired_vx, desired_vy, desired_vz, x_global, y_global, z_global)
    new_position = [pos_x, pos_y, pos_z]
    translation_field.setSFVec3f(new_position)

    # show label
    robot.setLabel(5, 'global position of target: (' + '{:0.2f} '.format(x_global) + '{:0.2f} '.format(
        y_global) +
                   '{:0.2f})'.format(z_global), 0.5, 0.15, 0.067, 0xFF0000, 0, "Arial")
    robot.setLabel(6, 'target_vx: ' + '{:0.2f}'.format(vx), 0.75, 0.18,
                   0.07, 0x000080, 0, "Arial")
    robot.setLabel(7, 'target_vy: ' + '{:0.2f}'.format(vy), 0.75, 0.21,
                   0.07, 0x000080, 0, "Arial")
    robot.setLabel(8, 'target_vz: ' + '{:0.2f}'.format(vz), 0.75, 0.24,
                   0.07, 0x000080, 0, "Arial")  # 0x23d96e

    # update global variable
    past_x_global = x_global
    past_y_global = y_global
    past_z_global = z_global





































