import numpy as np


class motor_power:
    __slots__ = ('_m1', '_m2', '_m3', '_m4')

    def __init__(self, m1=0., m2=0., m3=0., m4=0.):
        self._m1 = m1
        self._m2 = m2
        self._m3 = m3
        self._m4 = m4

    @property
    def m1(self):
        return self._m1

    @m1.setter
    def m1(self, value):
        self._m1 = value

    @property
    def m2(self):
        return self._m2

    @m2.setter
    def m2(self, value):
        self._m2 = value

    @property
    def m3(self):
        return self._m3

    @m3.setter
    def m3(self, value):
        self._m3 = value

    @property
    def m4(self):
        return self._m4

    @m4.setter
    def m4(self, value):
        self._m4 = value


class drone_state:

    def __init__(self, roll=0., pitch=0., yaw_rate=0., vx=0., vy=0., vz=0.):
        self._roll = roll
        self._pitch = pitch
        self._yaw_rate = yaw_rate
        self._vx = vx
        self._vy = vy
        self._vz = vz

    # 访问器 - getter方法
    @property
    def roll(self):
        return self._roll

    # 修改器 - setter方法
    @roll.setter
    def roll(self, value):
        self._roll = value

    @property
    def pitch(self):
        return self._pitch

    @pitch.setter
    def pitch(self, value):
        self._pitch = value

    @property
    def yaw_rate(self):
        return self._yaw_rate

    @yaw_rate.setter
    def yaw_rate(self, value):
        self._yaw_rate = value

    @property
    def vx(self):
        return self._vx

    @vx.setter
    def vx(self, value):
        self._vx = value

    @property
    def vy(self):
        return self._vy

    @vy.setter
    def vy(self, value):
        self._vy = value

    @property
    def vz(self):
        return self._vz

    @vz.setter
    def vz(self, value):
        self._vz = value


class gains_pid_t:

    def __init__(self, kp_att_y, kd_att_y, kp_att_rp, kd_att_rp, kp_vel_xy, kd_vel_xy, kp_z,
                 ki_z, kd_z):
        self._kp_att_y = kp_att_y
        self._kd_att_y = kd_att_y
        self._kp_att_rp = kp_att_rp
        self._kd_att_rp = kd_att_rp
        self._kp_vel_xy = kp_vel_xy
        self._kd_vel_xy = kd_vel_xy
        self._kp_z = kp_z
        self._ki_z = ki_z
        self._kd_z = kd_z

    @property
    def kd_vel_xy(self):
        return self._kd_vel_xy

    @kd_vel_xy.setter
    def kd_vel_xy(self, value):
        self._kd_vel_xy = value

    @property
    def kp_vel_xy(self):
        return self._kp_vel_xy

    @kp_vel_xy.setter
    def kp_vel_xy(self, value):
        self._kp_vel_xy = value

    @property
    def kd_att_rp(self):
        return self._kd_att_rp

    @kd_att_rp.setter
    def kd_att_rp(self, value):
        self._kd_att_rp = value

    @property
    def kp_att_rp(self):
        return self._kp_att_rp

    @kp_att_rp.setter
    def kp_att_rp(self, value):
        self._kp_att_rp = value

    @property
    def kd_att_y(self):
        return self._kd_att_y

    @kd_att_y.setter
    def kd_att_y(self, value):
        self._kd_att_y = value

    @property
    def kp_att_y(self):
        return self._kp_att_y

    @kp_att_y.setter
    def kp_att_y(self, value):
        self._kp_att_y = value

    @property
    def kp_z(self):
        return self._kp_z

    @kp_z.setter
    def kp_z(self, value):
        self._kp_z = value

    @property
    def kd_z(self):
        return self._kd_z

    @kd_z.setter
    def kd_z(self, value):
        self._kd_z = value

    @property
    def ki_z(self):
        return self._ki_z

    @ki_z.setter
    def ki_z(self, value):
        self._ki_z = value


# global variable
pastAltitudeError = 0.
pastPitchError = 0.
pastRollError = 0.
pastVxError = 0.
pastVyError = 0.
altitudeIntegral = 0.
altitudeErrorIntegral = 0.
horizontalIntegral = 0.


def UAV_height_controller(actual_state, gains_pid, dt, motor_powers, desired_altitude, image_height,
                          vertical_pixel_error=0):
    """垂直像素误差(PD控制)->高度误差(PID控制)-> 电机控制
    """

    desired_height = pid_altitude_position_controller(vertical_pixel_error, desired_altitude, dt, image_height)
    altitude_val = pid_altitude_attitude_controller(actual_state, desired_height, gains_pid, dt)
    mixing_height(motor_powers, altitude_val)

    # print('altitude_val', altitude_val)
    return desired_height


def pid_altitude_position_controller(vertical_pixel_error, desired_altitude, dt, image_height=450):
    """垂直像素误差(PD控制), 生成新地期望高度值
    """

    global altitudeErrorIntegral

    altitude_error = calculate_altitude_acceleration(vertical_pixel_error, image_height) * dt

    altitudeErrorIntegral += altitude_error * dt
    desired_altitude += altitude_error + 1.15 * altitudeErrorIntegral  # 3

    return desired_altitude


def UAV_pitch_controller(actual_state, gains_pid, dt, motor_powers, desired_vx):
    """x轴位置误差->速度误差控制-> 姿态误差控制-> 电机控制
    """

    pitch_command = pid_pitch_velocity_controller(actual_state, desired_vx, gains_pid, dt)
    pitch_val = pid_pitch_attitude_controller(actual_state, pitch_command, gains_pid, dt)
    mixing_pitch(motor_powers, pitch_val)


def UAV_roll_controller(actual_state, gains_pid, dt, motor_powers, horizontal_error, image_width, desired_vy=None):
    """y轴位置误差->速度误差控制-> 姿态误差控制-> 电机控制
    """
    # if desired_vy is None:
    #     desired_vy = pid_horizontal_position_controller(horizontal_error, dt, image_width)
    desired_vy = pid_horizontal_position_controller(horizontal_error, dt, image_width)
    roll_command = pid_roll_velocity_controller(actual_state, desired_vy, gains_pid, dt)

    roll_val = pid_roll_attitude_controller(actual_state, roll_command, gains_pid, dt)
    mixing_roll(motor_powers, roll_val)

    return desired_vy, roll_val


def pid_horizontal_position_controller(horizontal_pixel_error, dt, image_width):
    """水平像素误差(PI控制)得到期望的水平速度控制
    """
    global horizontalIntegral

    horizontal_error = calculate_horizontal_acceleration(horizontal_pixel_error, image_width)

    horizontalIntegral += horizontal_error
    desired_v = horizontal_error + horizontalIntegral * dt

    return desired_v


def calculate_horizontal_acceleration(pixel_error, image_width=450):
    """把水平方向的像素差值转化到0-1之间
    """

    min_value, max_value = 0, image_width / 2
    new_min, new_max = 0, 1

    if pixel_error < -1:
        return -((abs(pixel_error) - min_value) / (max_value - min_value) * (new_max - new_min) + new_min)
    elif pixel_error > 1:
        return (abs(pixel_error) - min_value) / (max_value - min_value) * (new_max - new_min) + new_min
    else:
        return 0


def calculate_altitude_acceleration(pixel_error, image_height=450):
    """由垂直像素误差计算期望的高度增加/减少量
    """

    min_value, max_value = 0, image_height / 2
    new_min, new_max = 0, 1  # 0.25

    if pixel_error < -1:
        return -((abs(pixel_error) - min_value) / (max_value - min_value) * (new_max - new_min) + new_min)
    elif pixel_error > 1:
        return (abs(pixel_error) - min_value) / (max_value - min_value) * (new_max - new_min) + new_min
    else:
        return 0


def UAV_yaw_controller(gains_pid, motor_powers, yaw_error):
    """yaw的姿态控制器

        使用P控制器
    """

    # PID control
    yaw_val = gains_pid.kp_att_y * np.clip(yaw_error, -1, 1)

    mixing_yaw(motor_powers, yaw_val)


def pid_altitude_attitude_controller(actual_state, desired_altitude, gains_pid, dt):
    """高度的PID控制器
        """

    global pastAltitudeError
    global altitudeIntegral

    # calculate errors
    altitude_error = desired_altitude - actual_state.altitude

    # calculate the differential of the error
    altitudeDifferential = (altitude_error - pastAltitudeError) / dt

    # calculate integral item
    altitudeIntegral += altitude_error * dt

    # # PID control
    altitude_val = gains_pid.kp_z * altitude_error + gains_pid.kd_z * \
                   altitudeDifferential + gains_pid.ki_z * np.clip(altitudeIntegral, -2, 2) + 55.5

    # save error for the next round
    pastAltitudeError = altitude_error

    return altitude_val


def pid_pitch_attitude_controller(actual_state, desired_pitch, gains_pid, dt):
    """pitch的姿态控制器

            使用PD控制器
        """

    global pastPitchError

    # calculate errors
    pitch_error = desired_pitch - actual_state.pitch

    # calculate the differential of the error
    pitch_derivative_error = (pitch_error - pastPitchError) / dt

    # PID control
    pitch_val = -gains_pid.kp_att_rp * np.clip(pitch_error, -1, 1) - gains_pid.kd_att_rp * pitch_derivative_error

    # save error for the next round
    pastPitchError = pitch_error

    return pitch_val


def pid_roll_attitude_controller(actual_state, desired_roll, gains_pid, dt):
    """roll的姿态控制器

    使用PD控制器
    """

    global pastRollError

    # calculate errors
    roll_error = desired_roll - actual_state.roll

    # calculate the differential of the error
    roll_derivative_error = (roll_error - pastRollError) / dt

    # PID control
    roll_val = gains_pid.kp_att_rp * np.clip(roll_error, -1, 1) + gains_pid.kd_att_rp * roll_derivative_error

    # save error for the next round
    pastRollError = roll_error

    return roll_val


def pid_pitch_velocity_controller(actual_state, desired_vx, gains_pid, dt):
    """pitch的速度的PD控制器
    """

    global pastVxError

    # calculate error in x and y velocity directions
    vx_error = desired_vx - actual_state.vx

    # the differential error in the x and y velocity directions
    vx_derivative = (vx_error - pastVxError) / dt

    # PID control
    pitch_command = gains_pid.kp_vel_xy * np.clip(vx_error, -1, 1) + gains_pid.kd_vel_xy * vx_derivative

    # save error for the next round
    pastVxError = vx_error

    return pitch_command


def pid_roll_velocity_controller(actual_state, desired_vy, gains_pid, dt):
    """roll的速度的PD控制器
    """

    global pastVyError

    # calculate error in x and y velocity directions
    vy_error = desired_vy - actual_state.vy

    # the differential error in the x and y velocity directions
    vy_derivative = (vy_error - pastVyError) / dt

    # PID control
    roll_command = -gains_pid.kp_vel_xy * np.clip(vy_error, -1, 1) - gains_pid.kd_vel_xy * vy_derivative

    # save error for the next round
    pastVyError = vy_error

    return roll_command


def mixing_height(motor_power, value):
    motor_power.m1 = value
    motor_power.m2 = value
    motor_power.m3 = value
    motor_power.m4 = value


def mixing_roll(motor_power, value):
    motor_power.m1 -= value
    motor_power.m2 -= value
    motor_power.m3 += value
    motor_power.m4 += value


def mixing_pitch(motor_power, value):
    motor_power.m1 += value
    motor_power.m2 -= value
    motor_power.m3 -= value
    motor_power.m4 += value


def mixing_yaw(motor_power, value):
    motor_power.m1 += value
    motor_power.m2 -= value
    motor_power.m3 += value
    motor_power.m4 -= value


# setting motor speed
def setting_motor_speed(motors, motor_power):
    motors[0].setVelocity(-np.clip(motor_power.m1, 0, 600))
    motors[1].setVelocity(np.clip(motor_power.m2, 0, 600))
    motors[2].setVelocity(-np.clip(motor_power.m3, 0, 600))
    motors[3].setVelocity(np.clip(motor_power.m4, 0, 600))
