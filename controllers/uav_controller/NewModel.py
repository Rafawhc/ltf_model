"""
四个方向的模型，并不是六个

# 计算无人机速度微分
        delta_vz = drone_vz - self.drone_vz
        delta_vy = drone_vy - self.drone_vy


        if drone_vy < 0: # 背景向左运动
            if delta_vy > 0:  # 减速
                self.state_differential[0] = -delta_vy
                self.state_differential[1] = 0
            elif delta_vy < 0:  # 加速
                self.state_differential[0] = -delta_vy
                self.state_differential[1] = 0
        elif drone_vy > 0:  # 背景向右运动
            if delta_vy > 0:  # 加速
                self.state_differential[0] = 0
                self.state_differential[1] = delta_vy
            elif delta_vy < 0:  # 减速
                self.state_differential[0] = 0
                self.state_differential[1] = delta_vy

        if drone_vz > 0:  # 背景向下运动
            if delta_vz > 0:  # 加速
                self.state_differential[2] = 0
                self.state_differential[3] = delta_vz
            elif delta_vz < 0:  # 减速
                self.state_differential[2] = 0
                self.state_differential[3] = delta_vz
        elif drone_vz < 0:  # 背景向上运动
            if delta_vz > 0:  # 减速
                self.state_differential[2] = -delta_vz
                self.state_differential[3] = 0
            elif delta_vz < 0:  # 加速
                self.state_differential[2] = -delta_vz
                self.state_differential[3] = 0


"""

import cupy as cp
import numpy as np
from scipy.special import gamma
from cupyx.scipy.ndimage import gaussian_filter
from dataclasses import dataclass
from typing import Any, List


class Model:
    find_index = {1: 'left', 2: 'right', 3: 'up', 4: 'down'}  # 用于求fit_direction时的方向匹配字典
    direction_list = ('left', 'right', 'up', 'down')

    def __init__(self,
                 image_height=450, image_width=450,
                 gaussian_kernel_sigma=1.0,
                 alpha_lamina=0.8, tau_lamina=4,
                 emd_tau=1, emd_order=10, emd_x0=1,
                 stmd_tau=None, stmd_order=10,
                 eta_on=-1.0, eta_off=-1.0,
                 alpha_feedback=4, wide_feedback=10,
                 alpha_restrain_ON_OFF=4, wide_restrain_ON_OFF=10,
                 lamda_DT=0.6  # 0.0001
                 ):

        if stmd_tau is None:
            stmd_tau = [1, 2, 3, 4, 5]

        # lamina层差分时，如果self.list_lamina有None,则用帧间差分
        self.lastLaminaIpt = cp.zeros((image_height, image_width))
        self.temporalDiff = cp.zeros((image_height, image_width))

        # Retina层高斯核的sigma参数
        self.sigma = gaussian_kernel_sigma

        # Lamina层循环队列与权重核
        self.tau_lamina = tau_lamina  # 差分的时间延迟帧
        self.list_lamina_len = int(tau_lamina * 4)
        self.list_lamina = CircularList(self.list_lamina_len)
        self.kernel_lamina = self.create_lamina_kernel(alpha=alpha_lamina, wide=int(self.tau_lamina * 3))  # 使用4*tau长度平滑

        # tau参数初始化(tau1: EMD延迟tau; tau2: 目标匹配tau)
        self.emd_tau = emd_tau
        self.stmd_tau = stmd_tau

        # EMD方向匹配的参数x0
        self.x0 = emd_x0

        # 光流后的on和off
        self.gamma_kernel_opt = {}
        for tau in self.stmd_tau:
            self.gamma_kernel_opt[tau] = self.create_gamma_kernel(order=stmd_order, tau=tau,
                                                                  wide=int(tau + 3))
        self.l2SignalOff_list = CircularList(
            max(emd_tau + 3, max(self.stmd_tau) + 3))  # 负部用于EMD和ESTMD匹配,所以tau参数取其中最大值

        # feedback抑制参数
        self.eta_on = eta_on
        self.eta_off = eta_off

        # 求fit_tau与fit_direction的截断阈值
        self.lamda_DT = lamda_DT

        # lptc
        self.lptc = cp.zeros(4)

        # Feedback中的时间平滑
        self.C_LO_kernel = self.create_lamina_kernel(alpha=alpha_feedback, wide=wide_feedback)
        self.C_LO_list = {index: CircularList(wide_feedback) for index in range(4)}

        # EMD匹配的gamma核与循环列表数据
        self.gamma_kernel_emd = self.create_gamma_kernel(order=emd_order, tau=emd_tau, wide=int(emd_tau + 3))
        self.l1_list = CircularList(int(emd_tau + 3))
        self.l2_list = CircularList(max(emd_tau + 3, max(self.stmd_tau) + 3))  # 负部用于EMD和ESTMD匹配,所以tau参数取其中最大值

        # 目标匹配的gamma核
        self.gamma_kernel_stmd = {}
        for tau in self.stmd_tau:
            self.gamma_kernel_stmd[tau] = self.create_gamma_kernel(order=stmd_order, tau=tau, wide=int(tau + 3))

        # 目标匹配中ON和OFF抑制后的平滑
        self.restrain_ON = {tau: CircularList(wide_restrain_ON_OFF) for tau in self.stmd_tau}
        self.restrain_OFF = {tau: CircularList(wide_restrain_ON_OFF) for tau in self.stmd_tau}
        self.restrain_kernel = self.create_lamina_kernel(alpha=alpha_restrain_ON_OFF, wide=wide_restrain_ON_OFF)

        # 计算背景光流的卷积核
        self.W_lo2Lptc = self.create_kernel_LO_LPTC()
        self.W_lptc2stmd = self.create_kernel_LPTC_STMD()

        # 目标方向匹配的T4数据与目标匹配的feedback数据
        self.t4_list = {direction: CircularList(int(max(stmd_tau) + 3)) for direction in
                        ('left', 'right', 'up', 'down')}  # 列表长度应与最大tau相一致
        self.feedback_list = CircularList(int(max(stmd_tau) + 3))

        """Kalman Filtering"""
        # 无人机速度权重
        self.uav_vel = cp.zeros(4)

        # 先验估计值
        self.X_prior = cp.zeros(4)

        # 先验误差协方差矩阵P
        self.P_prior = cp.zeros((4, 4))

        # 后验估计值(最优)，四个值分别表示背景向左右上下四个方向运动
        self.motion_state = cp.zeros(4)

        # 后验估计值与真实值的误差协方差矩阵
        self.P_posterior = cp.eye(4) * 0.05

        # 单位阵
        self.I = cp.eye(4)
        self.A = cp.eye(4) * 0.8
        self.B = cp.eye(4) * 0.2

        # 协方差矩阵
        self.Q = cp.eye(4) * 0.1
        self.R = cp.eye(4) * 0.1

    @staticmethod
    def create_lamina_kernel(alpha=3.0, wide=None):
        """这里wide取tau的3倍，也就是向前回溯的长度
        """

        if wide is None: raise Exception("Invalid wide!", wide)

        # Compute the values of the T vector
        timeList = cp.arange(wide)
        TKernel = 1 / cp.exp(timeList / alpha)

        # Normalize the T vector
        TKernel /= cp.sum(TKernel)
        TKernel[TKernel < 1e-4] = 0
        TKernel /= cp.sum(TKernel)

        return TKernel

    @staticmethod
    def create_gamma_kernel(order=100, tau=25, wide=None):
        """
        Generates a discretized Gamma vector.

        Parameters:
        - order: The order of the Gamma function.
        - tau: The time constant of the Gamma function.
        - wide: The length of the vector.

        Returns:
        - gammaKernel: The generated Gamma vector.
        """
        if wide is None: wide = int(cp.ceil(3 * tau))

        # Ensure wide is at least 2
        if wide <= 1: wide = 2

        # Compute the values of the Gamma vector
        timeList = cp.arange(wide)
        gammaKernel = (
                (order * timeList / tau) ** order *
                cp.exp(-order * timeList / tau) /
                (gamma(order) * tau)
        )

        # Normalize the Gamma vector
        gammaKernel /= cp.sum(gammaKernel)
        gammaKernel[gammaKernel < 1e-4] = 0
        gammaKernel /= cp.sum(gammaKernel)

        return gammaKernel

    @staticmethod
    def create_kernel_LPTC_STMD() -> dict:
        """ 生成速度向量核

        Returns:
           返回 W_lptc2stmd 核
        """

        W_lptc2stmd = {}

        # 2倍速度场
        # W_lptc2stmd['left'] = cp.array([0, 0, -1, 1])
        # W_lptc2stmd['right'] = cp.array([0, 0, 1, -1])
        # W_lptc2stmd['up'] = cp.array([-1, 1, 0, 0])
        # W_lptc2stmd['down'] = cp.array([1, -1, 0, 0])

        # 1倍的速度场
        W_lptc2stmd['left'] = np.array([0, 0, -1, 0])
        W_lptc2stmd['right'] = np.array([0, 0, 1, 0])
        W_lptc2stmd['up'] = np.array([-1, 0, 0, 0])
        W_lptc2stmd['down'] = np.array([1, 0, 0, 0])

        return W_lptc2stmd

    @staticmethod
    def create_kernel_LO_LPTC() -> dict:
        """ 生成自运动核

        Returns:
           返回 W_lo2Lptc核
        """

        W_lo2Lptc = {}

        # W_lo2Lptc['left'] = cp.array([-1, 1, 0, 0])
        # W_lo2Lptc['right'] = cp.array([1, -1, 0, 0])
        # W_lo2Lptc['up'] = cp.array([0, 0, -1, 1])
        # W_lo2Lptc['down'] = cp.array([0, 0, 1, -1])

        W_lo2Lptc['left'] = cp.array([0, 1, 0, 0])
        W_lo2Lptc['right'] = cp.array([1, 0, 0, 0])
        W_lo2Lptc['up'] = cp.array([0, 0, 0, 1])
        W_lo2Lptc['down'] = cp.array([0, 0, 1, 0])

        return W_lo2Lptc

    def lamina_process(self, laminaIpt):

        self.list_lamina.record_next(laminaIpt)

        opt_matrix_1 = Model.compute_temporal_conv(self.list_lamina,
                                                   self.kernel_lamina,
                                                   self.list_lamina.pointer)
        opt_matrix_2 = Model.compute_temporal_conv(self.list_lamina,
                                                   self.kernel_lamina,
                                                   (self.list_lamina.pointer - self.tau_lamina) % self.list_lamina_len)

        if opt_matrix_2 is not None:
            temporalDiff = opt_matrix_1 - opt_matrix_2
        else:
            temporalDiff = opt_matrix_1 - self.lastLaminaIpt
            self.lastLaminaIpt = opt_matrix_1

        l1Signal = cp.maximum(temporalDiff, 0)  # ON
        l2Signal = cp.maximum(-temporalDiff, 0)  # OFF

        # 存储正负部，正部L1、负部L2
        self.l1_list.record_next(l1Signal)
        self.l2_list.record_next(l2Signal)

        return l1Signal, l2Signal, opt_matrix_1

    @staticmethod
    def shift_matrix(matrix, x0, direction):
        """把矩阵matrix往direction方向平移x0个像素位置
        """

        if direction == 'up':
            shifted_up = cp.roll(matrix, shift=-x0, axis=0)
            shifted_up[-x0:, :] = 0
            return shifted_up
        elif direction == 'down':
            shifted_down = cp.roll(matrix, shift=x0, axis=0)
            shifted_down[:x0, :] = 0
            return shifted_down
        elif direction == 'left':
            shifted_left = cp.roll(matrix, shift=-x0, axis=1)
            shifted_left[:, -x0:] = 0
            return shifted_left
        elif direction == 'right':
            shifted_right = cp.roll(matrix, shift=x0, axis=1)
            shifted_right[:, :x0] = 0
            return shifted_right
        else:
            raise ValueError("Invalid direction: choose from 'up', 'down', 'left', 'right'")

    def lobula_process(self, tm1Opt: cp.array,
                       tm2Opt: cp.array,
                       mi1Opt: cp.array,
                       tm3Opt: cp.array,
                       x0: int = 1):
        """lobula层的处理, 对应公式(5)~(8)

        Args:
            tm1Opt: TM1
            tm2Opt: TM2
            tm3Opt: TM3
            mi1Opt: Mi1
            x0: 像素偏置值

        Returns:
            做EMD匹配，输出d的四个背景运动方向的顺序为：右左下上
            输出结果为一个list列表类型
        """

        # 用于保存结果
        T4_matrix = []
        T5_matrix = []

        # 图像往右左下上四个方向平移的offset参数值
        for d in ('right', 'left', 'down', 'up'):
            T4_matrix.append(tm1Opt * self.shift_matrix(tm2Opt, x0, d))
            T5_matrix.append(mi1Opt * self.shift_matrix(tm3Opt, x0, d))

        res_T4_matrix = [cp.maximum(T4_matrix[i] - T4_matrix[1 - i], 0) for i in range(4)]
        res_T5_matrix = [cp.maximum(T5_matrix[i] - T5_matrix[1 - i], 0) for i in range(4)]

        # 存储t4
        for index, direction in enumerate(('right', 'left', 'down', 'up')):
            self.t4_list[direction].record_next(res_T4_matrix[index])

        return res_T4_matrix, res_T5_matrix

    def lobula_plate_lptc_process(self, t4Opt, t5Opt, kernelLo2Lptc):
        """实现公式(9)
            返回lptc。背景运动方向：左右上下
        """

        self.lptc.fill(0)

        for index, direction in enumerate(self.direction_list):
            for i in range(4):
                self.lptc[index] += cp.sum((t4Opt[i] + t5Opt[i]) * kernelLo2Lptc[direction][i])

        # print(f"original lptc:  {self.lptc}")

        # 计算 L2 范数
        norm = cp.linalg.norm(self.lptc)

        # 如果范数为零，则无需进行单位向量缩放
        if norm != 0:
            self.lptc = self.lptc / norm

    def self_motion_feedback(self, motionState, inputMatrix, W_lptc2stmd):
        """ 实现公式(14)、(19)
            Args:
                motionState: 权重值
                inputMatrix: 原始灰度图
                W_lptc2stmd: 速度向量矩阵
            Returns:
                C_feedback: 自运动光流
            """

        # # 返回沿图像垂直方向和水平方向的梯度
        # grad_matrix = cp.gradient(icputMatrix)
        #
        # # C_LO : [grad_x, -grad_x, grad_y, -grad_y]
        # C_LO = [None] * 4
        # # C_LO[0] = grad_matrix[0]
        # # C_LO[1] = -grad_matrix[0]
        # # C_LO[2] = grad_matrix[1]
        # # C_LO[3] = -grad_matrix[1]
        #
        # self.C_LO_list[0].record_next(grad_matrix[0])
        # self.C_LO_list[1].record_next(-grad_matrix[0])
        # self.C_LO_list[2].record_next(grad_matrix[1])
        # self.C_LO_list[3].record_next(-grad_matrix[1])
        #
        # for i in range(4):
        #     C_LO[i] = Model.compute_temporal_conv(self.C_LO_list[i], self.C_LO_kernel, self.C_LO_list[i].pointer)
        #
        # # 计算自运动光流
        # C_feedback = cp.zeros_like(icputMatrix)
        # for v in lptcOpt.keys():
        #     for w in range(4):
        #         C_feedback += lptcOpt[v] * C_LO[w] * W_lptc2stmd[v][w]
        #
        # # 存储当前的计算值
        # self.feedback_list.record_next(C_feedback)
        #
        # return C_feedback

        # 2.
        # 返回沿图像垂直方向和水平方向的梯度
        grad_matrix = cp.gradient(inputMatrix)

        # C_LO : [grad_x, -grad_x, grad_y, -grad_y]
        C_LO = [grad_matrix[0], -grad_matrix[0], grad_matrix[1], -grad_matrix[1]]

        # 计算自运动光流
        C_feedback = cp.zeros_like(inputMatrix)
        for index, v in enumerate(self.direction_list):
            for w in range(4):
                C_feedback += motionState[index] * C_LO[w] * W_lptc2stmd[v][w]

        # # 存储当前的计算值
        self.feedback_list.record_next(C_feedback)

        return C_feedback

    def kalman_filter(self, drone_vy, drone_vz):
        """

         self.lptc:左右上下四个方向的权重值
         drone_vy: 当前的水平方向无人机速度(随提坐标系)
         drone_vz:当前的垂直方向无人机的速度(随提坐标系)
        """

        self.uav_vel[0] = -drone_vy if drone_vy < 0 else 0  # 背景向左运动
        self.uav_vel[1] = drone_vy if drone_vy > 0 else 0  # 背景向右运动
        self.uav_vel[2] = -drone_vz if drone_vz < 0 else 0  # 背景向上运动
        self.uav_vel[3] = drone_vz if drone_vz > 0 else 0  # 背景向下运动

        # 1. 预测
        # 先验估计
        self.X_prior = cp.dot(self.A, self.motion_state) + cp.dot(self.B, self.uav_vel)

        # 计算先验误差协方差矩阵P
        self.P_prior = cp.dot(cp.dot(self.A, self.P_posterior), self.A.T) + self.Q

        # print(f'P: {self.P_posterior}')
        # 2. 矫正
        # 计算卡尔曼增益K
        z = self.P_prior + self.R
        x = np.linalg.inv(cp.asnumpy(z))
        K = cp.dot(self.P_prior, cp.asarray(x))

        # 后验估计
        self.motion_state = self.X_prior + cp.dot(K, self.lptc - self.X_prior)

        # 更新状态估计协方差矩阵P
        self.P_posterior = cp.dot(self.I - K, self.P_prior)

    def lobula_plate_stmd_process(self, t5Opt, C_feedback, Mi1):
        """ 实现公式(20)、(21)

            Args:
                t5Opt: list类型，存储负部的右左下上四个背景运动方向矩阵
                C_feedback: cp.array类型，表示背景光流
                Mi1: 正部

            Returns:
                输出C_STMD
            """

        # C_STMD_Output = {}  # (20)式输出
        # for tau in gammaTau.keys():
        #     self.restrain_ON[tau].record_next(cp.maximum(Mi1 - eta_on * C_feedback, 0))
        #
        #     self.restrain_OFF[tau].record_next(
        #         cp.maximum(Model.compute_temporal_conv(listL2Opt, gammaTau[tau], listL2Opt.pointer) +
        #                    eta_off * Model.compute_temporal_conv(feedbackList, gammaTau[tau], feedbackList.pointer),
        #                    0))
        #
        #     C_STMD_Output[tau] = Model.compute_temporal_conv(self.restrain_ON[tau],
        #                                                         self.restrain_kernel,
        #                                                         self.restrain_ON[
        #                                                             tau].pointer) * Model.compute_temporal_conv(
        #         self.restrain_OFF[tau], self.restrain_kernel, self.restrain_OFF[tau].pointer)
        #
        #
        # STMD_Output = {}  # (21)式输出
        # # 计算小目标运动光流
        # for tau in gammaTau.keys():
        #     STMD_Output[tau] = {}
        #
        #     for index, direction in enumerate(('right', 'left', 'down', 'up')):
        #         STMD_Output[tau][direction] = C_STMD_Output[tau] * (
        #                 t5Opt[index] + Model.compute_temporal_conv(t4OptList[direction], gammaTau[tau],
        #                                                               t4OptList[direction].pointer))
        #
        # return STMD_Output, C_STMD_Output, Mi1 - eta_on * C_feedback

        # 1.
        stmd_no_feedback = {}  # (20)式输出
        for tau in self.stmd_tau:
            stmd_no_feedback[tau] = Mi1 * Model.compute_temporal_conv(self.l2_list, self.gamma_kernel_stmd[tau],
                                                                      self.l2_list.pointer)
            # 计算最小值和最大值
            matrix_min = cp.min(stmd_no_feedback[tau])
            matrix_max = cp.max(stmd_no_feedback[tau])

            # 进行归一化
            stmd_no_feedback[tau] = (stmd_no_feedback[tau] - matrix_min) / (matrix_max - matrix_min)

        # 2.
        C_STMD_Output = {}  # (20)式输出
        for tau in self.stmd_tau:
            on_output = cp.maximum(Mi1 - self.eta_on * C_feedback, 0)

            off_output = cp.maximum(
                Model.compute_temporal_conv(self.l2_list, self.gamma_kernel_stmd[tau], self.l2_list.pointer) +
                self.eta_off * Model.compute_temporal_conv(self.feedback_list, self.gamma_kernel_stmd[tau],
                                                           self.feedback_list.pointer), 0)

            C_STMD_Output[tau] = on_output * off_output
            C_STMD_Output[tau] = cp.maximum(C_STMD_Output[tau], 0)

        # 3.
        STMD_Output = {}  # (21)式输出
        # 计算小目标运动光流
        for tau in self.stmd_tau:
            STMD_Output[tau] = {}

            for index, direction in enumerate(self.direction_list):
                STMD_Output[tau][direction] = C_STMD_Output[tau] * (
                        t5Opt[index] + self.compute_temporal_conv(self.t4_list[direction], self.gamma_kernel_stmd[tau],
                                                                  self.t4_list[direction].pointer))

                STMD_Output[tau][direction] = cp.maximum(STMD_Output[tau][direction], 0)
                # 计算最小值和最大值
                matrix_min = cp.min(STMD_Output[tau][direction])
                matrix_max = cp.max(STMD_Output[tau][direction])

                # 进行归一化
                STMD_Output[tau][direction] = (STMD_Output[tau][direction] - matrix_min) / (matrix_max - matrix_min)

        return stmd_no_feedback, C_STMD_Output, STMD_Output

    def lobula_plate_stmd_process1(self, t5Opt, C_feedback, Mi1):
        """ 实现公式(20)、(21)

            Args:
                t5Opt: list类型，存储负部的右左下上四个背景运动方向矩阵
                C_feedback: cp.array类型，表示背景光流
                Mi1: 正部

            Returns:
                输出C_STMD
            """

        # 2.
        temp = self.temporalDiff - C_feedback
        SignalOn = cp.maximum(temp, 0)  # ON
        l2SignalOff = cp.maximum(-temp, 0)  # OFF
        self.l2SignalOff_list.record_next(l2SignalOff)
        # 2.
        C_STMD_Output = {}  # (20)式输出
        for tau in self.stmd_tau:
            C_STMD_Output[tau] = SignalOn * Model.compute_temporal_conv(self.l2SignalOff_list,
                                                                        self.gamma_kernel_opt[tau],
                                                                        self.l2SignalOff_list.pointer)
        # 3.
        STMD_Output = {}  # (21)式输出
        # 计算小目标运动光流
        for tau in self.stmd_tau:
            STMD_Output[tau] = {}

            for index, direction in enumerate(self.direction_list):
                STMD_Output[tau][direction] = C_STMD_Output[tau] * (
                        t5Opt[index] + self.compute_temporal_conv(self.t4_list[direction], self.gamma_kernel_stmd[tau],
                                                                  self.t4_list[direction].pointer))

                STMD_Output[tau][direction] = cp.maximum(STMD_Output[tau][direction], 0)
                # 计算最小值和最大值
                matrix_min = cp.min(STMD_Output[tau][direction])
                matrix_max = cp.max(STMD_Output[tau][direction])

                # 进行归一化
                STMD_Output[tau][direction] = (STMD_Output[tau][direction] - matrix_min) / (matrix_max - matrix_min)

        return C_STMD_Output, STMD_Output

    def VNC_TSDN(self, C_STMD_Output, lamda_DT):
        """实现公式(22)~(25)
            Args:
                C_STMD_Output: 双重字典
                lamda_DT: list类型，存储正部的右左下上四个背景运动方向矩阵

            Returns:
                1. C_TSDN
                2. fit_tau: 最优tau
                3. x0,j0: 目标位置
        """

        position = {}  # 保存每个tau下不同方向的点集
        position_union = {}  # 保存每个tau下四个方向的点集和
        C_TSDN = {}  # 公式(25)

        for tau in self.stmd_tau:
            C_TSDN[tau] = {}
            position[tau] = {}
            position_union[tau] = cp.zeros((0, 2), dtype=cp.uint16)

            for direction in self.direction_list:
                find_matrix = C_STMD_Output[tau][direction]
                position[tau][direction] = cp.argwhere(find_matrix > lamda_DT)

                if len(position[tau][direction]):
                    C_TSDN[tau][direction] = cp.mean(find_matrix[find_matrix > lamda_DT])
                    position_union[tau] = cp.vstack([position_union[tau], position[tau][direction]])
                else:
                    # 固定tau和方向下找不到点集, 就设置对应TSDN为0
                    C_TSDN[tau][direction] = 0.0

            # 最后求每个tau下四个方向点集的并集
            if len(position_union[tau]) > 1:
                position_union[tau] = cp.asarray(np.unique(position_union[tau].get(), axis=0))

        # 如果每个tau下点集都为0或者1,说明没有检测到结果
        if all(len(position_union[tau]) <= 1 for tau in self.stmd_tau):
            return C_TSDN, -1.0, -1.0, -1.0, -1.0, -1.0

        # 如果fit_tau下每个方向点集都为0或者1,说明没有检测到结果
        fit_tau = self.compute_fit_tau(position_union)
        if all(len(position[fit_tau][direction]) <= 1 for direction in self.direction_list):
            return C_TSDN, -1.0, -1.0, -1.0, -1.0, -1.0

        # 方向与数值的对应：{1: 'left', 2: 'right', 3: 'up', 4: 'down'}
        fit_direction, x0, j0 = self.compute_fit_direction(position[fit_tau])
        return C_TSDN, position, fit_tau, Model.find_index[fit_direction], x0, j0

    @staticmethod
    def compute_density(positions):
        """计算某个点集中的方差与均值
            Args:
                positions: shape为(n,2)，每一行为一个点集

            Returns:
                返回求出的三个值
        """

        # 计算质心点 (x0, y0)
        x0, y0 = cp.mean(positions, axis=0)

        # 计算每个点到质心点的欧式距离
        distances = cp.sqrt((positions[:, 0] - x0) ** 2 + (positions[:, 1] - y0) ** 2)

        # 计算欧式距离和的均值
        avg_of_distances = cp.sum(distances) / distances.shape[0]

        return avg_of_distances, cp.round(x0), cp.round(y0)

    def compute_fit_direction(self, position):
        """返回fit_direction与 target_position

            Args:
                position: 对应的位置点集矩阵
                iteration_list: 迭代列表


            Returns:最优的方向与求出的目标点
        """

        choose_matrix = cp.zeros((0, 4))  # 保存迭代值对应的方差、迭代值、均值点(x0,j0)

        i = cp.array(1)
        # 求不同方向下对应点集的方差与均值点
        for direction in self.direction_list:
            # 点集的数量一定要大于等于1
            if len(position[direction]) > 1:
                dis, x0, j0 = Model.compute_density(position[direction])
                choose_matrix = cp.vstack([choose_matrix, [dis, i, x0, j0]])
            i += 1

        # 按第一列数据排序矩阵
        sorted_matrix = choose_matrix[choose_matrix[:, 0].argsort()]

        # 找到数值最小的那一行
        return int(sorted_matrix[0, 1]), sorted_matrix[0, 2], sorted_matrix[0, 3]

    def compute_fit_tau(self, position):
        """求fit_tau

            Args:
                position: 不同tau下的点集

            Returns:
                最优的迭代tau
        """

        choose_matrix = cp.zeros((0, 2))  # 保存迭代值对应的方差、迭代值、均值点(x0,j0)

        for tau in self.stmd_tau:  # 求不同tau或者方向下对应点集的方差与均值点

            if len(position[tau]) > 1:  # 点集的数量一定要大于1
                dis, *_ = Model.compute_density(position[tau])
                choose_matrix = cp.vstack([choose_matrix, [dis, cp.array(tau)]])

        # 按第一列数据排序矩阵
        sorted_matrix = choose_matrix[choose_matrix[:, 0].argsort()]

        # 返回最小一行的tau
        return int(sorted_matrix[0, 1])

    @staticmethod
    def direction_of_target_motion(C_TSDN, fit_tau):
        """实现公式(26)

            Args:
                C_TSDN: 双重字典,(25)式输出
                fit_tau: 最优的tau值

            Returns:
                返回C_TSDN_Dir
        """

        # 生成从0到7*cp.pi/4的8个均匀分布的角度
        angles = cp.linspace(0, 7 * cp.pi / 4, 8)

        # 计算正弦值和余弦值
        sin_values = cp.sin(angles)
        cos_values = cp.cos(angles)

        # 将正弦值和余弦值组合成一个二维矩阵
        C_Dp = cp.row_stack((sin_values, cos_values))

        # 四个基本方向
        C_alpha_d = cp.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
        C_beta_d = cp.dot(C_alpha_d, C_Dp)

        vector = cp.empty(4)
        for index, direction in enumerate(['left', 'right', 'up', 'down']):
            vector[index] = C_TSDN[fit_tau][direction]

        return cp.dot(vector, C_beta_d)

    @staticmethod
    def velocity_of_target_motion(C_TSDN, tauList=None):
        """实现公式(27)

            Args:
                C_TSDN: 双重字典,(25)式输出
                tauList: 预定义tau列表

            Returns:
                返回C_TSDN_Vel
        """

        C_TSDN_Vel = {}

        for tau in tauList:
            value = 0.
            for direction in ['left', 'right', 'up', 'down']:
                value += C_TSDN[tau][direction]

            C_TSDN_Vel[tau] = value

        return C_TSDN_Vel

    @staticmethod
    def direction_of_target_position(P_prey, x0, j0):
        """实现公式(28)

            Args:
                P_prey: 目标位置
                x0、j0:图像中心点
                x0: 行下标
                j0: 列下标

            Returns:
                返回C_TSDN_Pos
        """

        # 生成从0到7*cp.pi/4的8个均匀分布的角度
        angles = cp.linspace(0, 7 * cp.pi / 4, 8)

        # 计算正弦值和余弦值
        sin_values = cp.sin(angles)
        cos_values = cp.cos(angles)

        # 将正弦值和余弦值组合成一个二维矩阵
        C_Dp = cp.row_stack((sin_values, cos_values))

        vector = cp.array([P_prey[1] - j0, P_prey[0] - x0])

        return cp.dot(vector, C_Dp)

    @staticmethod
    def compute_temporal_conv(iptCell, kernel, pointer=None):
        """
        Computes temporal convolution.

        Parameters:
        - iptCell: A list of arrays where each element has the same dimension.
        - kernel: A vector representing the convolution kernel.
        - headPointer: Head pointer of the input cell array (optional).

        Returns:
        - optMatrix: The result of the temporal convolution.
        """

        # Default value for headPointer
        if pointer is None:
            pointer = len(iptCell) - 1

        # Initialize output matrix
        if iptCell[pointer] is None:
            return None

        # Ensure kernel is a vector
        kernel = cp.squeeze(kernel)
        if not cp.ndim(kernel) == 1:
            raise ValueError('The kernel must be a vector.')

        # Determine the lengths of input cell array and kernel
        k1 = len(iptCell)
        k2 = len(kernel)
        length = min(k1, k2)

        optMatrix = cp.zeros_like(iptCell[pointer])
        # Perform temporal convolution
        for t in range(length):
            j = (pointer - t) % k1
            if cp.abs(kernel[t]) > 1e-6 and iptCell[j] is not None:
                optMatrix += iptCell[j] * kernel[t]

        return optMatrix

    def run(self, gray_img, drone_vy, drone_vz, my_list=None):
        """ 主函数
            gray_img为cupy类型矩阵
            drone_vy,drone_vz为无人机速度
        """

        """
            gray_img为cupy类型矩阵
            drone_vy,drone_vz为无人机速度
        """

        '''1.Retina'''
        retinaOpt = gaussian_filter(gray_img, self.sigma)

        '''2.Lamina'''
        l1, l2, img_smoothing = self.lamina_process(retinaOpt)  # mil, tm1

        '''3.Medulla'''
        # ME ON/OFF Channel
        # 按照现在的gamma_kernel_emd, 相当于往前延迟了1帧
        mi1, tm3 = l1, self.compute_temporal_conv(self.l1_list, self.gamma_kernel_emd, self.l1_list.pointer)
        tm1, tm2 = l2, self.compute_temporal_conv(self.l2_list, self.gamma_kernel_emd, self.l2_list.pointer)

        '''4.Lobula'''
        # LO
        # 做EMD匹配，t4和t5输出的四个背景响应方向的顺序为：右左下上
        t4, t5 = self.lobula_process(tm1, tm2, mi1, tm3, self.x0)

        # LOP LPTC
        self.lobula_plate_lptc_process(t4, t5, self.W_lo2Lptc)

        # self_motion_feedback
        feedback = self.self_motion_feedback(self.lptc, img_smoothing, self.W_lptc2stmd)

        # LOP STMD(公式21)
        stmd_no_feedback, tau_stmd, C_STMD_Output = self.lobula_plate_stmd_process(t5, feedback, mi1)

        C_TSDN, position, fit_tau, fit_direction, x0, j0 = self.VNC_TSDN(C_STMD_Output, self.lamda_DT)

        my_list[2] = stmd_no_feedback[fit_tau]
        my_list[0] = [-1.0, -1.0]
        if x0 != -1.0:
            my_list[0] = [x0.get(), j0.get()]
            my_list[1] = C_STMD_Output[fit_tau][fit_direction]

        # 计算目标运动方向
        # C_TSDN_Dir = direction_of_target_motion(C_TSDN, fit_tau)

        # 计算目标运动速度
        # C_TSDN_Vel = velocity_of_target_motion(C_TSDN, tau2)

        # 计算目标位置方向
        # C_TSDN_Pos = direction_of_target_position(target_pos, grayImg.shape[0] // 2, grayImg.shape[1] // 2)

    def save_picture(self, gray_img):
        '''1.Retina'''
        retinaOpt = gaussian_filter(gray_img, self.sigma)
        self.list_lamina.record_next(retinaOpt)

    def run1(self, gray_img, drone_vy, drone_vz, my_list=None):
        """
            gray_img为cupy类型矩阵
            drone_vy,drone_vz为无人机速度
        """

        '''1.Retina'''
        retinaOpt = gaussian_filter(gray_img, self.sigma)

        '''2.Lamina'''
        l1, l2, img_smoothing = self.lamina_process(retinaOpt)  # mil, tm1

        '''3.Medulla'''
        # ME ON/OFF Channel
        # 按照现在的gamma_kernel_emd, 相当于往前延迟了1帧
        mi1, tm3 = l1, self.compute_temporal_conv(self.l1_list, self.gamma_kernel_emd, self.l1_list.pointer)
        tm1, tm2 = l2, self.compute_temporal_conv(self.l2_list, self.gamma_kernel_emd, self.l2_list.pointer)

        '''4.Lobula'''
        # LO
        # 做EMD匹配，t4和t5输出的四个背景响应方向的顺序为：右左下上
        t4, t5 = self.lobula_process(tm1, tm2, mi1, tm3, self.x0)

        # LOP LPTC
        self.lobula_plate_lptc_process(t4, t5, self.W_lo2Lptc)

        self.kalman_filter(drone_vy, drone_vz)

        # self_motion_feedback
        feedback = self.self_motion_feedback(self.motion_state, img_smoothing, self.W_lptc2stmd)

        # LOP STMD(公式21)
        # stmd_no_feedback, tau_stmd, tsdn = self.lobula_plate_stmd_process(t5, feedback, mi1)
        tau_stmd, tsdn = self.lobula_plate_stmd_process1(t5, feedback, mi1)

        C_TSDN, position, fit_tau, fit_direction, x0, j0 = self.VNC_TSDN(tsdn, self.lamda_DT)

        # my_list[2] = stmd_no_feedback[2].get()
        my_list[0] = [-1.0, -1.0]
        if x0 != -1.0:
            my_list[0] = [x0.get(), j0.get()]
            # my_list[1] = tsdn[fit_tau][fit_direction].get()
            # my_list[3] = l1.get()
            # my_list[4] = l2.get()
            # print(cp.max(my_list[1]))
            # print('x0, j0', x0, j0)

        my_list[5] = [self.lptc[1].get() - self.lptc[0].get(), self.lptc[3].get() - self.lptc[2].get()]
        my_list[6] = [self.motion_state[1].get() - self.motion_state[0].get(),
                       self.motion_state[3].get() - self.motion_state[2].get()]
        #
        # # 打印
        # print(f"X_prior: {self.X_prior}")
        # print(f"motion_state: {self.motion_state}  {self.motion_state[0] - self.motion_state[1]}")
        # print('-----')


@dataclass
class CircularList(list):
    """
    CircularList represents a circular buffer for storing input matrices.
    """
    initLen: int = 0  # Default length of the circular buffer
    pointer: int = -1  # Pointer to current position in the circular buffer

    def __post_init__(self) -> List:
        """
        Post-initialization method to initialize the circular buffer with empty list.
        """
        if self.initLen:
            self.extend([None] * self.initLen)

    def reset(self) -> None:
        """
        Method to reset the circular buffer to a new length.
        """
        self.clear()  # clear List
        self.__post_init__()  # Reinitialize CircularList object
        self.pointer = -1  # Reset the pointer to initial position

    def move_pointer(self) -> None:
        """
        Method to move the circular buffer pointer to the next position.
        """
        self.pointer = (self.pointer + 1) % self.initLen

    def cover(self, iptMatrix: Any) -> None:
        """
        Method to cover the current position of the circular buffer with an input matrix.

        Parameters:
        - iptMatrix: Input matrix to cover the current position.
        """
        self[self.pointer] = iptMatrix

    def record_next(self, iptMatrix: Any) -> None:
        """
        Method to record an input matrix in the circular buffer, after moving the pointer to the next position.

        Parameters:
        - iptMatrix: Input matrix to be recorded.
        """
        self.move_pointer()
        self.cover(iptMatrix)
