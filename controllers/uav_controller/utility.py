import csv

import cv2
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import cupy as cp

# global variables
i = 0


def webots_save_image(camera, pathname):
    """保存camera捕捉到的图像数据
    """

    global i

    if camera.saveImage(pathname + "/{:05}.png".format(i), 50) == 0:
        i += 1
    else:
        print(pathname + "图片未能保存成功...")


def webots_get_image(camera, camera_height, camera_width):
    """获取camera捕捉到的图像数据

    description:
        像素值整体除以255并且转成了float32的数据类型,精度为6-7位

    """

    cameraData = camera.getImage()
    rgba_raw = np.frombuffer(cameraData, np.uint8).reshape((camera_height, camera_width, 4))
    img = rgba_raw[..., :3].astype(np.uint8)
    grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY) / 255
    grayImg = grayImg.astype(np.float32)

    return grayImg

def get_position(img):
    """针对纯色背景的小目标位置获取
    """

    filtered_points = np.argwhere(img < 100)

    # 计算这些点的 x 和 y 坐标的均值
    x_mean, y_mean = np.mean(filtered_points, axis=0)

    return np.around(x_mean), np.around(y_mean)


def initialize():
    """初始化全局变量
    """
    global i

    i = 0
    print('初始化成功...')


def img_shows(img):
    """展示图像
    """

    # 如果不进行判断, 负数会被astype(np.uint8)转化为255
    img = np.where(img >= 0, img, 0.)
    imgs = (img * 255).astype(np.uint8)
    cv2.imshow('image', imgs)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()


def save_image(img, path):
    """保存图像数据
    """
    global i

    # 如果不进行判断, 负数会被astype(np.uint8)转化为255
    img = np.where(img >= 0, img, 0.)
    imgs = np.uint8(img * 255)
    cv.imwrite(path + "/{:05}.png".format(i), imgs)


def save_figure(img, points, path):
    """保存3D可视化矩阵

    Arg:
        img: 二维矩阵图像
        points: 矩阵中的一些坐标点位置,对应shape为(n,2)
        path:保存图片的路径

    description:
        首先把img绘制成3维线框图(通过绘制像素点之间的连线以形成网格状的结构),然后再把points中的点集在三维图中在对应位置用plot进行绘制覆盖

    """
    global i

    # 生成默认的x和y坐标
    x = np.arange(img.shape[0])
    y = np.arange(img.shape[1])
    z = img.copy()

    # 筛选的点集首先要进行像素点值赋值0, 不然会覆盖会导致重合
    for point in points:
        z[int(point[0])][int(point[1])] = 0
    y, x = np.meshgrid(y, x)

    # 创建一个3D绘图对象
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制mesh图
    ax.plot_wireframe(x, y, z, cmap='viridis', color="green")
    # ax.plot_wireframe(x_stmd, y_stmd, z_stmd, cmap='viridis', color="red")
    for point in points:
        x_coord, y_coord = point
        z_value = img[int(x_coord), int(y_coord)]

        # 在三维图中用plot函数绘制红色直线
        ax.plot([x_coord, x_coord], [y_coord, y_coord], [0, z_value], color='red', linewidth=2)

    # 设置图形标题和轴标签
    ax.set_xlabel('i axis')
    ax.set_ylabel('j axis')
    ax.set_zlabel('response')

    # 设置视角
    ax.view_init(elev=1, azim=2)  # elev 是仰角, azim 是方位角

    # 保存图形到文件
    plt.savefig(path + "/{:05}.png".format(i), dpi=300)  # f'./line_plot{i}.png'

    # 显示图形
    # plt.show()

    # 释放资源
    plt.close()


def show_one_or_two_figure(img1, points=None, target_pos=None, img2=None, original=False):
    """3D可视化两张或者一张图片进行对比

        img1是二维矩阵的3D化, 可以把对应点集points用红色线条醒目标出
        img2可以是二维矩阵的3D化, 或者是原始彩色图的展示(如果是原始彩色图,使用opencv读取到的图片需要进行通道转化: cv.cvtColor(img2, cv2.COLOR_BGR2RGB))

    Arg:
        img1: 图像1
        img: 图像2
        points: 矩阵中的一些坐标点位置,对应shape为(n,2)
        target_pos: 目标位置
        original: 传入的img2是否是彩色图
    """

    """ 创建一个3D绘图对象"""
    fig = plt.figure(figsize=(22, 10))

    '''绘制第一个子图'''
    ax1 = fig.add_subplot(121 if img2 is not None else 111, projection='3d')
    # 生成默认的x和y坐标
    x = np.arange(img1.shape[0])
    y = np.arange(img1.shape[1])
    y, x = np.meshgrid(y, x)
    z = img1.copy()

    if points is not None:
        # 筛选的点集首先要进行像素点值赋值0, 不然会覆盖会导致重合
        for point in points:
            z[int(point[0])][int(point[1])] = 0

        # 绘制mesh图
        ax1.plot_wireframe(x, y, z, cmap='viridis', color="red")

        for point in points:
            x_coord, y_coord = point
            z_value = img1[int(x_coord), int(y_coord)]

            # 在三维图中用plot函数绘制绿色直线
            ax1.plot([x_coord, x_coord], [y_coord, y_coord], [0, z_value], color='red', linewidth=2)

        if target_pos is not None:
            ax1.text(0, 0, 0, f'target pos(j,i): ({target_pos[1]:.1f}, {target_pos[0]:.2f})', transform=ax1.transAxes,
                     fontsize=12, color='blue',
                     va='center', ha='left')
    else:
        # 绘制mesh图
        ax1.plot_wireframe(x, y, z, cmap='viridis', color="green")

    # 设置 x/y 轴刻度每50个单位
    ax1.set_xticks(np.arange(0, img1.shape[0] + 1, 50))
    ax1.set_yticks(np.arange(0, img1.shape[1] + 1, 50))

    # 设置图形标题和轴标签
    ax1.set_xlabel('i axis')
    ax1.set_ylabel('j axis')
    ax1.set_zlabel('response')
    ax1.view_init(elev=0, azim=0)  # elev 是仰角, azim 是方位角

    ''' 绘制第二个子图'''
    if img2 is not None:

        if original:
            ax2 = fig.add_subplot(122)
            ax2.set_xticks(np.arange(0, img1.shape[0] + 1, 25))
            ax2.set_yticks(np.arange(0, img1.shape[1] + 1, 25))
            ax2.imshow(cv.cvtColor(img2, cv2.COLOR_BGR2RGB))

        else:

            ax2 = fig.add_subplot(122, projection='3d')
            # 生成默认的x和y坐标
            x = np.arange(img2.shape[0])
            y = np.arange(img2.shape[1])
            z = img2.copy()

            y, x = np.meshgrid(y, x)

            # 绘制mesh图
            ax2.plot_wireframe(x, y, z, cmap='viridis', color="green")
            # 设置视角
            ax2.view_init(elev=0, azim=0)  # elev 是仰角, azim 是方位角

            # 设置图形标题和轴标签
            ax2.set_xlabel('i axis')
            ax2.set_ylabel('j axis')
            ax2.set_zlabel('response')

        # 设置 x/y 轴刻度每50个单位
        ax2.set_xticks(np.arange(0, img2.shape[0] + 1, 50))
        ax2.set_yticks(np.arange(0, img2.shape[1] + 1, 50))

    # 显示图形
    plt.tight_layout()
    plt.show()

    # 释放资源
    plt.close()


def show_multiple_figure(img, target_data=None):
    """3D可视化img. img是一个包含多张图片的对象


    Arg:
        img: 字典类型的图像列表
        target_pos: 目标位置

    """

    """ 创建一个3D绘图对象"""
    fig = plt.figure(figsize=(22, 10))

    figure_choose = {
        1: [111],
        2: [121, 122],
        3: [131, 132, 133],
        4: [221, 222, 223, 224],
        5: [231, 232, 233, 234, 235],
        6: [231, 232, 233, 234, 235, 236],
        7: [241, 242, 243, 244, 245, 246, 247],
        8: [241, 242, 243, 244, 245, 246, 247, 248],
        9: [251, 252, 253, 254, 255, 256, 257, 258, 259],
        10: [251, 252, 253, 254, 255, 256, 257, 258, 259, [2, 5, 10]]
    }

    image_num = len(img)
    subplot_list = figure_choose[image_num]

    axes = []
    for i in range(image_num):  # 添加子图
        if i >= 9:
            ax = fig.add_subplot(subplot_list[i][0], subplot_list[i][1], subplot_list[i][2], projection='3d')
        else:
            ax = fig.add_subplot(subplot_list[i], projection='3d')

        ax.view_init(elev=22, azim=30)  # elev 是仰角, azim是方位角
        axes.append(ax)

    for ax, j in zip(axes, img.keys()):

        cur_img = img[j]
        # 生成默认的x和y坐标
        x = np.arange(cur_img.shape[0])
        y = np.arange(cur_img.shape[1])
        y, x = np.meshgrid(y, x)
        z = cur_img

        if target_data is not None:

            target_pos_x = int(target_data[0])
            target_pos_y = int(target_data[1])
            target_pos_z = cur_img[target_pos_x, target_pos_y]

            # 绘制mesh图
            z[target_pos_x][target_pos_y] = 0
            ax.plot_wireframe(x, y, z, cmap='viridis', color="green")

            # 在三维图中用plot函数绘制绿色直线
            ax.plot([target_pos_x, target_pos_x], [target_pos_y, target_pos_y], [0, target_pos_z], color='red',
                    linewidth=6)

            ax.text(0, 0, 0, f'target pos(i,j): ({target_pos_x:.1f}, {target_pos_y:.2f})', transform=ax.transAxes,
                    fontsize=12, color='blue',
                    va='center', ha='left')

        else:
            # 绘制mesh图
            ax.plot_wireframe(x, y, z, cmap='viridis', color="green")

        # 设置 x/y 轴刻度每50个单位
        ax.set_xticks(np.arange(0, cur_img.shape[0] + 1, 50))
        ax.set_yticks(np.arange(0, cur_img.shape[1] + 1, 50))

        # 设置图形标题和轴标签以及标题
        ax.set_xlabel('i axis')
        ax.set_ylabel('j axis')
        ax.set_zlabel('response')
        ax.set_title(f'tau: {j} ')
        ax.view_init(elev=0, azim=0)  # elev 是仰角, azim是方位角

    plt.tight_layout()
    plt.show()
    plt.close()



def save_to_csv(filename, target_positions, tracking_positions):
    """保存数据到指定的csv文件中

        Arg:
            filename: 保存的文件路径,比如: './simulation_data.csv'
            target_positions: 目标三维坐标数据, [[x, y, z]]类型的坐标列表
            tracking_position: 追踪器三维坐标数据, [[x, y, z]]类型的坐标列表

        Description:
            可以多次调用该函数用来向CSV中添加多个数据:
                for i in range(2):
                    save_to_csv('simulation_data.csv', target_positions, tracking_positions)

    """

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        for tp, trp in zip(target_positions, tracking_positions):
            writer.writerow([*tp, *trp])


'''评价指标(ROC)'''


def TN(true_img, predict_img):
    """true_img与predict_img都是二值图像
        predict_img是阈值截取后的二值图像
    """

    assert(true_img.shape == predict_img.shape)
    return cp.sum(cp.where(true_img[predict_img == 0] == 0, 1, 0))


def FP(true_img, predict_img):
    """true_img与predict_img都是二值图像
            predict_img是阈值截取后的二值图像
        """

    assert (true_img.shape == predict_img.shape)
    return cp.sum(cp.where(true_img[predict_img == 1] == 0, 1, 0))


def FN(true_img, predict_img):
    """true_img与predict_img都是二值图像
            predict_img是阈值截取后的二值图像
        """

    assert (true_img.shape == predict_img.shape)
    return cp.sum(cp.where(true_img[predict_img == 0] == 1, 1, 0))


def TP(true_img, predict_img):
    """true_img与predict_img都是二值图像
            predict_img是阈值截取后的二值图像
        """

    assert (true_img.shape == predict_img.shape)
    return cp.sum(cp.where(true_img[predict_img == 1] == 1, 1, 0))


def TPR(true_img, predict_img):
    """true_img与predict_img都是二值图像
                predict_img是阈值截取后的二值图像
            """

    assert (true_img.shape == predict_img.shape)
    tp = TP(true_img, predict_img)
    fn = FN(true_img, predict_img)
    try:
        return tp / (tp + fn)
    except:
        return 0.0


def FPR(true_img, predict_img):
    """true_img与predict_img都是二值图像
                    predict_img是阈值截取后的二值图像

    做横坐标
                """

    assert (true_img.shape == predict_img.shape)
    fp = FP(true_img, predict_img)
    tn = TN(true_img, predict_img)
    try:
        return fp / (fp + tn)
    except:
        return 0.0


def confusion_matrix(true_img, predict_img):
    """true_img与predict_img都是二值图像
                predict_img是阈值截取后的二值图像
            """

    assert (true_img.shape == predict_img.shape)
    return cp.array([
        [TN(true_img, predict_img), FP(true_img, predict_img)],
        [FN(true_img, predict_img), TP(true_img, predict_img)]
    ])


def precision_score(true_img, predict_img):
    """true_img与predict_img都是二值图像
                    predict_img是阈值截取后的二值图像
                """

    assert (true_img.shape == predict_img.shape)
    tp = TP(true_img, predict_img)
    fp = FP(true_img, predict_img)
    try:
        return tp / (tp + fp)
    except:
        return 0.0


def recall_score(true_img, predict_img):
    """true_img与predict_img都是二值图像
                        predict_img是阈值截取后的二值图像
                    """

    assert (true_img.shape == predict_img.shape)
    tp = TP(true_img, predict_img)
    fn = FN(true_img, predict_img)
    try:
        return tp / (tp + fn)
    except:
        return 0.0


def f1_score(true_img, predict_img):
    """更好的表征precision与recall,这两个指标有一个小，就会使得发f1也小"""

    assert (true_img.shape == predict_img.shape)
    precision = precision_score(true_img, predict_img)
    recall = recall_score(true_img, predict_img)

    try:
        return 2 * precision * recall / (precision + recall)
    except:
        return 0.0


def ROC_curve_single_img(true_img, predict_img):

    return FPR(true_img, predict_img), TPR(true_img, predict_img)


def show_scatter(data1, data2, value1=None, value2=None):
    """
    两种点集的可视化


    data1 data2:图像上的点集
    value1 value2:对应点集的图像矩阵

    """

    # 绘制散点图
    plt.scatter(data1[:, 1], data1[:, 0], color='blue', label='diff_img', marker='o')
    plt.scatter(data2[:, 1], data2[:, 0], color='red', label='feedback', marker='x')

    # 示例数据
    if value1 is not None and value2 is not None:
        print(data1.shape, data2.shape)
        len = data1.shape[0] if data1.shape[0] <= data2.shape[0] else data2.shape[0]
        for i in range(len):
            if np.any(np.all(data2 == data1[i], axis=1)):
                j ,i = data1[i, 1], data1[i, 0]
                plt.text(j, i, f'{value2[i, j] + value1[i, j]:.2f}', fontsize=10, ha='right', color='blue')


    # 添加标题和标签
    plt.title('Scatter Plot with Two Data Groups')
    plt.xlabel('j-axis')
    plt.ylabel('i-axis')

    # 显示图例
    plt.legend()

    # 显示网格
    plt.grid(True)

    # 显示图表
    plt.show()

def test_model():
    # 打开两个视频文件
    video_path_1 = './bgd0.4_target0.20.avi'  # 替换为第一个视频的路径
    video_path_2 = './GX010071-1.mp4'  # 替换为第二个视频的路径

    cap1 = cv2.VideoCapture(video_path_1)
    cap2 = cv2.VideoCapture(video_path_2)

    # 设置图形的显示大小
    plt.figure(figsize=(10, 5))

    while True:
        # 读取两个视频的每一帧
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # 如果视频读取完毕，则退出
        if not ret1 or not ret2:
            break

        # 将每一帧转换为灰度图像
        gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # 清除之前的图像，准备显示新的一帧
        plt.clf()

        # 显示两个视频的灰度图
        plt.subplot(1, 2, 1)  # 第一个子图
        plt.imshow(gray_frame1, cmap='gray')
        plt.title('Video 1 - Grayscale')
        plt.axis('off')

        plt.subplot(1, 2, 2)  # 第二个子图
        plt.imshow(gray_frame2, cmap='gray')
        plt.title('Video 2 - Grayscale')
        plt.axis('off')

        # 刷新显示
        plt.draw()
        plt.pause(0.01)

    # 释放视频资源
    cap1.release()
    cap2.release()
    plt.show()

