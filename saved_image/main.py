import cv2 as cv


def save_follow_video():
    # 1. 获取图像的属性（宽和高，）,并将其转换为整数
    frame_width = 600
    frame_height = 500
    # 2. 创建保存视频的对象，设置编码格式，帧率，图像的宽高等
    fourcc = cv.VideoWriter_fourcc(*'DIVX')
    follow = []
    for i in range(1529):  # 保存跟随运动视频的前1552帧，可修改
        follow.append("follow/{:05}.png".format(i))

    result_follow = cv.VideoWriter('./videos/follow04.avi', fourcc, 40.0, (frame_width, frame_height))

    i = 0
    while i != 1529:
        # 获取视频中的每一帧图像
        follow_frame = cv.imread(follow[i])
        follow_frame = cv.cvtColor(follow_frame, cv.COLOR_BGR2GRAY)  # 换换成灰度图
        cv.imwrite("./test.jpg", follow_frame)
        gray_output_image = cv.imread("./test.jpg")
        # 视频写入
        result_follow.write(gray_output_image)

        # 跟新下标
        i += 1

    # 释放资源
    result_follow.release()


def save_uniform_velocity_video():
    # 1. 获取图像的属性（宽和高，）,并将其转换为整数
    frame_width = 600
    frame_height = 500
    # 2. 创建保存视频的对象，设置编码格式，帧率，图像的宽高等
    fourcc = cv.VideoWriter_fourcc(*'DIVX')
    up = []
    down = []
    left = []
    right = []
    forward = []
    backward = []
    for i in range(500):  # 保存匀速运动视频的前500帧，可修改
        up.append("up/{:05}.png".format(i))
        down.append("down/{:05}.png".format(i))
        left.append("left/{:05}.png".format(i))
        right.append("right/{:05}.png".format(i))
        forward.append("forward/{:05}.png".format(i))
        backward.append("backward/{:05}.png".format(i))

    result_up = cv.VideoWriter('./videos/up.avi', fourcc, 20.0, (frame_width, frame_height))
    result_down = cv.VideoWriter('./videos/down.avi', fourcc, 20.0, (frame_width, frame_height))
    result_left = cv.VideoWriter('./videos/left.avi', fourcc, 20.0, (frame_width, frame_height))
    result_right = cv.VideoWriter('./videos/right.avi', fourcc, 20.0, (frame_width, frame_height))
    result_forward = cv.VideoWriter('./videos/forward.avi', fourcc, 20.0, (frame_width, frame_height))
    result_backward = cv.VideoWriter('./videos/backward.avi', fourcc, 20.0, (frame_width, frame_height))

    i = 0
    while i != 500:
        # 获取视频中的每一帧图像
        up_frame = cv.imread(up[i])
        down_frame = cv.imread(down[i])
        left_frame = cv.imread(left[i])
        right_frame = cv.imread(right[i])
        forward_frame = cv.imread(forward[i])
        backward_frame = cv.imread(backward[i])
        # 视频写入
        result_up.write(up_frame)
        result_down.write(down_frame)
        result_left.write(left_frame)
        result_right.write(right_frame)
        result_forward.write(forward_frame)
        result_backward.write(backward_frame)

        # 跟新下标
        i += 1

    # 6.释放资源
    result_up.release()
    result_down.release()
    result_left.release()
    result_right.release()
    result_forward.release()
    result_backward.release()


def save_video():
    # 1. 获取图像的属性（宽和高，）,并将其转换为整数
    width = 450
    height = 450
    # 2. 创建保存视频的对象，设置编码格式，帧率，图像的宽高等
    fourcc = cv.VideoWriter_fourcc(*'DIVX')

    img_list = []

    for i in range(26):  # 保存匀速运动视频的前500帧，可修改
        img_list.append("right/{:05}.png".format(i))

    result_right = cv.VideoWriter('./videos6/video12.avi', fourcc, 62.5, (width, height))

    for i in range(26):
        # 获取视频中的每一帧图像
        right_frame = cv.imread(img_list[i])
        # cv.imshow('r',right_frame)
        # cv.waitKey(0) # 等待按键
        # 视频写入
        result_right.write(right_frame)

    # 6.释放资源
    result_right.release()


if __name__ == '__main__':
    save_video()
