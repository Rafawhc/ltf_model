import cv2
import numpy as np
import json


def get_black_target_positions(video_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 存储每帧的目标位置
    positions = []

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # 转换为灰度图像
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 应用阈值，以获得黑色目标
        _, thresholded = cv2.threshold(gray_frame, 120, 255, cv2.THRESH_BINARY_INV)

        # 查找轮廓
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 存储当前帧目标位置
        current_frame_positions = []

        for contour in contours:
            # 计算边界框
            x, y, w, h = cv2.boundingRect(contour)
            # center_x = x + w / 2
            # center_y = y + h / 2
            # brcnt = np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]])
            # cv2.drawContours(frame, [brcnt], -1, (255, 255, 255), 1)
            # cv2.imshow("result", frame)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            # 将结果添加到当前帧的位置列表中
            current_frame_positions.append([y, x, w, h])

        # 如果没有检测到小目标，填充位置为 (0, 0, 0, 0)
        if not current_frame_positions:
            current_frame_positions.append([0, 0, 0, 0])

        # 将当前帧的位置添加到总列表中
        positions.append(current_frame_positions)

    cap.release()

    # 将列表转换为 numpy 数组
    # num 是每一帧中最大目标数量（最大数量的目标在不同帧之间可能会有所不同，因此需要找到最大值并进行填充）
    max_num_targets = max(len(frame) for frame in positions)
    print('max_num_targets', max_num_targets)
    # 填充到 (n, num, 4) 的格式
    padded_positions = np.zeros((len(positions), max_num_targets, 4), dtype=float)

    for i, frame_positions in enumerate(positions):
        for j, position in enumerate(frame_positions):
            padded_positions[i, j] = position

    return padded_positions


def save_positions_to_json(positions, json_file_path):
    # 将 numpy 数组转换为列表
    positions_list = positions.tolist()

    # 创建字典
    output_dict = {"groundTruth": positions_list}

    # 将字典保存为 JSON 文件
    with open(json_file_path, 'w') as json_file:
        json.dump(output_dict, json_file, indent=4)


# 使用示例
video_path = './videos6/target0.2.avi'  # 替换为您的视频路径
json_file_path = './videos6/target0.2_62.5Hz.json'  # 输出的 JSON 文件路径

positions = get_black_target_positions(video_path)
save_positions_to_json(positions, json_file_path)

print(f"Positions saved to {json_file_path}")
