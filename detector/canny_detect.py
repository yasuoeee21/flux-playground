import cv2
import numpy as np
from PIL import Image
from controlnet_aux import CannyDetector

class CannyDetect:
    def __init__(self):
        self.processor = CannyDetector()
    
    def polygon2canny(self, polygon_points_list):
        # 计算图像宽高（根据多边形的坐标动态调整）
        all_points = np.concatenate(polygon_points_list)  # 将所有多边形的点合并为一个数组
        x_coords, y_coords = all_points[:, 0], all_points[:, 1]  # 分别获取所有点的 x 和 y 坐标
        max_x, max_y = int(np.ceil(x_coords.max())), int(np.ceil(y_coords.max()))  # 找到最大值并取整
        min_x, min_y = int(np.floor(x_coords.min())), int(np.floor(y_coords.min()))  # 找到最小值并取整

        # 为了避免多边形刚好贴边，可以增加一些边距
        margin = 10
        width = max_x - min_x + 2 * margin
        height = max_y - min_y + 2 * margin

        # 创建一个空白图像
        image = np.zeros((height, width), dtype=np.uint8)

        # 遍历每个多边形并绘制到图像上
        for polygon_points in polygon_points_list:
            # 将每个多边形的点转换为整数坐标，并加上偏移量（考虑到最小值和边距）
            polygon_points = np.array(polygon_points, dtype=np.float32)
            polygon_points[:, 0] -= min_x - margin  # 调整 x 坐标
            polygon_points[:, 1] -= min_y - margin  # 调整 y 坐标
            polygon_points = polygon_points.astype(np.int32)
            
            # 在图像上绘制多边形
            cv2.polylines(image, [polygon_points], isClosed=True, color=255, thickness=2)

        # 使用 Canny 边缘检测
        canny_image = self.processor(image, low_threshold=50, high_threshold=200, detect_resolution=1024, image_resolution=1024)
        # 显示结果
        canny_image = Image.fromarray(canny_image)
        return canny_image