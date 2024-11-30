from typing import Any
from torch.utils.tensorboard import SummaryWriter
import cv2
import numpy as np

class TensorboardVisualizer:
    def __init__(self, log_dir: str="."):
        """
        初始化函数，创建一个SummaryWriter对象用于记录训练数据到指定的日志目录。

        Args:
            log_dir (str): 存储tensorboard日志文件的目录路径。
        """
        self.writer = SummaryWriter(log_dir)

    def add_scalar(self, tag, scalar_value, global_step):
        """
        向tensorboard添加标量数据。

        Args:
            tag (str): 数据的标签，用于在tensorboard中区分不同的数据系列。
            scalar_value (float or int): 要记录的标量值。
            global_step (int): 当前的训练步数。
        """
        self.writer.add_scalar(tag, scalar_value, global_step)

    def add_image(self, tag, img_tensor, global_step):
        """
        向tensorboard添加图像数据。

        Args:
            tag (str): 图像数据的标签。
            img_tensor (torch.Tensor or tf.Tensor): 要记录的图像张量，形状应为 [batch_size, height, width, channels] 或 [height, width, channels]。
            global_step (int): 当前的训练步数。
        """
        self.writer.add_image(tag, img_tensor, global_step)

    def close(self):
        """
        关闭SummaryWriter，完成日志记录后需要调用此方法来释放资源。
        """
        self.writer.close()
        
def generate_video(frames:np.array, video_name:str="a.mp4"):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter(video_name, fourcc, 25.0, (width, height))
    # 遍历图片列表并写入视频
    for index, frame in enumerate(frames):
        fid_label = f"fid: {index+1}"
        cv2.putText(frame, fid_label, org=(10, 30), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness = 1)
        video.write(frame)
    video.release()
    cv2.destroyAllWindows()
    print("=== complete ===")