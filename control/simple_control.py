import os
import json
import logging
from detect.yolov8 import YoloV8Detector
from common.config import config


# 向前 'A'
# 向右前 'B'
# 向右 'C'
# 向右后 'D'
# 向后 'E'
# 向左后 'F'
# 向左 'G'
# 向左前 'H'
# 加速 'X'
# 减速 'Y'
# 停止 'Z'
# 开始说话 'S'
# 停止说话 'T'
class Command:
    FORWARD = 'A'
    RIGHT_FRONT = 'B'
    RIGHT = 'C'
    RIGHT_BACK = 'D'
    BACK = 'E'
    LEFT_BACK = 'F'
    LEFT = 'G'
    LEFT_FRONT = 'H'
    ACCELERATE = 'X'
    DECELERATE = 'Y'
    STOP = 'Z'
    START_SPEAK = 'S'
    STOP_SPEAK = 'T'


class SimpleControl:
    def __init__(self, detector: YoloV8Detector):
        self.load_saved_state()
        self.detector: YoloV8Detector = detector
        for k in self.detector.names:
            if self.detector.names[k] == 'person':
                self.person_id = k
                logging.info(f'Person class id: {self.person_id}')
                break
        if self.person_id is None:
            raise ValueError('Person class not found in detector names')
        self.loop_count = 0
    
    def save_current_state(self):
        # 保存 person_id, dst_xyxy, is_following 到 json 文件
        infos = {
            'person_id': self.person_id,
            'dst_xyxy': self.dst_xyxy,
            'is_following': self.is_following
        }
        with open('simple_control.json', 'w') as f:
            # write beautified json string to file
            f.write(json.dumps(infos, indent=4))
    
    def load_saved_state(self) -> bool:
        # 从 json 文件读取 person_id, dst_xyxy, is_following
        try:
            with open('simple_control.json', 'r') as f:
                # infos = eval(f.read())
                # read and parse json string from file
                infos = json.loads(f.read())
                self.person_id = infos['person_id']
                self.dst_xyxy = infos['dst_xyxy']
                self.is_following = infos['is_following']
                logging.info(f'Load saved state: {self.person_id}, {self.dst_xyxy}, {self.is_following}')
        except Exception as e:
            logging.error(f'Load saved state failed: {e}')
            self.person_id = None
            self.dst_xyxy = None
            self.is_following = False
            logging.info('No saved state found')


    def start_follow(self, detect_results):
        not_found_response = config.common.motion_mode_cmd_response_notfound
        found_response = config.common.motion_mode_cmd_response_ok
        if len(detect_results) < 1:
            logging.info('Nothin found, stop follow')
            return False, not_found_response
        self.dst_xyxy  = self.get_first_person_xyxy(detect_results)       
        if self.dst_xyxy is None:
            logging.info('No person found, stop follow')
            return False, not_found_response
        self.is_following = True
        self.save_current_state()
        logging.info(f'Start follow: {self.dst_xyxy}')
        return True,found_response
    
    def get_first_person_xyxy(self, detect_results):
        if len(detect_results) < 1:
            return None
        found_person_index = -1
        for i in range(len(detect_results[0].boxes.cls)):
            if detect_results[0].boxes.cls[i] == self.person_id:
                found_person_index = i
                break
        if found_person_index == -1:
            return None
        data =  detect_results[0].boxes.xyxy[found_person_index]
        # torch.size([4]) 转成float数组
        return [float(x) for x in data]

    
    def stop_follow(self) -> str:
        self.is_following = False
        logging.info('Stop follow')
        return config.common.motion_mode_stop_response
    
    #获取对角线长度
    def get_diagonal_length(self, xyxy):
        x1, y1, x2, y2 = xyxy
        return ((x2-x1)**2 + (y2-y1)**2)**0.5
    
    def get_mid_position(self, xyxy):
        x1, y1, x2, y2 = xyxy
        return (x2+x1)/2, (y2+y1)/2

    async def loop(self, detect_results):
        if not self.is_following or self.dst_xyxy is None:
            return None
        current_xyxy = self.get_first_person_xyxy(detect_results)
        if current_xyxy is None:
            logging.info('No person found, stop move')
            return None
        self.loop_count += 1
        threshold_forward = 0.1 # x dst_diagonal_length
        threshold_stop_follow = 0.2 # x dst_diagonal_length, 停止跟随过小的目标
        threshold_lateral = 30 # px
        # 如果current_xyxy比dst_xyxy小threshold以上，则需要前进；
            # 同时如果不在坐标中心，则需要在前进的同时向左或向右移动；
        # 如果current_xyxy比dst_xyxy大threshold以下，则需要后退；
            # 同时如果不在坐标中心，则需要在后退的同时向左或向右移动；
        current_diagonal_length = self.get_diagonal_length(current_xyxy)
        dst_diagonal_length = self.get_diagonal_length(self.dst_xyxy)
        mid_x, _ = self.get_mid_position(current_xyxy)
        dst_mid_x, _ = self.get_mid_position(self.dst_xyxy)
        dx = mid_x - dst_mid_x

        # // dy > threshold &&  |dx| < threshold, 向前 'A'
        # // dy > threshold &&  dx  > threshold, 向右前 'B'
        # // |dy| < threshold &&  dx  > threshold, 向右 'C'
        # // dy < -threshold &&  dx  > threshold, 向右后 'D'
        # // dy < -threshold &&  |dx| < threshold, 向后 'E'
        # // dy < -threshold &&  dx  < -threshold, 向左后 'F'
        # // |dy| < threshold &&  dx  < -threshold, 向左 'G'
        # // dy > threshold &&  dx  < -threshold, 向左前 'H'
        if current_diagonal_length < dst_diagonal_length * ( 1 - threshold_forward) and current_diagonal_length > threshold_stop_follow * dst_diagonal_length:
            # 前进
            if dx > threshold_lateral:
                # 向右前进
                return Command.RIGHT_FRONT
            elif dx < -threshold_lateral:
                # 向左前进
                return Command.LEFT_FRONT
            else:
                # 前进
                return Command.FORWARD
        elif current_diagonal_length > dst_diagonal_length * (1 + threshold_forward):
            # 后退
            if dx > threshold_lateral:
                # 向左后退
                return Command.LEFT_BACK
            elif dx < -threshold_lateral:
                # 向右后退
                return Command.RIGHT_BACK
            else:
                # 后退
                return Command.BACK
        else:
                return Command.STOP
        
