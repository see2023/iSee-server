import os
import io
from ultralytics import YOLO
from livekit import rtc
from PIL import Image
import logging
from common.config import config


class YoloV8Detector:
    def __init__(self):
        # model: yolov8n.pt  yolov8s.pt  yolov8m.pt  yolov8l.pt  yolov8x.pt
        self.model = YOLO(config.agents.yolo_model, task='detect', verbose=config.agents.yolo_verbose)
        self.device = config.agents.yolo_device
        self.names = self.model.names

    @classmethod
    def VidioFrame_to_Image(cls, frame: rtc.VideoFrame) -> Image:
        try:
            argb_frame = rtc.ArgbFrame.create(
                format=rtc.VideoFormatType.FORMAT_RGBA,
                width=frame.buffer.width,
                height=frame.buffer.height,
            )
            frame.buffer.to_argb(dst=argb_frame)
            image = Image.frombytes(
                "RGBA", (argb_frame.width, argb_frame.height), argb_frame.data.tobytes()
            ).convert("RGB")
            return image
        except Exception as e:
            logging.error('YoloV8Detector.VidioFrame_to_Image error: %s', e)
            return None
    
    @classmethod
    def Image_to_IObytes(cls, image: Image) -> io.BytesIO:
        output_stream = io.BytesIO()
        image.save(output_stream, format="JPEG")
        output_stream.seek(0)
        return output_stream


    def detect(self, image: Image):
        results = self.model(image, device=self.device)
        logging.debug('yolov8 detect result: %d', len(results))
        return results
