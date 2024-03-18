from stockfish import Stockfish
from pynput.mouse import Controller
from mss import mss
from time import time, sleep
import cv2
import sys
import numpy as np
from ultralytics import YOLO
import torch
import torchvision.transforms as transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


class FPSCounter:
    def __init__(self):
        self.frames = 0
        self.start_time = time()
        self.prev_fps = 0.0

    def update(self):
        self.frames += 1

    def get_fps(self):
        elapsed_time = time() - self.start_time
        if elapsed_time >= 1.0:
            fps = int(self.frames / elapsed_time)
            self.prev_fps = fps
            print(fps)
            self.reset()
        else:
            fps = self.prev_fps
        return fps

    def reset(self):
        self.frames = 0
        self.start_time = time()
    
    def add_fps_overlay(self, frame):
        fps = self.get_fps()
        overlay_text = f"FPS: {fps}"
        cv2.putText(frame, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame


class ImageProcessing():
    def __init__(self) -> None:
        self.yolo_transform_instructions = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((640, 640)),
            transforms.ConvertImageDtype(torch.float),
        ])
        self.fen_transform_instructions = transforms.Compose([ 
            transforms.ToTensor(),
            transforms.Resize((200, 200)),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    
    def yolo_transform(self, img):
        return self.yolo_transform_instructions(img).unsqueeze(0)

    def fen_transform(self, img):
        return self.fen_transform(img)


def screen_capture(bounding_box = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}):
    sct_img = sct.grab(bounding_box)

    frame = np.array(sct_img)[:, :, :3] # last part => RGBA2RGB

    return frame


def detect_board(frame):
    # (1080, 1920, 3) => (1, 3, 640, 640)
    processed_frame = img_processor.yolo_transform(frame).to(device)
    results = yolo_model(processed_frame, conf=0.8, verbose=False, imgsz=640)
    xywhn = results[0].boxes.xywhn

    if len(xywhn) > 0:
        height, width, C = frame.shape
        x_pixel = xywhn[0] * width
        y_pixel = xywhn[1] * height
        w_pixel = xywhn[2] * width
        h_pixel = xywhn[3] * height
        xs = x_pixel - w_pixel // 2
        ys = y_pixel - h_pixel // 2
        xe = x_pixel + w_pixel // 2
        ye = y_pixel + h_pixel // 2

        return (xs, ys, xe, ye)


def main():
    while True:
        # take a screenshot
        frame = screen_capture()

        # detect the board
        board_region = detect_board(frame)

        # if detection is not none
        #   then crop the original frame and get the fen

        fps_counter.update()
        # fps_limiter.cap()


if __name__ == '__main__':
    # fps_limiter = FPSLimiter(target_fps=60)
    fps_counter = FPSCounter()

    stockfish = Stockfish(path="./stockfish/stockfish-ubuntu-x86-64-avx2", depth=18, parameters={"Threads": 4, "Hash": 4096})

    yolo_model = YOLO('./yolo_models/yolov8n_chess_board.engine', task='detect')

    # fen_model = 

    sct = mss()

    img_processor = ImageProcessing()

    # mv_mouse = MouseControls()

    main()
