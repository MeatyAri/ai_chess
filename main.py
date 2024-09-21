from stockfish import Stockfish
from pynput import mouse, keyboard
from mss import mss
import time
import cv2
import sys
import numpy as np
from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
import onnxruntime as ort
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


class FPSCounter:
    def __init__(self):
        self.frames = 0
        self.start_time = time.time()
        self.prev_fps = 0.0

    def update(self):
        self.frames += 1

    def get_fps(self):
        elapsed_time = time.time() - self.start_time
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
        self.start_time = time.time()
    
    def add_fps_overlay(self, frame):
        fps = self.get_fps()
        overlay_text = f"FPS: {fps}"
        cv2.putText(frame, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame


class KBControls():
    def __init__(self) -> None:
        self.activate = False

        self.listener = keyboard.Listener(on_press=self.on_press_kb)
        self.listener.start()
    
    def on_press_kb(self, key):
        try:
            if key.char == "p":
                self.activate = True
        except:
            pass


class MouseControls():
    def __init__(self):
        self.ms = mouse.Controller()
        self.letters = "abcdefgh"
    
    def best_move_to_cell_index(self, best_move):
        best_move = list(best_move)
        prev_cell = best_move[:2]
        new_cell = best_move[2:4]
        prev_cell[0] = self.letters.index(prev_cell[0])
        prev_cell[1] = 7 - (int(prev_cell[1]) - 1)
        new_cell[0] = self.letters.index(new_cell[0])
        new_cell[1] = 7 - (int(new_cell[1]) - 1)

        if args.side == "b":
            prev_cell = [7 - x for x in prev_cell]
            new_cell = [7 - x for x in new_cell]

        return prev_cell, new_cell

    def cell_to_pixel(self, cell, board_region):
        xs, ys, xe, ye = board_region
        cell_width = (xe - xs) // 8
        cell_height = (ye - ys) // 8
        x_cell = xs + cell[0] * cell_width + cell_width // 2
        y_cell = ys + cell[1] * cell_height + cell_height // 2

        return (x_cell, y_cell)

    def drag_mouse(self, xy_prev, xy_new):
        self.ms.position = xy_prev
        self.ms.press(mouse.Button.left)
        self.ms.position = xy_new
        self.ms.release(mouse.Button.left)
    
    def click_mouse(self, xy_prev, xy_new):
        self.ms.position = xy_prev
        self.ms.click(mouse.Button.left)
        self.ms.position = xy_new
        self.ms.click(mouse.Button.left)

    def move(self, best_move, board_region):
        prev_cell, new_cell = self.best_move_to_cell_index(best_move)
        xy_prev = self.cell_to_pixel(prev_cell, board_region)
        xy_new = self.cell_to_pixel(new_cell, board_region)

        if args.click_cells:
            self.click_mouse(xy_prev, xy_new)
        else:
            self.drag_mouse(xy_prev, xy_new)
        self.ms.position = (0, 0)


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
        return self.fen_transform_instructions(img)


def screen_capture(bounding_box = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}):
    sct_img = sct.grab(bounding_box)

    frame = np.array(sct_img)[:, :, :3] # last part => RGBA2RGB

    return frame


def visualize(results):
    annotated_frame = results[0].plot()

    # count the fps & display it
    annotated_frame = fps_counter.add_fps_overlay(annotated_frame)

    cv2.imshow('screen', annotated_frame)

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        sys.exit()


def detect_board(frame):
    # (1080, 1920, 3) => (1, 3, 640, 640)
    processed_frame = img_processor.yolo_transform(frame).to(device)
    results = yolo_model(processed_frame, conf=0.8, verbose=False, imgsz=640)
    xywhn = results[0].boxes.xywhn

    visualize(results)

    if len(xywhn) > 0:
        xywhn = xywhn[0]
        height, width, C = frame.shape
        x_pixel = xywhn[0] * width
        y_pixel = xywhn[1] * height
        w_pixel = xywhn[2] * width
        h_pixel = xywhn[3] * height
        xs = x_pixel - w_pixel // 2
        ys = y_pixel - h_pixel // 2
        xe = x_pixel + w_pixel // 2
        ye = y_pixel + h_pixel // 2

        xs = xs.to(torch.long)
        ys = ys.to(torch.long)
        xe = xe.to(torch.long)
        ye = ye.to(torch.long)

        return (xs, ys, xe, ye)


def crop_to_board(frame, board_region):
    # (1080, 1920, 3) => (board_size, board_size, 3)
    return frame[board_region[1]:board_region[3], board_region[0]:board_region[2]]


def crop_image_into_squares(image_array, num_squares_per_side=8):
    C, height, width = image_array.shape
    square_size = min(height, width) // num_squares_per_side

    cropped_squares = image_array.reshape(C, num_squares_per_side, square_size,
                                          num_squares_per_side, square_size)
    cropped_squares = cropped_squares.transpose(1, 3, 0, 2, 4)
    cropped_squares = cropped_squares.reshape(-1, C, square_size, square_size)

    return cropped_squares


def fen_from_onehot(one_hot):
    piece_symbols = 'prbnkqPRBNKQ'
    output = ''
    for j in range(8):
        for i in range(8):
            if(one_hot[j][i] == 12):
                output += ' '
            else:
                output += piece_symbols[one_hot[j][i]]
        if(j != 7):
            output += '/'

    for i in range(8, 0, -1):
        output = output.replace(' ' * i, str(i))
    
    if args.side == "b":
        output = f'{output[::-1]} b'
    else:
        output = f'{output} w'
    
    output += ' KQkq - 0 1'

    return output


def board_img_to_fen(board_image):
    board_image = img_processor.fen_transform(board_image).numpy()
    board_image = crop_image_into_squares(board_image)

    outputs = fen_model_ort.run(None, {'arg0': board_image})
    predicted = np.argmax(outputs[0], 1)
    # _, predicted = torch.max(torch.tensor(outputs[0]), 1)
    
    return fen_from_onehot(predicted.reshape(8, 8))


def get_best_move(fen):
    if stockfish.is_fen_valid(fen) or stockfish.is_fen_valid(fen := fen.replace("KQkq", "-")):
        stockfish.set_fen_position(fen)
        return stockfish.get_best_move()
    else:
        print("fen is not valid")


def main():
    while True:
        # take a screenshot
        frame = screen_capture()
        fps_counter.update()

        # detect the board
        board_region = detect_board(frame)

        # if detection is not none
        #   then crop the original frame and get the fen
        if board_region is None or not kb.activate:
            continue

        board_image = crop_to_board(frame, board_region)
        fen = board_img_to_fen(board_image)

        best_move = get_best_move(fen)
        if best_move is not None:
            print(best_move)
            mv_mouse.move(best_move, board_region)

        kb.activate = False


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("side", type=str, help="choose what side to play, b for black or w for white")

    parser.add_argument("-c", "--click-cells", action="store_true", help="clicks rather than dragging the cells for compatibility")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    fps_counter = FPSCounter()

    stockfish = Stockfish(path="./stockfish/stockfish-ubuntu-x86-64-avx2", depth=20, parameters={"Threads": 8, "Hash": 4096})

    yolo_model = YOLO('./yolo_models/yolov8n_chess_board.engine', task='detect')

    fen_model_ort = ort.InferenceSession("./fen_models/best/100_epochs/fen_gen.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    sct = mss()

    img_processor = ImageProcessing()

    mv_mouse = MouseControls()

    kb = KBControls()

    main()
