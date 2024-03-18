from stockfish import Stockfish
import cv2
import pyautogui as pg
import numpy as np
import sys
from pynput import keyboard


# constants (modify if needed)
BOARD_SIZE = 400
CELL_SIZE = int(BOARD_SIZE / 8)
BOARD_TOP_COORD = 141
BOARD_LEFT_COORD = 5
CONFIDENCE = 0.77
DETECTION_NOICE_THRESHOLD = 8
PIECES_PATH = './piece_recognition/pieces/'

# players
WHITE = 0
BLACK = 1

# side to move
side_to_move = 0

# square to coords
square_to_coords = []

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

# array to convert board square indices to coordinates (black)
get_square = [
    'a8', 'b8', 'c8', 'd8', 'e8', 'f8', 'g8', 'h8',
    'a7', 'b7', 'c7', 'd7', 'e7', 'f7', 'g7', 'h7',
    'a6', 'b6', 'c6', 'd6', 'e6', 'f6', 'g6', 'h6',
    'a5', 'b5', 'c5', 'd5', 'e5', 'f5', 'g5', 'h5',
    'a4', 'b4', 'c4', 'd4', 'e4', 'f4', 'g4', 'h4',
    'a3', 'b3', 'c3', 'd3', 'e3', 'f3', 'g3', 'h3',
    'a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2',
    'a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1'
];
  
# map piece names to FEN chars
piece_names = {
    'black_king': 'k',
    'black_queen': 'q',
    'black_rook': 'r',
    'black_bishop': 'b',
    'black_knight': 'n',
    'black_pawn': 'p',
    'white_knight': 'N',
    'white_pawn': 'P',
    'white_king': 'K',
    'white_queen': 'Q',
    'white_rook': 'R',
    'white_bishop': 'B'
}


# get coordinates of chess pieces
def recognize_position():
    # piece locations
    piece_locations = {
        'black_king': [],
        'black_queen': [],
        'black_rook': [],
        'black_bishop': [],
        'black_knight': [],
        'black_pawn': [],
        'white_knight': [],
        'white_pawn': [],
        'white_king': [],
        'white_queen': [],
        'white_rook': [],
        'white_bishop': []
    }

    # take a board snapshot
    # screenshot = cv2.cvtColor(np.array(pg.screenshot()), cv2.COLOR_RGB2BGR)
    
    # loop over piece names
    for piece in piece_names.keys():
        # store piece locations
        for location in pg.locateAllOnScreen(PIECES_PATH + piece + '.png', confidence=CONFIDENCE):
            # false detection flag
            noise = False
            
            # loop over matched pieces
            for position in piece_locations[piece]:
                # noice detection
                if abs(position.left - location.left) < DETECTION_NOICE_THRESHOLD and \
                   abs(position.top - location.top) < DETECTION_NOICE_THRESHOLD:
                    noise = True
                    break
            
            # skip noice detections
            if noise: continue
            
            # detect piece
            piece_locations[piece].append(location)
            print('detecting:', piece, location)
            
    # return piece locations 
    return piece_locations

# convert piece coordinates to FEN string
def locations_to_fen(piece_locations):
    # FEN string
    fen = ''
    
    # board top left corner coords
    x = BOARD_LEFT_COORD
    y = BOARD_TOP_COORD
    
    # loop over board rows
    for row in range(8):
        # empty square counter
        empty = 0
            
        # loop over board columns
        for col in range(8):
            # init square
            square = row * 8 + col
            
            # piece detection
            is_piece = ()
            
            # loop over piece types
            for piece_type in piece_locations.keys():
                # loop over pieces
                for piece in piece_locations[piece_type]:
                    if abs(piece.left - x) < DETECTION_NOICE_THRESHOLD and \
                       abs(piece.top - y) < DETECTION_NOICE_THRESHOLD:
                        if empty:
                            fen += str(empty)
                            empty = 0

                        fen += piece_names[piece_type]
                        is_piece = (square, piece_names[piece_type])

            
            if not len(is_piece):
                empty += 1
            
            # increment x coord by cell size
            x += CELL_SIZE
        
        if empty: fen += str(empty)
        if row < 7: fen += '/'
        
        # restore x coord, increment y coordinate by cell size
        x = BOARD_LEFT_COORD
        y += CELL_SIZE
        
    if side_to_move:
        # fen = fen.split('/')
        # fen.reverse()
        fen = f'{fen[::-1]} b'
    else:
        fen = f'{fen} w'
    
    # add placeholders (NO EN PASSANT AND CASTLING are static placeholders)
    fen += ' KQkq - 0 1'
    
    # return FEN string
    return fen


def mkmove(bm):
    global square_to_coords, get_square
    
    # extract source and destination square coordinates
    if side_to_move:
        a = alphabet[7 - alphabet.index(bm[0])]
        b = alphabet[7 - alphabet.index(bm[2])]
        print(a + b)
        
        from_sq = square_to_coords[get_square.index(a + str(9 - int(bm[1])))]
        print(from_sq)
        to_sq = square_to_coords[get_square.index(b + str(9 - int(bm[3])))]
    else:
        from_sq = square_to_coords[get_square.index(bm[0] + bm[1])]
        to_sq = square_to_coords[get_square.index(bm[2] + bm[3])]

    # make move on board
    pg.moveTo(from_sq)
    pg.click()
    pg.moveTo(to_sq)
    pg.click()
        
    
def bvisual():
    b = stockfish.get_board_visual()
    print(b)


def fstfen(complete_fen):
    try:
        complete_fen = complete_fen.split(' ')
        return complete_fen[0]
    except:
        return None


def on_release(key):
    if str(key) == "'p'":
        print('playnow')
        playnow()

    elif key == keyboard.Key.esc:
        # Stop listener
        return False

def playnow():
    global prvfen
    try:
        piece_locations = recognize_position()
        fen = locations_to_fen(piece_locations)
        print(fen)
        if fstfen(fen) != fstfen(prvfen):
            fenv = stockfish.is_fen_valid(fen)
            if fenv:
                stockfish.set_fen_position(fen)
                bvisual()

                bmove = stockfish.get_best_move()
                print(f'the best move is: {bmove}')

                mkmove(bmove)

                stockfish.make_moves_from_current_position([bmove])
                prvfen = stockfish.get_fen_position()
                bvisual()
            else:
                print('fen not valid')

    except Exception as e:
        print(str(e))


if __name__ == '__main__':
    try:
        if sys.argv[1] == 'black':
            side_to_move = BLACK
    except:
        print('usage: "playable.py white" or "playable.py black"')
        sys.exit(0)

    stockfish = Stockfish(path="stockfish_15.1_win_x64_popcnt\\stockfish-windows-2022-x86-64-modern.exe", depth=18, parameters={"Threads": 2, "Minimum Thinking Time": 30, "Hash": 1024})
    print('engine created')

    # board top left corner coords
    x = BOARD_LEFT_COORD
    y = BOARD_TOP_COORD

    # loop over board rows
    for row in range(8):
        # loop over board columns
        for col in range(8):
            # init square
            square = row * 8 + col
            
            # associate square with square center coordinates
            square_to_coords.append((int(x + CELL_SIZE / 2), int(y + CELL_SIZE / 2)))

            # increment x coord by cell size
            x += CELL_SIZE
        
        # restore x coord, increment y coordinate by cell size
        x = BOARD_LEFT_COORD
        y += CELL_SIZE

    prvfen = None

    with keyboard.Listener(
            on_release=on_release) as listener:
        listener.join()

