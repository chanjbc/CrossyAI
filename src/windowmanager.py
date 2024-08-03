from dotenv import load_dotenv
import logging
import mss
import numpy as np
import os
import pyautogui
import psutil
import subprocess
import win32api
import win32gui
import win32con
import time



def is_game_running(game_exe: str) -> bool:
    for proc in psutil.process_iter():
        try:
            if proc.name() == game_exe:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False
def start_game(game_exe: str, game_path: str) -> None:
    if not is_game_running(game_exe):
        subprocess.Popen(game_path)



def is_game_front(game_name: str) -> bool:
    return win32gui.GetWindowText(win32gui.GetForegroundWindow()) == game_name
def focus_game_window(game_name: str, hwnd: int) -> None:
    if not is_game_front(game_name):
        pyautogui.press("alt")
        win32gui.SetForegroundWindow(hwnd)



def is_game_left(hwnd: int) -> bool:
    dims = win32gui.GetWindowRect(hwnd)
    return dims == (-8, 0, 1288, 1388)
def move_game_left(hwnd: int) -> bool:
    if not is_game_left(hwnd):
        win32gui.MoveWindow(hwnd, -8, 0, 1296, 1388, True)




# load_dotenv()
# game_path = os.getenv("GAME_PATH")
# game_name = game_path.split('\\')[-1].split('.')[0]
# game_exe = game_path.split('\\')[-1]
# hwnd = win32gui.FindWindow(None, game_name)
# is_game_left(hwnd)
# win32gui.MoveWindow(hwnd, -8, 0, 1296, 1388, True)
# is_game_left(hwnd)
# screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
# screen_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
# print(screen_width, screen_height)




def window_handler(game_path: str) -> None:
    game_name = game_path.split('\\')[-1].split(".")[0]
    game_exe = game_path.split('\\')[-1]

    while True:
        try:
            start_game(game_exe, game_path)
            hwnd = win32gui.FindWindow(None, game_name)
            focus_game_window(game_name, hwnd)
            move_game_left(hwnd)
            with mss.mss() as sct:
                corner = np.array(sct.grab({
                    "left": 2,
                    "top": 2,
                    "width": 1,
                    "height": 1
                }))
                corner = corner[..., :3]
                if np.all(corner == 255):
                    print("Top-left corner is white")
                    win32gui.MoveWindow(hwnd, -7, 0, 1296, 1388, True)
                    continue
            if is_game_running(game_exe) and is_game_front(game_name) and is_game_left(hwnd):
                return
        except Exception as e:
            logging.error("Error handling window: %s", e)




    # while True:
    #     try:
    #         if not is_game_running(game_exe):
    #             logging.info("Game not running. Starting game")
    #             subprocess.Popen(game_path)
    #         if win32gui.GetWindowText(win32gui.GetForegroundWindow()) != game_name:
    #             logging.info("Game not in focus. Moving to front and placing in left half")
    #             hwnd = win32gui.FindWindow(None, game_name)
    #             if hwnd:
    #                 bring_game_to_front(hwnd)
    #         while True:
    #             move_window_to_left(hwnd, screen_width, screen_height)
    #             with mss.mss() as sct:
    #                 corner = np.array(sct.grab({
    #                     "left": 2,
    #                     "top": 2,
    #                     "width": 1,
    #                     "height": 1
    #                 }))
    #                 corner = corner[..., :3]
    #             if np.all(corner != 255):
    #                 break
    #         # return True
    #     except Exception as e:
    #         logging.error("Error handling window: %s", e)
    #         return False
    #     # time.sleep(1)







if __name__ == "__main__":
    load_dotenv()
    game_path = os.getenv("GAME_PATH")
    window_handler(game_path)
