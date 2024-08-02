from dotenv import load_dotenv
import logging
import mss
import os
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



def bring_game_to_front(hwnd: int) -> None:
    try:
        win32gui.SetForegroundWindow(hwnd)
    except Exception as e:
        logging.info("Failed to bring game to front")



def move_window_to_left(hwnd: int, screen_width: int, screen_height: int) -> None:
    win32gui.MoveWindow(hwnd, -7, 0, screen_width//2 + 14, screen_height - 40, True)



def window_handler(game_path: str):
    game_name = game_path.split('\\')[-1].split('.')[0]
    game_exe = game_path.split('\\')[-1]
    screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
    screen_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
    while True:
        if not is_game_running(game_exe):
            logging.info("Game not running. Starting game")
            subprocess.Popen(game_path)
            time.sleep(2)
        foreground_hwnd = win32gui.GetForegroundWindow()
        foreground_window_title = win32gui.GetWindowText(foreground_hwnd)
        if foreground_window_title != game_name:
            logging.info("Game not in focus. Moving to front and placing in left half")
            hwnd = win32gui.FindWindow(None, game_name)
            if hwnd:
                bring_game_to_front(hwnd)
                move_window_to_left(hwnd, screen_width, screen_height)
            else:
                window_handler(game_path)
        time.sleep(1)  # Check every second



if __name__ == "__main__":
    load_dotenv()
    game_path = os.getenv("GAME_PATH")
    window_handler(game_path)
