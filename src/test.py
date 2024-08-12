from collections import deque
import cv2
from dotenv import load_dotenv
from itertools import groupby
import logging
from matplotlib import pyplot as plt
import mss
import numpy as np
import os
from pathlib import Path
from PIL import Image
import pyautogui
import psutil
import random
import subprocess
import time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import win32gui






device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
gameover_template = torch.load("gameover.pt")



# MARK: load_templates
def load_templates():
    templates = []
    for i in range(10):
        template_path = os.path.join("..\\assets\\", f"{i}.png")
        template = cv2.imread(template_path, 0)
        templates.append(template)
    return templates



# MARK: capture_screen
def capture_screen() -> list[np.ndarray]:
    with mss.mss() as sct:
        # width of borders: 2px
        # height of Windows bar: 40px
        # height of taskbar: 60px 
        screen = np.array(sct.grab({
            "left": 2,
            "top": 40,
            "width": 1276,
            "height": 1340
        }))
        screen = screen[..., :3]
        return [screen[340:1341, 138:1139], 
                np.dot(screen[22:120, 20:700, :], [0.299, 0.587, 0.114]).astype(np.uint8)]



# MARK: preprocessing
# preprocess screen, returning both gamescreen and gameover buttons
def preprocess_image(image: np.ndarray) -> torch.Tensor:
    try:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(grayscale, (84, 84))
        normalized = resized / 255.0
        return torch.tensor(normalized)
    except Exception as e:
        logging.error("Error during image preprocessing: %s", e)



# MARK: save_preprocessed_gameover
def save_preprocessed_gameover(image: np.ndarray) -> None:
    try:
        gameover = image[67:82, 4:82]
        cv2.imshow("Gameover Region", gameover)
        cv2.waitKey(0)
        torch.save(torch.tensor(gameover), "gameover.pt")
    except Exception as e:
        logging.error("Error during saving image: %s", e)
# save_preprocessed_gameover(preprocess_image(capture_screen()[0]))




# visualize preprocessed image
def visualize_preprocessed_tensor(tensor: torch.Tensor) -> None:
    try:
        # reverses the normalization: x = (x * std) + mean
        tensor = tensor * 0.5 + 0.5
        tensor = tensor.squeeze(0)
        image = transforms.ToPILImage()(tensor)
        image.show()
    except Exception as e:
        logging.error("Error during image visualization: %s", e)




# MARK: is_gameover
def is_gameover(screen: torch.Tensor) -> bool:
    screen = screen[67:82, 4:82]
    err = torch.sum((gameover_template - screen) ** 2).item()
    err /= float(gameover_template.shape[0] * gameover_template.shape[1])
    return err < 0.01



# MARK: get_score
def get_score(screen: np.ndarray, templates: list[np.ndarray]) -> int:
    _, binary = cv2.threshold(screen, 1, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(screen.shape, dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, (255), cv2.FILLED)
    mask_inv = cv2.bitwise_not(mask)
    screen = cv2.bitwise_and(screen, mask)
    wh_mask = np.where(screen < 255)
    screen[wh_mask] = 0

    # identify islands
    vertical_projection = np.sum(screen, axis=0)
    gaps = np.where(vertical_projection > 0)[0]
    islands = []
    for k, g in groupby(enumerate(gaps), lambda ix: ix[0] - ix[1]):
        island = list(g)
        islands.append((island[0][1], island[-1][1] - island[0][1] + 1))

    # segment digits and store in array
    digits = []
    for i, (start, width) in enumerate(islands):
        digits.append(screen[8:88, start:start + width - 1])

    # iterate through 
    score = ""
    for digit in digits:
        for i, template in enumerate(templates):
            if digit.shape == template.shape and np.array_equal(digit, template):
                score += str(i)

    if len(score) == 0:
        return -1
    else:
        return int(score)




# MARK: take_action
def take_action(action) -> None:
    if action == 0:
        pyautogui.press("up")
    elif action == 1:
        pyautogui.press("down")
    elif action == 2:
        pyautogui.press("left")
    elif action == 3:
        pyautogui.press("right")
    elif action == 4:
        time.sleep(0.2)
    # action == 5 used only for resetting game
    elif action == 5: 
        pyautogui.press("space")
    else:
        logging.warning("Invalid action: %s", action)


















# MARK: reset_game
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

def reset_game(game_path: str) -> None:
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
                    pyautogui.hotkey("win", "left")
                    continue
            if is_game_running(game_exe) and is_game_front(game_name) and is_game_left(hwnd):
                return
        except Exception as e:
            logging.error("Error handling window: %s", e)











# MARK: QNetwork
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, num_frames=4):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(num_frames, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    



# MARK: ReplayMemory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
   
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
   
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
   
    def __len__(self):
        return len(self.memory)





    
# MARK: DQNAgent
class DQNAgent:
    def __init__(self, state_size, action_size, batch_size, gamma, epsilon, epsilon_min, epsilon_decay, learning_rate, memory_capacity, target_update_freq, num_frames=4):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.eps_threshold = 1
        self.memory = ReplayMemory(memory_capacity)
        self.policy_net = QNetwork(state_size, action_size, num_frames).to(device)
        self.target_net = QNetwork(state_size, action_size, num_frames).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.target_update_freq = target_update_freq
        self.steps_done = 0
        self.num_frames = num_frames
   
    def select_action(self, state):
        sample = random.random()
        self.eps_threshold = self.epsilon_min + (self.epsilon - self.epsilon_min) * np.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if sample > self.eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_size)]], device=device, dtype=torch.long)
   
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))
       
        state_batch = torch.cat(batch[0])
        action_batch = torch.cat(batch[1])
        reward_batch = torch.cat(batch[2])
        next_state_batch = torch.cat(batch[3])
        done_batch = torch.cat(batch[4])
        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
       
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma * (1 - done_batch)) + reward_batch
       
        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
       
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
   
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())






# Frame stacking function
def stack_frames(stacked_frames: deque, frame: torch.tensor, num_frames=4):
    stacked_frames.append(frame)
    return torch.stack(list(stacked_frames), dim=0).unsqueeze(0)







templates = load_templates()
load_dotenv()
GAME_PATH = os.getenv("GAME_PATH")








# MARK: hyperparameters
state_size = (84, 84)  # (channels, height, width)
num_frames = 4

actions = ["up", "down", "left", "right", "wait", "start"]
action_size = 5

gamma = 0.99
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 1_000
learning_rate = 0.06
memory_capacity = 10_000
target_update_freq = 10 

num_episodes = 10_000
batch_size = 32

best_score, best_ep = 0, 0
agent = DQNAgent(state_size=state_size, action_size=action_size, batch_size=batch_size, gamma=gamma, epsilon=epsilon, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay, learning_rate=learning_rate, memory_capacity=memory_capacity, target_update_freq=target_update_freq, num_frames=num_frames)



# MARK: training loop
reset_game(GAME_PATH)
for episode in range(num_episodes):
    print(f"========== EPISODE {episode + 1}/{num_episodes} ==========")

    # capture starting screen
    screen, _ = capture_screen()
    last_score = 0
    total_reward = 0
    if screen is None:
        logging.error("Skipping episode due to capture error")
        continue
    
    # initialize frame stack
    stacked_frames = deque([preprocess_image(screen)]*num_frames, maxlen=num_frames)
    state = stack_frames(stacked_frames, preprocess_image(screen))
    

    for t in range(10_000):
        t0 = time.perf_counter()

        # choose action
        if t == 0:
            time.sleep(2.5)
            action = torch.tensor([[0]], device=device, dtype=torch.long)
        else:
            action = agent.select_action(state)
        take_action(action.item())

        # capture next state/score/gameover region
        next_screen, next_score_region = capture_screen()
        if next_screen is None:
            logging.error("Skipping step due to capture error")
            continue
        curr_score = get_score(next_score_region, templates)
        next_state = stack_frames(stacked_frames, preprocess_image(next_screen))
        done = is_gameover(next_state[-1][-1])

        # track best score
        if curr_score > best_score:
            best_score, best_ep = curr_score, episode

        # detect crash by negative score and not gameover event
        if curr_score < 0 and not done:
            time.sleep(2)
            done = is_gameover(preprocess_image(capture_screen()[0]))
            if not done:
                logging.info("Crash detected. Resetting game...")
                reset_game(GAME_PATH)
                break
        
        # assign reward 
        if done:
            reward = -10
        else:
            reward = int(curr_score > last_score)
            last_score = curr_score
        total_reward += reward
        
        # train model
        agent.memory.push(state, action, torch.tensor([[reward]], device=device), next_state, torch.tensor([[done]], device=device, dtype=torch.uint8))
        agent.optimize_model()
        if t % agent.target_update_freq == 0:
            agent.update_target_network()
        state = next_state
        
        # print stats
        tf = time.perf_counter()
        print(f"Action: {actions[action.item()]}\tCurrent Score: {curr_score}\tReward: {reward}\tEps Threshold: {agent.eps_threshold:.3f}\tTime: {tf-t0:.4f} s")
        
        # reset if gameover
        if done:
            take_action(5)
            break
    
    logging.info(f"Episode: {episode}, Total Reward: {total_reward}\nBest Run: {best_score} on episode {best_ep}\n")

logging.info("Training Complete!")

if __name__ == "main":
    main()