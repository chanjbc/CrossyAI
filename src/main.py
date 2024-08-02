from collections import deque
import cv2
from itertools import groupby
import logging
from matplotlib import pyplot as plt
import mss
import numpy as np
import os
from pathlib import Path
from PIL import Image
import pyautogui
import random
import time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from win32gui import GetWindowText, GetForegroundWindow



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
gameover_template = torch.tensor(np.load("gameover.npy")).squeeze(0)




# load templates for score recognition
def load_templates():
    templates = []
    for i in range(10):
        template_path = os.path.join("..\\assets\\", f"{i}.png")
        template = cv2.imread(template_path, 0)
        templates.append(template)
    return templates



# capture left split screen using mss
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
        return [screen, np.dot(screen[22:120, 20:700, :3], [0.299, 0.587, 0.114]).astype(np.uint8)]



# preprocess screen, returning both gamescreen and gameover buttons
def preprocess_image(image: np.ndarray) -> torch.Tensor:
    try:
        extract_gamescreen = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(),
            transforms.Resize((84, 84)),
            transforms.Normalize((0.5,), (0.5,))
        ])
        gamescreen = extract_gamescreen(image)
        return gamescreen
    except Exception as e:
        logging.error("Error during image preprocessing: %s", e)
        return None



# preprocess screen, returning both gamescreen and gameover buttons
def save_preprocessed_gameover(image: np.ndarray) -> None:
    try:
        extract_gamescreen = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(),
            transforms.Resize((84, 84)),
            transforms.Normalize((0.5,), (0.5,))
        ])
        gamescreen = extract_gamescreen(image)
        gameover = gamescreen[:, 71:83, 12:74]
        np.save("gameover.npy", gameover)
    except Exception as e:
        logging.error("Error during image preprocessing: %s", e)
        return None



# visualize preprocessed image
def visualize_preprocessed(tensor: torch.Tensor) -> None:
    try:
        # reverses the normalization: x = (x * std) + mean
        tensor = tensor * 0.5 + 0.5
        tensor = tensor.squeeze(0)
        image = transforms.ToPILImage()(tensor)
        image.show()
    except Exception as e:
        logging.error("Error during image visualization: %s", e)



# checks if game is over
def is_gameover(screen: torch.Tensor) -> bool:
    screen = screen[:, 71:83, 12:74]
    screen = screen.squeeze(0)
    err = torch.sum((gameover_template - screen) ** 2).item()
    err /= float(gameover_template.shape[0] * gameover_template.shape[1])
    return err < 0.05



# get game score
def get_score(screen: np.ndarray, templates: list[np.ndarray]) -> int:
    _, binary = cv2.threshold(screen, 1, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(screen.shape, dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, (255), cv2.FILLED)
    mask_inv = cv2.bitwise_not(mask)
    screen = cv2.bitwise_and(screen, mask)
    wh_mask = np.where(screen < 255)
    screen[wh_mask] = 0

    vertical_projection = np.sum(screen, axis=0)

    gaps = np.where(vertical_projection > 0)[0]

    islands = []
    for k, g in groupby(enumerate(gaps), lambda ix: ix[0] - ix[1]):
        island = list(g)
        islands.append((island[0][1], island[-1][1] - island[0][1] + 1))


    # Segment and save each digit
    digits = []
    for i, (start, width) in enumerate(islands):
        digits.append(screen[8:88, start:start + width - 1])

    score = ""
    for digit in digits:
        for i, template in enumerate(templates):
            if digit.shape == template.shape and np.array_equal(digit, template):
                score += str(i)

    if len(score) == 0:
        return 0
    else:
        return int(score)



# take actions with pyautogui
def take_action(action) -> None:
    global curr_score
    if action == 0:
        pyautogui.press("up")
        curr_score += 1
    elif action == 1:
        pyautogui.press("down")
        curr_score -= 1
    elif action == 2:
        pyautogui.press("left")
    elif action == 3:
        pyautogui.press("right")
    elif action == 4:
        time.sleep(0.1)
    # used only for resetting game
    elif action == 5: 
        pyautogui.press("space")
    else:
        logging.warning("Invalid action: %s", action)



class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)

        # self.conv1 = nn.Conv2d(state_size[2], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x




class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)



class DQNAgent:
    def __init__(self, state_size, action_size, batch_size, gamma, epsilon, epsilon_min, epsilon_decay, learning_rate, memory_capacity, target_update_freq):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.memory = ReplayMemory(memory_capacity)
        self.policy_net = QNetwork(state_size, action_size).to(device)
        self.target_net = QNetwork(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.target_update_freq = target_update_freq
        self.steps_done = 0
    
    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.epsilon_min + (self.epsilon - self.epsilon_min) * np.exp(-1. * self.steps_done / self.epsilon_decay)
        # print(self.epsilon_min, self.epsilon, self.epsilon_decay, self.steps_done)
        # print(eps_threshold)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_size)]], device=device, dtype=torch.long)
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))
        
        state_batch = torch.cat(batch[0]).unsqueeze(1)
        action_batch = torch.cat(batch[1])
        reward_batch = torch.cat(batch[2])
        next_state_batch = torch.cat(batch[3]).unsqueeze(1)
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






# with mss.mss() as sct:
#     screen = np.array(sct.grab({
#         "left": 22,
#         "top": 62,
#         "width": 800,
#         "height": 179
#     }))
#     screen = screen[..., :3]
#     image = Image.fromarray(screen)
#     image.show()

# configure logging + pytorch device



# hyperparameters
state_size = (1, 84, 84)  # (channels, height, width)
action_size = 5
batch_size = 32
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 100
learning_rate = 0.005
memory_capacity = 10000    
target_update_freq = 10 

agent = DQNAgent(state_size, action_size, batch_size, gamma, epsilon, epsilon_min, epsilon_decay, learning_rate, memory_capacity, target_update_freq)

num_episodes = 1000

actions = ["up", "down", "left", "right", "wait"]

templates = load_templates()

for episode in range(num_episodes):
    print(f"========== EPISODE {episode + 1}/{num_episodes} ==========")
    time.sleep(2.5)
    take_action(5)
    time.sleep(2.5)
    take_action(5)

    screen, _ = capture_screen()
    curr_score, last_score = 0, 0

    if screen is None:
        logging.error("Skipping episode due to capture error")
        continue

    state = preprocess_image(screen)
    total_reward = 0

    for t in range(10000):
        t0 = time.perf_counter()
        action = agent.select_action(state)
        take_action(action.item())

        next_screen, next_score_region = capture_screen()
        if next_screen is None:
            logging.error("Skipping step due to capture error")
            break

        curr_score = get_score(next_score_region, templates)
        reward = int(curr_score > last_score)
        last_score = curr_score

        next_state = preprocess_image(next_screen)
        done = is_gameover(next_state) or curr_score == 0

        if done:
            reward = -10 
            agent.memory.push(state, action, torch.tensor([[reward]], device=device), next_state, torch.tensor([[done]], device=device, dtype=torch.uint8))
            break
        print(f"Action: {actions[action.item()]}\tReward: {reward}\tCurrent Score: {curr_score}\t", end="")

        agent.memory.push(state, action, torch.tensor([[reward]], device=device), next_state, torch.tensor([[done]], device=device, dtype=torch.uint8))
        state = next_state
        total_reward += reward
        agent.optimize_model()

        if t % agent.target_update_freq == 0:
            agent.update_target_network()

        tf = time.perf_counter()
        print(f"Time: {tf-t0}")

    logging.info(f"Episode: {episode}, Total Reward: {total_reward}\n")

logging.info("Training complete!")


