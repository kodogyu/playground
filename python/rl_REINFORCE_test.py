# source code mainly from https://github.com/seungeunrho/RLfrombasics/

import gymnasium as gym
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x

def display_status(n_epi, score, a):
    image = np.zeros((100, 300))

    epi_text = "n_epi: {}".format(n_epi)
    score_text = "score: {}".format(score)
    action_text = "action: {}".format(a)

    epi_org = (0, 30)
    score_org = (0, 60)
    action_org = (0, 90)
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0)

    text_image = cv2.putText(image, epi_text, epi_org, fontFace, fontScale, color)
    text_image = cv2.putText(text_image, score_text, score_org, fontFace, fontScale, color)
    text_image = cv2.putText(text_image, action_text, action_org, fontFace, fontScale, color)

    cv2.imshow("status", text_image)
    cv2.waitKey(1)


def main():
    env = gym.make('CartPole-v1', render_mode="human")
    pi = Policy()
    pi.load_state_dict(torch.load("model_parameters/ch9_REINFORCE.pt").state_dict())

    score = 0.0
    s, _ = env.reset()
    done = False

    # test model only once
    n_epi = 1
    while not done:
        prob = pi(torch.from_numpy(s).float())  # action probability
        a = np.argmax(prob.detach().numpy())  # get maximum action
        s_prime, r, done, truncated, info = env.step(a)  # step once
        s = s_prime  # change state
        score += r  # accumulate reward

        display_status(n_epi, score, a)

    print("policy test result \nscore : {}".format(score))
    env.close()


if __name__ == '__main__':
    main()