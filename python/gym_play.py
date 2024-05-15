import gymnasium as gym
import pygame
from gymnasium.utils.play import play
import numpy as np

if __name__ == "__main__":
    # a = np.array([0.0, 0.0, 0.0])
    actions = gym.spaces.Discrete(5)
    # action space가
    #   0: do nothing
    #   1: right
    #   2: left
    #   3: gas
    #   4: brake
    # 인데 document에 오해되도록 써있는 것 같음.
    action = np.asarray([1, 0, 0, 0, 0]).astype(np.int8)

    def register_input():
        global quit, restart
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                action[0] = 0
                if event.key == pygame.K_LEFT:
                    action[1] = 1   # ! steer right
                if event.key == pygame.K_RIGHT:
                    action[2] = 1   # ! steer left
                if event.key == pygame.K_UP:
                    action[3] = 1
                if event.key == pygame.K_DOWN:
                    action[4] = 1
                if event.key == pygame.K_RETURN:
                    restart = True
                if event.key == pygame.K_ESCAPE:
                    quit = True

            if event.type == pygame.KEYUP:
                action[0] = 1
                if event.key == pygame.K_LEFT:
                    action[1] = 0
                if event.key == pygame.K_RIGHT:
                    action[2] = 0
                if event.key == pygame.K_UP:
                    action[3] = 0
                if event.key == pygame.K_DOWN:
                    action[4] = 0

            if event.type == pygame.QUIT:
                quit = True

    env = gym.make("CarRacing-v2", render_mode="human", continuous=False)

    quit = False
    while not quit:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            register_input()
            s, r, terminated, truncated, info = env.step(actions.sample(action))
            total_reward += r
            if steps % 200 == 0 or terminated or truncated:
                print("\naction " + str([f"{x:+0.2f}" for x in action]))
                print(f"step {steps} total_reward {total_reward:+0.2f}")
            steps += 1
            if terminated or truncated or restart or quit:
                break
    env.close()


# mapping = {(ord('w'),): 1,
#            (pygame.K_LEFT,): 2,
#            (pygame.K_RIGHT,): 3,
#            (pygame.K_UP,): 4,
#            (pygame.K_DOWN,): 5}
# play(gym.make("CarRacing-v2", render_mode='rgb_array'), keys_to_action=mapping)