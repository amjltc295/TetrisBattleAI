import time
from engine import TetrisEngine
import fixed_policy_agent

width, height = 10, 20  # standard tetris friends rules
engine = TetrisEngine(width, height)


def run():
    score = 0
    while True:

        action, placement = fixed_policy_agent.select_action(engine, engine.shape, engine.anchor, engine.board)

        state, reward, done, cleared_lines = engine.step_to_final(action)

        # Accumulate reward
        score += int(cleared_lines)

        print(engine)
        print('reward : %.2f' % reward)
        print('score : %d' % score)
        # print(action)
        time.sleep(1)

        if done:
            print('score {0}'.format(score))
            break


if __name__ == '__main__':
    run()
