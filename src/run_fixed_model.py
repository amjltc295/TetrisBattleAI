import time
from engine import TetrisEngine
from fixed_policy_agent import FixedPolicyAgent

width, height = 10, 20  # standard tetris friends rules
engine = TetrisEngine(width, height)


def run():
    agent = FixedPolicyAgent()
    score = 0
    while True:
        actions_name, placement, actions = agent.select_action(engine, engine.shape, engine.anchor, engine.board)

        state, reward, done, cleared_lines, sent_lines = engine.step_to_final(actions_name)

        # Accumulate reward
        score += int(cleared_lines)

        print(engine)
        print('score : %d' % score)
        # print(action)
        time.sleep(.1)

        if done:
            print('score {0}'.format(score))
            break


if __name__ == '__main__':
    run()
