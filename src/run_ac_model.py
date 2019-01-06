import time
from engine import TetrisEngine
import torch
import ac_agent
import numpy as np
width, height = 10, 20  # standard tetris friends rules


def run(engine):
    model = ac_agent.AC()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    epoch, best_score = ac_agent.load_checkpoint(model, './tar/ac_sent.pth.tar')
    # epoch, best_score = ac_agent.load_checkpoint(model, './tar/ac_best.pth.tar')
    # model.train()
#     model.eval()  # That make network fail, I guess that because of BathNormal layer.
    print('training epoch : %d, best score %.2f' % (epoch, best_score))
    time.sleep(2)
    score = 0
    state = engine.clear()
    state, _, _, _, _ = engine.step('idle')
    i = 0
    while True:
        with torch.no_grad():
            _, _, actions = model.select_action(engine, state)
            state, reward, done, cleared_lines, _ = engine.step_to_final(actions)
            # Accumulate reward
            # score += int(cleared_lines)
            score += int(_)

            # print(engine)
            # print('reward : %.2f' % reward)
            # print('score : %d' % score)
            # time.sleep(0.1)
            i += 1
            if done or i > 1000:
                # print('score {0}'.format(score))
                # time.sleep(2)
                break
    return score


if __name__ == '__main__':
    buf = []
    for i in range(100):
        engine = TetrisEngine(width, height)
        s = run(engine)
        print(i, s)
        buf.append(s)
    print('mean score %.2f, var : %.3d' % (np.mean(buf), np.var(buf)))
