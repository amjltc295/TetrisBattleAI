import time
from engine import TetrisEngine
import torch
import ac_agent
import numpy as np
width, height = 10, 20  # standard tetris friends rules
engine = TetrisEngine(width, height)


def run():
    model = ac_agent.AC()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    epoch, best_score = ac_agent.load_checkpoint(model, './tar/ac_best.pth.tar')
    model.train()
#     model.eval()  # That make network fail, I guess that because of BathNormal layer.
    print('training epoch : %d, best score %.2f' % (epoch, best_score))
    time.sleep(2)
    score = 0
    state = engine.clear()
    state, _, _, _ = engine.step('idle')
    while True:
        with torch.no_grad():
            action_final_location_map = engine.get_valid_final_states(engine.shape, engine.anchor, engine.board)
            act_pairs = [(k, v[2]) for k, v in action_final_location_map.items()]
            act_prob, V = ac_agent.get_action_probability(model, state, act_pairs)
            act_idx = int(np.random.choice(len(act_prob), 1, p=act_prob.cpu().detach().numpy()))
            act, placement = act_pairs[act_idx]
            state, reward, done, cleared_lines = engine.step_to_final(act)
            # Accumulate reward
            score += int(cleared_lines)

            print(engine)
            print('reward : %.2f' % reward)
            print('score : %d' % score)
            time.sleep(0.1)

            if done:
                print('score {0}'.format(score))
                break


if __name__ == '__main__':
    run()
