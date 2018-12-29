import numpy as np
from genetic_heuristic import gen_heuristic

best_dict_genes = {'holes_stack_area': -0.8461356563843694,
                   'holes_clean_area': -0.4615721400388464,
                   'height_stack_area': 0.06473160638588638,
                   'height_clean_area': -0.14280823125594533,
                   'aggregation_stack_area': 0.0801835665088445,
                   'bumpiness': -0.37476627592742084,
                   'clear_lines': 0.8648134249140578}


class GeneticPolicyAgent:
    def __int__(self):
        self.current_actions = []

    def get_action(self, engine, shape, anchor, board):
        if len(self.current_actions) == 0:
            _, _, self.current_actions = self.select_action(engine, shape, anchor, board)
        action = self.current_actions.pop(0)
        return action

    def select_action(self, engine, shape, anchor, board, dict_genes=best_dict_genes):
        # All possible final states
        actions_name_final_location_map = engine.get_valid_final_states(shape, anchor, board)
        # act_pairs = (dict_key, final_board, actions)
        act_pairs = [(k, v[2], v[3]) for k, v in actions_name_final_location_map.items()]
        # Only final boards
        placements = [p for k, p, actions in act_pairs]
        # Uses the heuristic for every possible placement
        h_score = [gen_heuristic(s, dict_genes) for s in placements]
        act_idx = np.argmax(h_score)
        actions_name, final_placement, actions = act_pairs[act_idx]
        return actions_name, final_placement, actions
