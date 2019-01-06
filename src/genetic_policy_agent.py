import numpy as np
from genetic_heuristic import gen_heuristic
from engine import board_to_bool

genes = ['holes_stack_area', 'holes_clean_area', 'height_stack_area', 'height_clean_area',
         'aggregation_stack_area', 'bumpiness', 'clear_lines']
best_dict_genes = {'holes_stack_area': 0.0, 'holes_clean_area': 0.0,
                   'height_stack_area': 0.0, 'height_clean_area': 0.0,
                   'aggregation_stack_area': 0.0, 'bumpiness': 0.0,
                   'clear_lines': 0.0}
best_genes = [-0.9137025073438482, -0.7636964099499323, 0.3542078531470947,
              0.07625851575250508, 0.11611171127846975, -0.8938826745335662,
              0.9907288665763544]
best_genes = [-0.7821491111895829, -0.6671129164435949, 0.12497164917174183, 0.15764411276340012,
              0.12024968472592018, -0.6700437752718681, 0.7000094499592582]
best_genes = [-0.8124, -0.9490, 0.2454, 0.1547, 0.1565, -0.7950, 0.2895]  # Almost equaly with fixed policy
best_genes = [-0.9190, -0.8406, 0.3974, 0.3549, 0.1013, -0.6039, 0.6139]  # Almost equaly with fixed policy
best_genes = [-0.9190, -0.8406, 0.2633, 0.3549, 0.1013, -0.6039, 0.6139]  # Equal
best_genes = [-0.2717, -0.3470, 0.2690, 0.5539, 0.1971, -0.9196, 0.8979]  # Win a lot with no much line sent
best_genes = [-0.9562, -0.5508, 0.0103, 0.2119, 0.2583, -0.5287, 0.6013]
best_genes1 = [-0.8493104761925934, -0.8012356257711567,
               0.14844302001398635, 0.21794814052624,
               0.06695370686132018, -0.6347949174805239,
               0.8552733682112534]
best_genes2 = [-0.9507019177907516, -0.8753333601276717, 0.2490605092681012,
               0.36751156858451006, 0.024458263275652037, -0.8586804511809092,
               0.8698604051422936]
best_genes3 = [-0.8985143311844761, -0.9229569801813504, 0.7242996635737323,
               0.3480772367706604, 0.014645026400500138, -0.8089358529633948,
               0.9768549625440047]


class GeneticPolicyAgent:
    def __init__(self):
        self.current_actions = []

    def get_action(self, engine, shape, anchor, board):
        if len(self.current_actions) == 0:
            _, _, self.current_actions = self.select_action(engine, shape, anchor, board)
        action = self.current_actions.pop(0)
        return action

    def select_action(self, engine, shape, anchor, board, dict_genes=None):
        if dict_genes is None:
            for i in range(len(best_genes)):
                best_dict_genes[genes[i]] = best_genes[i]
            dict_genes = best_dict_genes
        # All possible final states
        actions_name_final_location_map = engine.get_valid_final_states(shape, anchor, board)
        # act_pairs = (dict_key, final_board, actions)
        act_pairs = [(k, v[2], v[3]) for k, v in actions_name_final_location_map.items()]
        # Only final boards
        placements = [board_to_bool(p) for k, p, actions in act_pairs]
        # Uses the heuristic for every possible placement
        h_score = [gen_heuristic(s, dict_genes) for s in placements]
        act_idx = np.argmax(h_score)
        actions_name, final_placement, actions = act_pairs[act_idx]
        return actions_name, final_placement, actions
