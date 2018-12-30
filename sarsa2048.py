import rl_glue
import numpy as np
import game2048


class AgentSarsa2048(rl_glue.BaseAgent):
    @classmethod
    def _codenum(cls, num: int)->list:
        return [int(num & 2**i != 0) for i in range(1, 12)]

    @classmethod
    def _codetile(cls, tile)->list:
        raw_feature = []
        for row in tile:
            for col in row:
                raw_feature.extend(cls._codenum(col))
        return raw_feature

    def __init__(self, epsilon=0.1, alpha=0.1, gamma=0.9):
        self._epsilon = epsilon
        self._gamma = gamma
        self._alpha = alpha
        self._w_vec = None

    def agent_end(self, reward):
        raw_grad_shift = (reward - (self._oldfeatvec @ self._w_vec))
        self._w_vec = self._w_vec + \
            (self._alpha*(raw_grad_shift*self._oldfeatvec))
        self._oldfeatvec = None

        print('trial ended. reward=', reward)
        # print('w=', self._w_vec)

    def agent_init(self):
        self._w_vec = np.zeros(16*11*4)
        self._oldfeatvec = None

    def q_func(self, state, action):
        return self._featvec(state, action) @ self._w_vec

    def _featvec(self, tile, action):
        arrsize = 16*11
        base_feat_vec = np.zeros(arrsize*4)
        base_tile = self._codetile(tile)

        base_feat_vec[action*arrsize:(action+1)*arrsize] = base_tile
        return base_feat_vec

    def epsilongreedy(self, tile, possible_actions: set)->int:
        if np.random.rand() < self._epsilon:
            return np.random.choice(list(possible_actions))
        else:
            max_policy = -np.inf
            ideal_policy = None

            for action in possible_actions:
                new_q_val = self.q_func(tile, action)
                if new_q_val > max_policy:
                    ideal_policy = action
                    max_policy = new_q_val
            return ideal_policy

    def agent_start(self, state):
        action = self.epsilongreedy(state['tiles'], state['okaymoves'])
        self._oldfeatvec = self._featvec(state['tiles'], action)
        return action

    def agent_step(self, reward, state):
        new_action = self.epsilongreedy(state['tiles'], state['okaymoves'])

        raw_grad_shift = (reward+self._gamma*(self.q_func(state['tiles'], new_action)) -
                          (self._oldfeatvec @ self._w_vec))

        self._w_vec = self._w_vec + \
            (self._alpha*(raw_grad_shift*self._oldfeatvec))

        self._oldfeatvec = self._featvec(state['tiles'], new_action)

        return new_action
