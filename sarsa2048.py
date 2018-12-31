import rl_glue
import numpy as np
import game2048


class AgentSarsa2048(rl_glue.BaseAgent):
    _enable_quadratic_base = False

    @classmethod
    def _codenum(cls, num: int)->list:
        return [int(num & 2**i != 0) for i in range(1, 11)]

    @classmethod
    def _codetile(cls, tile)->list:
        raw_feature = []
        for row in tile:
            for col in row:
                raw_feature.extend(cls._codenum(col))

        quadratic_base = []

        if cls._enable_quadratic_base:
            for x in raw_feature:
                for y in raw_feature:
                    quadratic_base.append(x*y)

        return raw_feature+quadratic_base

    def __init__(self, epsilon=0.1, alpha=0.05, gamma=0.9):
        self._epsilon = epsilon
        self._gamma = gamma
        self._alpha = alpha
        self._w_vec = None

    def agent_end(self, reward):
        raw_grad_shift = (reward - (self._oldfeatvec @ self._w_vec))
        self._w_vec = self._w_vec + \
            (self._alpha*(raw_grad_shift*self._oldfeatvec))
        self._oldfeatvec = None

        # print('trial ended. reward=', reward)
        # print('w=', self._w_vec)

    def agent_init(self):
        self._w_vec = None
        self._oldfeatvec = None

    def q_func(self, state, action):
        q_val = self._featvec(state, action) @ self._w_vec
        assert not np.isnan(q_val)
        return q_val

    def _featvec(self, tile, action):
        base_tile = self._codetile(tile)
        arrsize = len(base_tile)
        base_feat_vec = np.zeros(arrsize*4)

        if action is None:
            return base_feat_vec
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
            assert ideal_policy is not None
            return ideal_policy

    def agent_start(self, state):
        if self._w_vec is None:
            self._w_vec = np.zeros(self._featvec(
                state['tiles'], list(state['okaymoves'])[0]).shape)

        action = self.epsilongreedy(state['tiles'], state['okaymoves'])
        self._oldfeatvec = self._featvec(state['tiles'], action)
        return action

    def agent_step(self, reward, state):
        new_action = self.epsilongreedy(state['tiles'], state['okaymoves'])

        raw_grad_shift = (reward+self._gamma*(self.q_func(state['tiles'], new_action)) -
                          (self._oldfeatvec @ self._w_vec))

        assert not np.isnan(raw_grad_shift)

        self._w_vec = self._w_vec + \
            (self._alpha*(raw_grad_shift*self._oldfeatvec))

        self._oldfeatvec = self._featvec(state['tiles'], new_action)

        return new_action
