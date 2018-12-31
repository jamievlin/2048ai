import rl_glue
from game2048 import Game2048


class Env2048(rl_glue.BaseEnvironment):
    def __init__(self):
        self._2048game = None
        self.moves_dict = None

    def env_init(self):
        self._2048game = None
        self.moves_dict = None
        self._gamecount = 0

    def env_message(self, message):
        return super().env_message(message)

    def env_start(self):
        self._2048game = Game2048()
        self.moves_dict = {
            0: self._2048game.slideLeft,
            1: self._2048game.slideRight,
            2: self._2048game.slideUp,
            3: self._2048game.slideDown
        }
        return self.get_state()

    def env_step(self, action):
        result = self.moves_dict[action]()
        assert result == True

        terminal = False
        reward = 0

        self._2048game.RandomFillTile(self._2048game.random2or4())

        if self._2048game.win():
            terminal = True
            reward = 10000+self._2048game.getScore()
        elif self._2048game.getNbEmptyTiles() == 0 and not self._2048game.collapsible():
            terminal = True
            reward = 0.1*self._2048game.getScore()

        if terminal:
            self._gamecount = self._gamecount+1

            if self._gamecount % 50 == 0:
                self._2048game.print()

        return reward, self.get_state(), terminal

    def get_state(self)->dict:

        valid_moves = set()
        for moves, action in self.moves_dict.items():
            if action(True):
                valid_moves.add(moves)

        return {
            'score': self._2048game.getScore(),
            'tiles': self._2048game.gamegrid,
            'okaymoves': valid_moves
        }

        # for now,
        # 2048 -> 16*11 list
        # each number corresponds to a bit
        # and the last two corresponds to actions.
