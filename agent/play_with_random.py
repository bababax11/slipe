from game.game_state import GameState

logger = getLogger(__name__)

class PlayWithRandomWorker:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.model = None  # type: SlipeModel
        
    def play_with_random(self):
        gs = GameState()
        model.predict()