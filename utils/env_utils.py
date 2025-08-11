from typing import Dict


class Env:
    """
    Fake environment class.
    """

    def __init__(self):
        self.max_step = 2
        self.prompt = None
        self._cnt = 0
        self.done = False

    def reset(self, prompt: str) -> Dict:
        self.prompt = prompt
        self._cnt = 0
        self.done = False
        return {
            'obs': self.prompt,
            'reward': 0,
            'done': False,
            'info': {}
        }
    
    def step(self, action: int) -> Dict:
        self._cnt += 1
        ret = {
            'obs': self.prompt + str(action),
            'reward': 1 if action == 2 else 0,
            'done': self._cnt >= self.max_step or action == 2,
            'info': {}
        }
        if ret['done']:
            self.done = True
            ret['info']['final_eval_return'] = 10 if action == 2 else 0
        return ret
