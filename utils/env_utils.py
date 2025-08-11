class Env:
    def __init__(self):
        self.max_step = 10
        self.prompt = None
        self._cnt = 0

    def reset(self, prompt):
        self.prompt = prompt
        self._cnt = 0
        return {
            'obs': self.prompt,
            'reward': 0,
            'done': False,
            'info': {}
        }
    
    def step(self, action):
        self._cnt += 1
        ret = {
            'obs': self.prompt + str(self.action),
            'reward': 0,
            'done': self._cnt >= self.max_step,
            'info': {}
        }
        return ret
