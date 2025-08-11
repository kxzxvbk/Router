from .env_utils import Env
from torch.nn.functional import log_softmax
import torch
import random


def default_decollate(batch):
    keys = batch.keys()
    return [{k: batch[k][i] for k in keys} for i in range(len(batch))]


def default_collate(batch):
    keys = batch[0].keys()
    return {k: [d[k] for d in batch] for k in keys}


def compute_returns(reward, done, gamma=0.99):
    returns = []
    R = 0
    for r, d in zip(reward[::-1], done[::-1]):
        if d:
            R = r
        else:
            R = r + gamma * R
        returns.append(R)
    returns.reverse()
    return returns


class PGTrainer:
    def __init__(self, model, tokenizer, train_dataloader, test_dataloader, batch_size):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.batch_size = self.batch_size

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-5, weight_decay=1e-2
        )
        self.train_idx = 0

        self.envs = [Env() for _ in range(self.batch_size)]
    
    def _rollout(self, num_traj=8):
        assert num_traj % self.batch_size == 0, "num_traj must be divisible by batch_size"
        num_episode = num_traj // self.batch_size
        trajectories = [[] for i in range(self.num_traj)]

        for _ in range(num_episode):
            transitions = []
            for i, env in enumerate(self.envs):
                prompt_idx = (self.train_idx + i) % len(self.train_dataloader)
                transition = env.reset(self.train_dataloader[prompt_idx])
                transitions.append(transition)
            transitions = default_collate(transitions)
            
            while True:
                obs = transitions['obs']
                obs = self.tokenizer(obs, padding=True, return_tensors='pt')
                logits = self.model(**obs)
                # Use mutinomial sampling to sample an action.
                action = torch.multinomial(logits, num_samples=1).squeeze(1).tolist()

                transitions = []
                for i, env in enumerate(self.envs):
                    transition = env.step(action[i])
                    transitions.append(transition)
                transitions = default_collate(transitions)
                
                batch_data = {
                    'obs': obs,
                    'action': action,
                    'reward': transitions['reward'],
                    'done': transitions['done'],
                }
                batch_data = default_decollate(batch_data)
                for i in range(self.batch_size):
                    trajectories[i].append(batch_data[i])
                
                if all(transitions['done']):
                    break

            self.train_idx = (self.train_idx + self.batch_size) % len(self.train_dataloader)
        
        trajectories = [default_collate(traj) for traj in trajectories]
        for i in range(num_traj):
            trajectories[i]['return'] = compute_returns(trajectories[i]['reward'], trajectories[i]['done'])
        return default_decollate(trajectories)

    def _update(self, trajectories):
        random.shuffle(trajectories)
        for start_idx in range(0, len(trajectories), self.batch_size):
            end_idx = start_idx + self.batch_size
            batch = trajectories[start_idx:end_idx]
            batch = default_collate(batch)

            obs = self.tokenizer(batch['obs'], padding=True, return_tensors='pt')
            action = torch.tensor(batch['action'])
            logits = self.model(**obs)
            log_probs = log_softmax(logits, dim=1)
            log_probs = log_probs.gather(1, action.unsqueeze(1)).squeeze(1)
            returns = torch.tensor(batch['return'])
            loss = -(log_probs * returns).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self._iter_num += 1
    
    def train(self, num_iters):
        self._iter_num = 0
        while True:
            with torch.no_grad():
                trajectories = self._rollout()
            self._update(trajectories)
            if self._iter_num >= num_iters:
                break
