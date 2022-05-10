import numpy as np 
import torch 

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """ 

    def __init__(self, state_space, action_space, buffer_size):
        self.state_dim = state_space.shape[0] 
        self.action_dim = action_space.shape[0] 
        self.buffer_size = int(buffer_size) 

        """
        ### storage arrays
        """ 
        self.state_buffer = np.zeros((self.buffer_size, self.state_dim), dtype=np.float32)
        self.next_state_buffer = np.zeros((self.buffer_size, self.state_dim), dtype=np.float32)
        self.action_buffer = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32)
        self.reward_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.done_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        
        """
        ### keeping track of the buffer
        """
        self.mem_counter, self.count = 0, 0

    def store(self, obs, act, rew, next_obs, done):
        """
        ### storing the elements in the buffer
        """
        self.state_buffer[self.mem_counter] = obs
        self.next_state_buffer[self.mem_counter] = next_obs
        self.action_buffer[self.mem_counter] = act
        self.reward_buffer[self.mem_counter] = rew
        self.done_buffer[self.mem_counter] = done

        """
        ### updating the buffer trackers
        """
        self.mem_counter = (self.mem_counter+1) % self.buffer_size
        self.count = min(self.count+1, self.buffer_size)

    def sample_batch(self, batch_size=256):
        idxs = np.random.randint(0, self.count, size=batch_size)
        batch = dict(state=self.state_buffer[idxs],
                     next_state=self.next_state_buffer[idxs],
                     action=self.action_buffer[idxs],
                     reward=self.reward_buffer[idxs],
                     done=self.done_buffer[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}