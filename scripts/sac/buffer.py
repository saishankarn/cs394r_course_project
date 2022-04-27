class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_shape, act_shape, buffer_size):
        self.obs_shape = obs_shape 
        self.act_shape = act_shape 
        self.buffer_size = buffer_size 

        """
        ### storage arrays
        """
        self.obs_buffer = np.zeros((self.buffer_size, self.obs_shape), dtype=np.float32)
        self.next_obs_buffer = np.zeros((self.buffer_size, self.obs_shape), dtype=np.float32)
        self.action_buffer = np.zeros((self.buffer_size, self.action_shape), dtype=np.float32)
        self.reward_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.done_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        
        """
        ### keeping track of the buffer
        """
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        """
        ### storing the elements in the buffer
        """
        self.obs_buffer[self.ptr] = obs
        self.next_obs_buffer[self.ptr] = next_obs
        self.action_buffer[self.ptr] = act
        self.reward_buffer[self.ptr] = rew
        self.done_buffer[self.ptr] = done

        """
        ### updating the buffer trackers
        """
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buffer[idxs],
                     obs2=self.next_obs_buffer[idxs],
                     act=self.action_buffer[idxs],
                     rew=self.reward_buffer[idxs],
                     done=self.done_buffer[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}