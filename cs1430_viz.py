import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
import gym
from gym import spaces

# ---- Classes ----

class Satellite:
    def __init__(self, sat_id):
        self.id = sat_id
        self.queue_length = 0
        self.elevation_angle = 0
        self.is_visible = False

    def update_state(self, timestep):
        # simulates if sattelite is moving; elevation angle varies sinusoidally
        self.elevation_angle = 90 * np.abs(np.sin(2 * np.pi * (timestep + self.id) / 24))
        # visible if elevation angle > 10 degrees
        self.is_visible = self.elevation_angle > 10
        # randomly add packets to the queue
        self.queue_length += np.random.randint(0, 5)

class Controller:
    def __init__(self, satellites):
        self.satellites = satellites

    def compute_reward(self, sat):
        queue_penalty = sat.queue_length * 0.1  # penalize long queues
        visibility_bonus = sat.elevation_angle / 90  # normalize [0,1]
        if not sat.is_visible:
            visibility_bonus = 0  # if not visible, no bonus
        reward = visibility_bonus - queue_penalty
        return reward
       
class SatelliteEnv(gym.Env):
    """Custom Environment for Satellite Routing"""
    metadata = {'render.modes': ['human']}

    def __init__(self, num_sats=24, max_timesteps=200):
        super(SatelliteEnv, self).__init__()

        self.num_sats = num_sats
        self.max_timesteps = max_timesteps
        self.timestep = 0

        # define action and observation space

        # action: choose which satellite to forward to (0 to num_sats-1)
        self.action_space = spaces.Discrete(self.num_sats)

        # observation: queue length, elevation angle, visibility (per satellite)
        # For simplicity, assume agent controls one satellite at a time
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)

        # initialize satellite states
        self.reset()

    def reset(self):
        self.timestep = 0
        self.queue_length = np.random.randint(0, 10, size=self.num_sats)
        self.elevation_angle = 90 * np.abs(np.sin(2 * np.pi * (np.arange(self.num_sats)) / 24))
        self.visibility = self.elevation_angle > 10

        # start from satellite 0 for simplicity
        self.current_sat_id = 0
        return self._get_obs()

    def step(self, action):
        self.timestep += 1

        # Simulate environment transition: move to selected satellite
        self.current_sat_id = action

        # update satellite states
        self.elevation_angle = 90 * np.abs(np.sin(2 * np.pi * (self.timestep + np.arange(self.num_sats)) / 24))
        self.visibility = self.elevation_angle > 10
        self.queue_length += np.random.randint(0, 5, size=self.num_sats)

        # compute reward
        reward = self._compute_reward(self.current_sat_id)

        # check if done
        done = self.timestep >= self.max_timesteps

        # info dict (optional extra info)
        info = {}

        return self._get_obs(), reward, done, info

    def _get_obs(self):
        # return current satellite's state (normalized)
        sat_idx = self.current_sat_id
        queue_norm = self.queue_length[sat_idx] / 50  # assumption of max 50 packets
        elev_norm = self.elevation_angle[sat_idx] / 90
        visibility_flag = 1.0 if self.visibility[sat_idx] else 0.0
        return np.array([queue_norm, elev_norm, visibility_flag], dtype=np.float32)

    def _compute_reward(self, sat_idx):
        queue_penalty = self.queue_length[sat_idx] * 0.1
        visibility_bonus = (self.elevation_angle[sat_idx] / 90)
        if not self.visibility[sat_idx]:
            visibility_bonus = 0
        return visibility_bonus - queue_penalty

    def render(self, mode='human'):
        # print the current state
        print(f"Timestep: {self.timestep}, Sat: {self.current_sat_id}, Queue: {self.queue_length[self.current_sat_id]}, Elevation: {self.elevation_angle[self.current_sat_id]:.2f}, Visible: {self.visibility[self.current_sat_id]}")


# ---- streamlit App ----
st.title("LEO Satellite Routing Simulator + DRL Agent Trainer")

# sidebar: Settings
mode = st.sidebar.selectbox("Choose Mode", ["Reward Simulator", "DRL Agent Trainer"])

num_sats = st.sidebar.slider('Number of Satellites', 10, 50, 24)
timesteps = st.sidebar.slider('Number of Timesteps', 50, 500, 200)
queue_weight = st.sidebar.slider('Queue Penalty Weight', 0.0, 1.0, 0.1)
visibility_weight = st.sidebar.slider('Visibility Bonus Weight', 0.0, 2.0, 1.0)

# --- Reward Simulator Mode ---
if mode == "Reward Simulator":
    st.header("Reward Function Visualization")

    satellites = [Satellite(i) for i in range(num_sats)]
    controller = Controller(satellites)

    all_rewards = []

    for t in range(timesteps):
        rewards = []
        for sat in satellites:
            sat.update_state(t)
            reward = controller.compute_reward(sat)
            rewards.append(reward)
        all_rewards.append(rewards)

    all_rewards = np.array(all_rewards)

    fig, ax = plt.subplots(figsize=(12,6))
    cax = ax.imshow(all_rewards.T, aspect='auto', cmap='viridis', interpolation='nearest')
    fig.colorbar(cax, label='Reward Value')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Satellite ID')
    st.pyplot(fig)

# --- DRL Agent Trainer Mode ---
else:
    st.header("Deep RL Agent Training")

    env = SatelliteEnv(num_sats=num_sats, max_timesteps=timesteps)
    model = DQN('MlpPolicy', env, verbose=0)
    with st.spinner('Training the agent... (this takes ~20s)'):
        model.learn(total_timesteps=10000)

    st.success('Training complete!')

    # Training reward collection (Optional advanced feature)

    # Evaluate trained agent
    agent_rewards = []
    sat_choices = []

    obs = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        sat_choices.append(action)
        obs, reward, done, info = env.step(action)
        agent_rewards.append(reward)
        if done:
            obs = env.reset()

    # Evaluate random agent
    random_rewards = []
    obs = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        random_rewards.append(reward)
        if done:
            obs = env.reset()

    # --- Plots ---

    st.subheader("1. Trained Agent Satellite Selection Histogram")
    fig, ax = plt.subplots()
    ax.hist(sat_choices, bins=np.arange(num_sats+1)-0.5, rwidth=0.8)
    ax.set_xlabel('Satellite ID')
    ax.set_ylabel('Times Selected')
    ax.set_title('Which Satellites the Agent Prefers')
    st.pyplot(fig)

    st.subheader("2. Agent vs Random Average Reward")
    st.write(f"**Trained Agent Avg Reward**: {np.mean(agent_rewards):.3f}")
    st.write(f"**Random Agent Avg Reward**: {np.mean(random_rewards):.3f}")
