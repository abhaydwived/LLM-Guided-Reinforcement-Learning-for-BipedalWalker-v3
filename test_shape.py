import gymnasium as gym
import imageio
from env.bipedal_env import make_env
env = make_env(render_mode='rgb_array')
env.reset()
done = False
steps = 0
frames = []
while not done and steps < 6:
    obs, _, term, trunc, _ = env.step(env.action_space.sample())
    done = term or trunc
    frame = env.render()
    if frame is None:
        print(f"Step {steps}: frame is None!")
    else:
        print(f"Step {steps}: frame shape {frame.shape}")
    frames.append(frame)
    steps += 1
env.close()
print("Shapes:", [f.shape if f is not None else None for f in frames])
