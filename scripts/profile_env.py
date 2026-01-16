#!/usr/bin/env python3
"""Profile mjlab environment to identify performance bottlenecks.

This script profiles:
- Physics simulation (MuJoCo stepping)
- Observation computation
- Reward computation
- Termination computation
- Reset operations

Usage:
    python scripts/profile_env.py --task Mjlab-Velocity-Flat-Unitree-G1 --num-envs 4096
"""

import argparse
import time

import torch

import mjlab.tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv


def profile_environment(task_name: str, num_envs: int, num_steps: int = 1000):
  """Profile environment performance over multiple steps."""
  
  print(f"\n{'='*70}")
  print(f"Profiling: {task_name}")
  print(f"Num envs: {num_envs:,}")
  print(f"Num steps: {num_steps:,}")
  print(f"{'='*70}\n")

  # Create environment
  print("Creating environment...")
  env_cfg = load_env_cfg(task_name)
  env_cfg.scene.num_envs = num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda")
  sim = env.sim

  # Print backend status for performance debugging.
  try:
    import warp as wp  # type: ignore
    print("Backend status:")
    print(f"  Warp device....................... {sim.wp_device}")
    print(f"  CUDA graph enabled............... {sim.use_cuda_graph}")
    print(f"  Warp mempool enabled............ {wp.is_mempool_enabled(sim.wp_device)}")
    print(f"  MuJoCo iterations................ {sim.cfg.mujoco.iterations}")
    print(f"  MuJoCo ls_iterations............ {sim.cfg.mujoco.ls_iterations}")
    print(f"  Decimation...................... {env.cfg.decimation}")
  except Exception:
    pass
  
  # Warmup
  print("Warming up...")
  env.reset()
  num_actions = env.single_action_space.shape[0]
  for _ in range(10):
    actions = torch.randn(num_envs, num_actions, device="cuda").clamp(-1, 1)
    env.step(actions)
  torch.cuda.synchronize()

  # Profile loop
  print(f"\nProfiling {num_steps} steps...\n")
  
  total_time = 0.0
  sim_time = 0.0
  obs_time = 0.0
  reward_time = 0.0
  term_time = 0.0
  command_time = 0.0
  reset_time = 0.0
  
  episode_step = torch.zeros(num_envs, device="cuda", dtype=torch.long)
  max_episode_length = 200
  
  torch.cuda.synchronize()
  loop_start = time.perf_counter()
  
  for step in range(num_steps):
    # Generate random actions
    actions = torch.randn(num_envs, num_actions, device="cuda").clamp(-1, 1)
    
    # Break down individual components
    # Commands
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if hasattr(env, "command_manager"):
        env.command_manager.compute(dt=env.physics_dt)
    torch.cuda.synchronize()
    command_time += time.perf_counter() - t0

    # Physics simulation
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(env.cfg.decimation):
      sim.step()
    torch.cuda.synchronize()
    sim_time += time.perf_counter() - t0
    
    # Observations  
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = env.observation_manager.compute()
    torch.cuda.synchronize()
    obs_time += time.perf_counter() - t0
    
    # Rewards
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = env.reward_manager.compute(dt=env.physics_dt)
    torch.cuda.synchronize()
    reward_time += time.perf_counter() - t0
    
    # Terminations
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = env.termination_manager.compute()
    truncated = episode_step >= max_episode_length
    torch.cuda.synchronize()
    term_time += time.perf_counter() - t0
    
    # Handle resets (simplified - using full reset)
    dones = truncated
    if dones.any():
      torch.cuda.synchronize()
      t0 = time.perf_counter()
      reset_ids = torch.nonzero(dones).squeeze(-1)
      env._reset_idx(reset_ids)
      episode_step[reset_ids] = 0
      torch.cuda.synchronize()
      reset_time += time.perf_counter() - t0
    
    episode_step += 1
    
    # Progress update
    if (step + 1) % 100 == 0:
      print(f"  Step {step+1}/{num_steps}", end="\r")
  
  torch.cuda.synchronize()
  total_time = time.perf_counter() - loop_start
  
  # Print results
  print(f"\nTotal time breakdown:")
  print(f"  {'Total loop time':.<40} {total_time*1000:.2f} ms")
  print(f"  {'Physics simulation':.<40} {sim_time*1000:.2f} ms ({sim_time/total_time*100:.1f}%)")
  print(f"  {'Command computation':.<40} {command_time*1000:.2f} ms ({command_time/total_time*100:.1f}%)")
  print(f"  {'Observation computation':.<40} {obs_time*1000:.2f} ms ({obs_time/total_time*100:.1f}%)")
  print(f"  {'Reward computation':.<40} {reward_time*1000:.2f} ms ({reward_time/total_time*100:.1f}%)")
  print(f"  {'Termination computation':.<40} {term_time*1000:.2f} ms ({term_time/total_time*100:.1f}%)")
  print(f"  {'Reset operations':.<40} {reset_time*1000:.2f} ms ({reset_time/total_time*100:.1f}%)")
  print()
  print("Per-step averages:")
  print(f"  {'Total per step':.<40} {total_time/num_steps*1000:.3f} ms")
  print(f"  {'Physics per step':.<40} {sim_time/num_steps*1000:.3f} ms")
  print(f"  {'Commands per step':.<40} {command_time/num_steps*1000:.3f} ms")
  print(f"  {'Observations per step':.<40} {obs_time/num_steps*1000:.3f} ms")
  print(f"  {'Rewards per step':.<40} {reward_time/num_steps*1000:.3f} ms")
  print(f"  {'Terminations per step':.<40} {term_time/num_steps*1000:.3f} ms")
  print()
  env_steps = num_steps * num_envs
  print("Throughput:")
  print(f"  {'Env-steps/second':.<40} {env_steps/total_time:,.0f}")
  print(f"  {'Million env-steps/second':.<40} {env_steps/total_time/1e6:.2f}")
  print()
  print(f"{'='*70}\n")


def profile_environment_with_cfg(env_cfg, num_envs: int, num_steps: int = 1000):
  """Profile environment performance with a preconfigured env cfg."""
  print(f"\n{'='*70}")
  print(f"Profiling: (custom cfg)")
  print(f"Num envs: {num_envs:,}")
  print(f"Num steps: {num_steps:,}")
  print(f"{'='*70}\n")

  env_cfg.scene.num_envs = num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda")
  sim = env.sim

  # Backend status
  try:
    import warp as wp  # type: ignore
    print("Backend status:")
    print(f"  Warp device....................... {sim.wp_device}")
    print(f"  CUDA graph enabled............... {sim.use_cuda_graph}")
    print(f"  Warp mempool enabled............ {wp.is_mempool_enabled(sim.wp_device)}")
    print(f"  MuJoCo iterations................ {sim.cfg.mujoco.iterations}")
    print(f"  MuJoCo ls_iterations............ {sim.cfg.mujoco.ls_iterations}")
    print(f"  Solver.......................... {sim.cfg.mujoco.solver}")
    print(f"  Decimation...................... {env.cfg.decimation}")
  except Exception:
    pass

  # Warmup
  env.reset()
  num_actions = env.single_action_space.shape[0]
  for _ in range(10):
    actions = torch.randn(num_envs, num_actions, device="cuda").clamp(-1, 1)
    env.step(actions)
  torch.cuda.synchronize()

  # Profile loop (same as above)
  total_time = 0.0
  sim_time = 0.0
  obs_time = 0.0
  reward_time = 0.0
  term_time = 0.0
  command_time = 0.0
  reset_time = 0.0

  episode_step = torch.zeros(num_envs, device="cuda", dtype=torch.long)
  max_episode_length = 200

  torch.cuda.synchronize()
  loop_start = time.perf_counter()

  for step in range(num_steps):
    actions = torch.randn(num_envs, num_actions, device="cuda").clamp(-1, 1)

    # Commands
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if hasattr(env, "command_manager"):
        env.command_manager.compute(dt=env.physics_dt)
    torch.cuda.synchronize()
    command_time += time.perf_counter() - t0

    # Physics simulation
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(env.cfg.decimation):
      sim.step()
    torch.cuda.synchronize()
    sim_time += time.perf_counter() - t0

    # Observations
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = env.observation_manager.compute()
    torch.cuda.synchronize()
    obs_time += time.perf_counter() - t0

    # Rewards
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = env.reward_manager.compute(dt=env.physics_dt)
    torch.cuda.synchronize()
    reward_time += time.perf_counter() - t0

    # Terminations
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = env.termination_manager.compute()
    truncated = episode_step >= max_episode_length
    torch.cuda.synchronize()
    term_time += time.perf_counter() - t0

    # Resets
    dones = truncated
    if dones.any():
      torch.cuda.synchronize()
      t0 = time.perf_counter()
      reset_ids = torch.nonzero(dones).squeeze(-1)
      env._reset_idx(reset_ids)
      episode_step[reset_ids] = 0
      torch.cuda.synchronize()
      reset_time += time.perf_counter() - t0

    episode_step += 1

  torch.cuda.synchronize()
  total_time = time.perf_counter() - loop_start

  # Print results
  print(f"\nTotal time breakdown:")
  print(f"  {'Total loop time':.<40} {total_time*1000:.2f} ms")
  print(f"  {'Physics simulation':.<40} {sim_time*1000:.2f} ms ({sim_time/total_time*100:.1f}%)")
  print(f"  {'Command computation':.<40} {command_time*1000:.2f} ms ({command_time/total_time*100:.1f}%)")
  print(f"  {'Observation computation':.<40} {obs_time*1000:.2f} ms ({obs_time/total_time*100:.1f}%)")
  print(f"  {'Reward computation':.<40} {reward_time*1000:.2f} ms ({reward_time/total_time*100:.1f}%)")
  print(f"  {'Termination computation':.<40} {term_time*1000:.2f} ms ({term_time/total_time*100:.1f}%)")
  print(f"  {'Reset operations':.<40} {reset_time*1000:.2f} ms ({reset_time/total_time*100:.1f}%)")
  print()
  print("Per-step averages:")
  print(f"  {'Total per step':.<40} {total_time/num_steps*1000:.3f} ms")
  print(f"  {'Physics per step':.<40} {sim_time/num_steps*1000:.3f} ms")
  print(f"  {'Commands per step':.<40} {command_time/num_steps*1000:.3f} ms")
  print(f"  {'Observations per step':.<40} {obs_time/num_steps*1000:.3f} ms")
  print(f"  {'Rewards per step':.<40} {reward_time/num_steps*1000:.3f} ms")
  print(f"  {'Terminations per step':.<40} {term_time/num_steps*1000:.3f} ms")

  print()
  env_steps = num_steps * num_envs
  print("Throughput:")
  print(f"  {'Env-steps/second':.<40} {env_steps/total_time:,.0f}")
  print(f"  {'Million env-steps/second':.<40} {env_steps/total_time/1e6:.2f}")
  print()
  
  print("Total time breakdown:")
  print(f"  {'Total loop time':.<40} {total_time*1000:.2f} ms")
  print(f"  {'Physics simulation':.<40} {sim_time*1000:.2f} ms ({sim_time/total_time*100:.1f}%)")
  print(f"  {'Observation computation':.<40} {obs_time*1000:.2f} ms ({obs_time/total_time*100:.1f}%)")
  print(f"  {'Reward computation':.<40} {reward_time*1000:.2f} ms ({reward_time/total_time*100:.1f}%)")
  print(f"  {'Termination computation':.<40} {term_time*1000:.2f} ms ({term_time/total_time*100:.1f}%)")
  print(f"  {'Reset operations':.<40} {reset_time*1000:.2f} ms ({reset_time/total_time*100:.1f}%)")
  print()
  
  print("Per-step averages:")
  print(f"  {'Total per step':.<40} {total_time/num_steps*1000:.3f} ms")
  print(f"  {'Physics per step':.<40} {sim_time/num_steps*1000:.3f} ms")
  print(f"  {'Observations per step':.<40} {obs_time/num_steps*1000:.3f} ms")
  print(f"  {'Rewards per step':.<40} {reward_time/num_steps*1000:.3f} ms")
  print(f"  {'Terminations per step':.<40} {term_time/num_steps*1000:.3f} ms")
  print()
  
  env_steps = num_steps * num_envs
  print("Throughput:")
  print(f"  {'Env-steps/second':.<40} {env_steps/total_time:,.0f}")
  print(f"  {'Million env-steps/second':.<40} {env_steps/total_time/1e6:.2f}")
  print()
  
  print(f"{'='*70}\n")


def profile_reward_terms(task_name: str, num_envs: int, num_steps: int = 100):
  """Profile individual reward terms to identify expensive ones."""
  
  print(f"\n{'='*70}")
  print(f"Profiling individual reward terms: {task_name}")
  print(f"{'='*70}\n")
  
  # Create environment
  env_cfg = load_env_cfg(task_name)
  env_cfg.scene.num_envs = num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda")
  
  # Warmup
  env.reset()
  num_actions = env.single_action_space.shape[0]
  for _ in range(10):
    actions = torch.randn(num_envs, num_actions, device="cuda").clamp(-1, 1)
    env.step(actions)
  torch.cuda.synchronize()
  
  # Profile each reward term
  reward_times = {}
  
  for reward_name in env.reward_manager.active_terms:
    term_cfg = env.reward_manager.get_term_cfg(reward_name)
    
    term_accum = 0.0
    
    for _ in range(num_steps):
      # physics step (decimation)
      for _ in range(env.cfg.decimation):
        env.sim.step()
      
      torch.cuda.synchronize()
      t0 = time.perf_counter()
      # compute only this term function (exclude weight/dt)
      _ = term_cfg.func(env, **term_cfg.params)
      torch.cuda.synchronize()
      term_accum += time.perf_counter() - t0
    
    reward_times[reward_name] = term_accum / num_steps * 1000  # ms per step
  
  # Sort by time
  sorted_rewards = sorted(reward_times.items(), key=lambda x: x[1], reverse=True)
  
  print("Reward term computation times (per step):\n")
  total_reward_time = sum(reward_times.values())
  for name, time_ms in sorted_rewards:
    weight = env.reward_manager.get_term_cfg(name).weight
    print(f"  {name:.<35} {time_ms:.4f} ms ({time_ms/total_reward_time*100:4.1f}%) [weight={weight:.1f}]")
  
  print(f"\n  {'TOTAL':.<35} {total_reward_time:.4f} ms\n")
  print(f"{'='*70}\n")


def main():
  parser = argparse.ArgumentParser(description="Profile mjlab environment performance")
  parser.add_argument(
    "--task",
    type=str,
    default="Mjlab-Velocity-Flat-Unitree-G1",
    help="Task name (default: Mjlab-Velocity-Flat-Unitree-G1)",
  )
  parser.add_argument(
    "--num-envs",
    type=int,
    default=4096,
    help="Number of parallel environments (default: 4096)",
  )
  parser.add_argument(
    "--num-steps",
    type=int,
    default=1000,
    help="Number of steps to profile (default: 1000)",
  )
  parser.add_argument(
    "--profile-rewards",
    action="store_true",
    help="Profile individual reward terms",
  )
  parser.add_argument(
    "--iterations",
    type=int,
    default=None,
    help="Override MuJoCo solver iterations (e.g., 10)",
  )
  parser.add_argument(
    "--ls-iterations",
    type=int,
    default=None,
    help="Override MuJoCo line-search iterations (e.g., 10)",
  )
  parser.add_argument(
    "--solver",
    type=str,
    choices=["newton","pgs","cg"],
    default=None,
    help="Override MuJoCo solver (newton/pgs/cg)",
  )
  parser.add_argument(
    "--decimation",
    type=int,
    default=1,
    help="Override environment decimation (physics steps per env-step)",
  )
  
  args = parser.parse_args()
  
  # Apply overrides if provided
  if any(v is not None for v in (args.iterations, args.ls_iterations, args.solver, args.decimation)):
    env_cfg = load_env_cfg(args.task)
    if args.iterations is not None:
      env_cfg.sim.mujoco.iterations = args.iterations
    if args.ls_iterations is not None:
      env_cfg.sim.mujoco.ls_iterations = args.ls_iterations
    if args.solver is not None:
      env_cfg.sim.mujoco.solver = args.solver  # type: ignore
    if args.decimation is not None:
      env_cfg.decimation = args.decimation
    profile_environment_with_cfg(env_cfg, args.num_envs, args.num_steps)
  else:
    # Main profiling
    profile_environment(args.task, args.num_envs, args.num_steps)
  
  # Reward term profiling
  if args.profile_rewards:
    profile_reward_terms(args.task, args.num_envs, num_steps=100)


if __name__ == "__main__":
  main()
