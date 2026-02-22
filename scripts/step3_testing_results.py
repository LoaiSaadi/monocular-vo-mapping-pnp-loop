import numpy as np

traj = np.loadtxt("trajectory.txt")
t = traj[:,0]
xyz = traj[:,1:4]

mins = xyz.min(axis=0)
maxs = xyz.max(axis=0)
print("Trajectory bbox min:", mins)
print("Trajectory bbox max:", maxs)

# duplicate poses (exact match)
dup = np.sum(np.all(np.isclose(xyz[1:], xyz[:-1]), axis=1))
print("Exact duplicate consecutive poses:", dup, "out of", len(xyz)-1)

# step sizes
steps = np.linalg.norm(xyz[1:] - xyz[:-1], axis=1)
print("Step size stats: min/median/max =", steps.min(), np.median(steps), steps.max())
