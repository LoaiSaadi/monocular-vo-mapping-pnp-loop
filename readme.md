# Monocular SLAM – HW3

This project implements a basic **Monocular SLAM pipeline** including:

- Feature extraction (ORB)
- Frame-to-frame motion estimation
- Epipolar geometry
- Manual triangulation (DLT)
- Map point creation
- PnP localization
- Loop closure
- Real-time visualization using Pangolin

---

## Project Structure


├── main.py # Main SLAM pipeline

├── pose.py # Pose class (camera pose representation)

├── frame.py # Frame class (image + features + pose)

├── point.py # 3D map point representation

├── map.py # Map structure storing frames and points

├── pangolin_viewer.py # Pangolin viewer for trajectory + map visualization

├── pangolin.py # Pangolin wrapper script (REQUIRED)

├── scripts/ # Old / helper scripts

└── outputs/ # Generated outputs (trajectory, map, metrics)


- main.py # Main SLAM pipeline
- pose.py # Pose class (camera pose representation)
- frame.py # Frame class (image + features + pose)
- point.py # 3D map point representation
- map.py # Map structure storing frames and points
- pangolin_viewer.py # Pangolin viewer for trajectory + map visualization
- pangolin.py # Pangolin wrapper script (REQUIRED)
- scripts/ # Old / helper scripts
- outputs/ # Generated outputs (trajectory, map, metrics)

---

# Important Note

⚠ **Do NOT remove `pangolin.py`.**

This file is required for **Pangolin visualization**.  
It ensures the Pangolin viewer works correctly with the SLAM pipeline.

Removing it may break the visualization.

---

# Outputs

Running the pipeline generates:

- `outputs/map_points.ply` → reconstructed 3D map
- `outputs/trajectory_corrected.txt` → corrected camera trajectory
- `outputs/metrics.csv` → SLAM evaluation metrics

---

# Visualization

The system visualizes in real time:

- Camera trajectory
- Reconstructed 3D map
- Camera orientation

using **Pangolin**.

---

# Run

```bash
python main.py
