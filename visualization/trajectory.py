#!/usr/bin/env python3

"""
Plot the Particle Filter estimated trajectory from a ROS2 rosbag.

The particle filter publishes its *mean pose* as `nav_msgs/msg/Odometry` on
`/pf/pose/odom` (see `localization/localization/particle_filter.py`). Internally,
particles are stored as an `Nx3` numpy array `[x, y, theta]`, and the node
publishes the average as the Odometry pose.

This script reads `/pf/pose/odom` from a rosbag (rosbag2 sqlite `.db3` directory)
and saves matplotlib plots (PNG) to disk.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    # ROS uses geometry_msgs/Quaternion with (x, y, z, w).
    # Yaw from quaternion (assuming ENU + Z-up).
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _norm_topic(topic: str) -> str:
    topic = topic.strip()
    if not topic.startswith("/"):
        topic = "/" + topic
    return topic


@dataclass(frozen=True)
class Trajectory:
    bag: Path
    topic: str
    t_sec: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray


def _coerce_bag_dir(path: Path) -> Path:
    """Accept either a rosbag2 directory or a `*.db3` file path."""
    if path.is_file() and path.suffix == ".db3":
        return path.parent
    return path


def _read_trajectory_rosbags(bag_dir: Path, topic: str) -> Trajectory:
    try:
        from rosbags.highlevel import AnyReader  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover
        # Fallback to rosbag2_py (ships with many ROS2 installs) so the script
        # still works when `rosbags` isn't pip-installed.
        return _read_trajectory_rosbag2_py(bag_dir, topic)

    topic = _norm_topic(topic)

    xs: list[float] = []
    ys: list[float] = []
    yaws: list[float] = []
    ts: list[int] = []

    bag_dir = _coerce_bag_dir(bag_dir)
    if not bag_dir.exists():
        raise FileNotFoundError(f"Bag path does not exist: {bag_dir}")

    with AnyReader([bag_dir]) as reader:
        # Match both exact and non-leading-slash topics just in case.
        conns = [
            c
            for c in reader.connections
            if _norm_topic(c.topic) == topic or c.topic == topic
        ]
        if not conns:
            available = sorted({_norm_topic(c.topic) for c in reader.connections})
            raise ValueError(
                f"Topic {topic} not found in bag {bag_dir}.\n"
                f"Available topics: {available}"
            )

        for connection, timestamp, rawdata in reader.messages(connections=conns):
            # rosbags versions differ: prefer reader.deserialize when available.
            if hasattr(reader, "deserialize"):
                msg = reader.deserialize(rawdata, connection.msgtype)
            else:  # pragma: no cover
                from rosbags.serde import deserialize_cdr  # type: ignore

                msg = deserialize_cdr(rawdata, connection.msgtype)

            pose = msg.pose.pose
            xs.append(float(pose.position.x))
            ys.append(float(pose.position.y))
            q = pose.orientation
            yaws.append(_quat_to_yaw(float(q.x), float(q.y), float(q.z), float(q.w)))
            ts.append(int(timestamp))

    if not ts:
        raise ValueError(f"No messages read for topic {topic} in bag {bag_dir}")

    t0 = ts[0]
    t_sec = (np.asarray(ts, dtype=np.int64) - t0) / 1e9
    return Trajectory(
        bag=bag_dir,
        topic=topic,
        t_sec=t_sec.astype(np.float64),
        x=np.asarray(xs, dtype=np.float64),
        y=np.asarray(ys, dtype=np.float64),
        yaw=np.asarray(yaws, dtype=np.float64),
    )


def _read_trajectory_rosbag2_py(bag_dir: Path, topic: str) -> Trajectory:
    try:
        from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions  # type: ignore
        from rclpy.serialization import deserialize_message  # type: ignore
        from rosidl_runtime_py.utilities import get_message  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Neither `rosbags` nor `rosbag2_py` is available to read rosbag2 files.\n"
            "Fix options:\n"
            "- Install rosbags: `pip install rosbags`\n"
            "- Or run inside a ROS2 environment that provides `rosbag2_py`."
        ) from exc

    topic = _norm_topic(topic)
    bag_dir = _coerce_bag_dir(bag_dir)
    if not bag_dir.exists():
        raise FileNotFoundError(f"Bag path does not exist: {bag_dir}")

    reader = SequentialReader()
    storage_options = StorageOptions(uri=str(bag_dir), storage_id="sqlite3")
    converter_options = ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
    reader.open(storage_options, converter_options)

    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    # Try both normalized and raw forms.
    msgtype = topic_types.get(topic) or topic_types.get(topic.lstrip("/"))
    if msgtype is None:
        available = sorted({_norm_topic(k) for k in topic_types.keys()})
        raise ValueError(
            f"Topic {topic} not found in bag {bag_dir}.\n"
            f"Available topics: {available}"
        )

    msgcls = get_message(msgtype)

    xs: list[float] = []
    ys: list[float] = []
    yaws: list[float] = []
    ts: list[int] = []

    while reader.has_next():
        topic_name, data, timestamp = reader.read_next()
        if _norm_topic(topic_name) != topic and topic_name != topic:
            continue
        msg = deserialize_message(data, msgcls)
        pose = msg.pose.pose
        xs.append(float(pose.position.x))
        ys.append(float(pose.position.y))
        q = pose.orientation
        yaws.append(_quat_to_yaw(float(q.x), float(q.y), float(q.z), float(q.w)))
        ts.append(int(timestamp))

    if not ts:
        raise ValueError(f"No messages read for topic {topic} in bag {bag_dir}")

    t0 = ts[0]
    t_sec = (np.asarray(ts, dtype=np.int64) - t0) / 1e9
    return Trajectory(
        bag=bag_dir,
        topic=topic,
        t_sec=t_sec.astype(np.float64),
        x=np.asarray(xs, dtype=np.float64),
        y=np.asarray(ys, dtype=np.float64),
        yaw=np.asarray(yaws, dtype=np.float64),
    )


def _downsample(traj: Trajectory, stride: int) -> Trajectory:
    stride = max(1, int(stride))
    if stride == 1:
        return traj
    sl = slice(None, None, stride)
    return Trajectory(
        bag=traj.bag,
        topic=traj.topic,
        t_sec=traj.t_sec[sl],
        x=traj.x[sl],
        y=traj.y[sl],
        yaw=traj.yaw[sl],
    )


def _save_plots(
    traj: Trajectory,
    out_dir: Path,
    *,
    points_only: bool,
    arrows: bool,
    arrow_step: int,
    dpi: int,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")  # headless save
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    stem = traj.bag.name.replace(" ", "_")
    topic_tag = traj.topic.strip("/").replace("/", "_")
    xy_path = out_dir / f"{stem}__{topic_tag}__xy.png"
    yaw_path = out_dir / f"{stem}__{topic_tag}__yaw.png"

    # XY trajectory
    fig, ax = plt.subplots(figsize=(7, 7), dpi=dpi)
    if points_only:
        ax.scatter(traj.x, traj.y, s=6, color="tab:blue")
    else:
        ax.plot(traj.x, traj.y, linewidth=1.5, color="tab:blue")
    ax.scatter([traj.x[0]], [traj.y[0]], s=40, color="tab:green", label="start", zorder=3)
    ax.scatter([traj.x[-1]], [traj.y[-1]], s=40, color="tab:red", label="end", zorder=3)
    if arrows and len(traj.x) > 1:
        step = max(1, int(arrow_step))
        idx = np.arange(0, len(traj.x), step, dtype=int)
        u = np.cos(traj.yaw[idx])
        v = np.sin(traj.yaw[idx])
        ax.quiver(
            traj.x[idx],
            traj.y[idx],
            u,
            v,
            angles="xy",
            scale_units="xy",
            scale=4.0,
            width=0.003,
            alpha=0.7,
            color="tab:orange",
            zorder=2,
        )
    ax.set_title(f"Trajectory - Main Hallway of Stata")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(xy_path)
    plt.close(fig)

    # Yaw vs time
    fig, ax = plt.subplots(figsize=(10, 4), dpi=dpi)
    ax.plot(traj.t_sec, traj.yaw, linewidth=1.0, color="tab:purple")
    ax.set_title(f"Yaw over time: {traj.topic}\n{traj.bag}")
    ax.set_xlabel("t [s] (relative)")
    ax.set_ylabel("yaw [rad]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(yaw_path)
    plt.close(fig)

    return [xy_path, yaw_path]


def _iter_bag_paths(paths: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        out.append(_coerce_bag_dir(Path(p).expanduser()).resolve())
    return out


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Plot /pf/pose/odom trajectory from rosbag2.")
    parser.add_argument(
        "bags",
        nargs="+",
        help="Rosbag2 directory (contains metadata.yaml) or a *.db3 file.",
    )
    parser.add_argument("--topic", default="/pf/pose/odom", help="Topic to plot (default: /pf/pose/odom).")
    parser.add_argument("--out", default="trajectory_plots", help="Output directory for saved plots.")
    parser.add_argument("--stride", type=int, default=1, help="Downsample messages by taking every Nth point.")
    parser.add_argument("--points-only", action="store_true", help="Plot points without connecting lines.")
    parser.add_argument("--arrows", action="store_true", help="Draw yaw direction arrows on the XY plot.")
    parser.add_argument("--arrow-step", type=int, default=25, help="Arrow stride for --arrows.")
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI.")
    args = parser.parse_args(argv)

    out_dir = Path(args.out).expanduser().resolve()
    bag_dirs = _iter_bag_paths(args.bags)

    for bag_dir in bag_dirs:
        traj = _read_trajectory_rosbags(bag_dir, args.topic)
        traj = _downsample(traj, args.stride)
        _save_plots(
            traj,
            out_dir,
            points_only=args.points_only,
            arrows=args.arrows,
            arrow_step=args.arrow_step,
            dpi=args.dpi,
        )

    print(f"Saved plots to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
