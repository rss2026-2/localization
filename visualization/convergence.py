#!/usr/bin/env python3

"""
Compute and plot particle-filter convergence metrics from a ROS2 rosbag.

This script expects a `geometry_msgs/msg/PoseArray` topic (default: `/particles`)
as published by `localization/localization/particle_filter.py`. It measures
convergence via particle spread over time:

- std_x, std_y: standard deviation of particle positions
- std_yaw: circular standard deviation of particle yaw
- pos_std: sqrt(std_x^2 + std_y^2)

It can also estimate "convergence time per sensor update" by using `/scan`
timestamps as sensor-update events (since the PF's sensor model runs in the scan
callback). For each scan time, it looks at particle messages until the next scan
and finds the time to the minimum spread in that interval.

Outputs matplotlib PNG plots into an output directory.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


def _norm_topic(topic: str) -> str:
    topic = topic.strip()
    if not topic.startswith("/"):
        topic = "/" + topic
    return topic


def _coerce_bag_dir(path: Path) -> Path:
    """Accept either a rosbag2 directory or a `*.db3` file path."""
    if path.is_file() and path.suffix == ".db3":
        return path.parent
    return path


def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _circular_std(angles: np.ndarray) -> float:
    """
    Circular standard deviation for angles in radians.

    Uses: std = sqrt(-2 ln R), where R is mean resultant length.
    """
    if angles.size == 0:
        return float("nan")
    s = np.sin(angles).mean()
    c = np.cos(angles).mean()
    r = float(np.hypot(s, c))
    r = max(min(r, 1.0), 1e-12)
    return float(math.sqrt(-2.0 * math.log(r)))


@dataclass(frozen=True)
class ConvergenceSeries:
    bag: Path
    topic: str
    t_ns: np.ndarray
    std_x: np.ndarray
    std_y: np.ndarray
    std_yaw: np.ndarray

    @property
    def t_sec(self) -> np.ndarray:
        t0 = self.t_ns[0]
        return (self.t_ns.astype(np.int64) - int(t0)) / 1e9

    @property
    def pos_std(self) -> np.ndarray:
        return np.hypot(self.std_x, self.std_y)


def _read_posearray_rosbags(bag_dir: Path, topic: str, stride: int) -> ConvergenceSeries:
    topic = _norm_topic(topic)
    bag_dir = _coerce_bag_dir(bag_dir)
    stride = max(1, int(stride))
    if not bag_dir.exists():
        raise FileNotFoundError(f"Bag path does not exist: {bag_dir}")

    try:
        from rosbags.highlevel import AnyReader  # type: ignore
    except ModuleNotFoundError:
        return _read_posearray_rosbag2_py(bag_dir, topic, stride)

    ts: list[int] = []
    std_x: list[float] = []
    std_y: list[float] = []
    std_yaw: list[float] = []

    with AnyReader([bag_dir]) as reader:
        conns = [c for c in reader.connections if _norm_topic(c.topic) == topic or c.topic == topic]
        if not conns:
            available = sorted({_norm_topic(c.topic) for c in reader.connections})
            raise ValueError(
                f"Topic {topic} not found in bag {bag_dir}.\n"
                f"Available topics: {available}"
            )

        msg_idx = 0
        for connection, timestamp, rawdata in reader.messages(connections=conns):
            msg_idx += 1
            if (msg_idx - 1) % stride != 0:
                continue

            if hasattr(reader, "deserialize"):
                msg = reader.deserialize(rawdata, connection.msgtype)
            else:  # pragma: no cover
                from rosbags.serde import deserialize_cdr  # type: ignore

                msg = deserialize_cdr(rawdata, connection.msgtype)

            if not getattr(msg, "poses", None):
                continue

            xs = np.fromiter((float(p.position.x) for p in msg.poses), dtype=np.float64)
            ys = np.fromiter((float(p.position.y) for p in msg.poses), dtype=np.float64)
            yaws = np.fromiter(
                (
                    _quat_to_yaw(float(p.orientation.x), float(p.orientation.y), float(p.orientation.z), float(p.orientation.w))
                    for p in msg.poses
                ),
                dtype=np.float64,
            )

            ts.append(int(timestamp))
            std_x.append(float(xs.std(ddof=0)))
            std_y.append(float(ys.std(ddof=0)))
            std_yaw.append(_circular_std(yaws))

    if not ts:
        raise ValueError(f"No messages read for topic {topic} in bag {bag_dir}")

    return ConvergenceSeries(
        bag=bag_dir,
        topic=topic,
        t_ns=np.asarray(ts, dtype=np.int64),
        std_x=np.asarray(std_x, dtype=np.float64),
        std_y=np.asarray(std_y, dtype=np.float64),
        std_yaw=np.asarray(std_yaw, dtype=np.float64),
    )


def _read_posearray_rosbag2_py(bag_dir: Path, topic: str, stride: int) -> ConvergenceSeries:
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
    stride = max(1, int(stride))

    reader = SequentialReader()
    storage_options = StorageOptions(uri=str(bag_dir), storage_id="sqlite3")
    converter_options = ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
    reader.open(storage_options, converter_options)

    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    msgtype = topic_types.get(topic) or topic_types.get(topic.lstrip("/"))
    if msgtype is None:
        available = sorted({_norm_topic(k) for k in topic_types.keys()})
        raise ValueError(
            f"Topic {topic} not found in bag {bag_dir}.\n"
            f"Available topics: {available}"
        )

    msgcls = get_message(msgtype)

    ts: list[int] = []
    std_x: list[float] = []
    std_y: list[float] = []
    std_yaw: list[float] = []

    msg_idx = 0
    while reader.has_next():
        topic_name, data, timestamp = reader.read_next()
        if _norm_topic(topic_name) != topic and topic_name != topic:
            continue
        msg_idx += 1
        if (msg_idx - 1) % stride != 0:
            continue
        msg = deserialize_message(data, msgcls)
        poses = getattr(msg, "poses", None)
        if not poses:
            continue
        xs = np.fromiter((float(p.position.x) for p in poses), dtype=np.float64)
        ys = np.fromiter((float(p.position.y) for p in poses), dtype=np.float64)
        yaws = np.fromiter(
            (_quat_to_yaw(float(p.orientation.x), float(p.orientation.y), float(p.orientation.z), float(p.orientation.w)) for p in poses),
            dtype=np.float64,
        )

        ts.append(int(timestamp))
        std_x.append(float(xs.std(ddof=0)))
        std_y.append(float(ys.std(ddof=0)))
        std_yaw.append(_circular_std(yaws))

    if not ts:
        raise ValueError(f"No messages read for topic {topic} in bag {bag_dir}")

    return ConvergenceSeries(
        bag=bag_dir,
        topic=topic,
        t_ns=np.asarray(ts, dtype=np.int64),
        std_x=np.asarray(std_x, dtype=np.float64),
        std_y=np.asarray(std_y, dtype=np.float64),
        std_yaw=np.asarray(std_yaw, dtype=np.float64),
    )


def _read_topic_timestamps(bag_dir: Path, topic: str, stride: int) -> np.ndarray:
    """
    Read only timestamps for a topic (fast path used for `/scan` event times).
    """
    topic = _norm_topic(topic)
    bag_dir = _coerce_bag_dir(bag_dir)
    stride = max(1, int(stride))

    try:
        from rosbags.highlevel import AnyReader  # type: ignore
    except ModuleNotFoundError:
        return _read_topic_timestamps_rosbag2_py(bag_dir, topic, stride)

    ts: list[int] = []
    with AnyReader([bag_dir]) as reader:
        conns = [c for c in reader.connections if _norm_topic(c.topic) == topic or c.topic == topic]
        if not conns:
            available = sorted({_norm_topic(c.topic) for c in reader.connections})
            raise ValueError(
                f"Topic {topic} not found in bag {bag_dir}.\n"
                f"Available topics: {available}"
            )

        msg_idx = 0
        for _connection, timestamp, _rawdata in reader.messages(connections=conns):
            msg_idx += 1
            if (msg_idx - 1) % stride != 0:
                continue
            ts.append(int(timestamp))

    if not ts:
        raise ValueError(f"No messages read for topic {topic} in bag {bag_dir}")
    return np.asarray(ts, dtype=np.int64)


def _read_topic_timestamps_rosbag2_py(bag_dir: Path, topic: str, stride: int) -> np.ndarray:
    try:
        from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Neither `rosbags` nor `rosbag2_py` is available to read rosbag2 files.\n"
            "Fix options:\n"
            "- Install rosbags: `pip install rosbags`\n"
            "- Or run inside a ROS2 environment that provides `rosbag2_py`."
        ) from exc

    reader = SequentialReader()
    storage_options = StorageOptions(uri=str(bag_dir), storage_id="sqlite3")
    converter_options = ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
    reader.open(storage_options, converter_options)

    ts: list[int] = []
    msg_idx = 0
    while reader.has_next():
        topic_name, _data, timestamp = reader.read_next()
        if _norm_topic(topic_name) != topic and topic_name != topic:
            continue
        msg_idx += 1
        if (msg_idx - 1) % stride != 0:
            continue
        ts.append(int(timestamp))

    if not ts:
        raise ValueError(f"No messages read for topic {topic} in bag {bag_dir}")
    return np.asarray(ts, dtype=np.int64)


def _metric(series: ConvergenceSeries, name: str) -> np.ndarray:
    if name == "pos_std":
        return series.pos_std
    if name == "std_x":
        return series.std_x
    if name == "std_y":
        return series.std_y
    if name == "std_yaw":
        return series.std_yaw
    raise ValueError(f"Unknown metric: {name}")


def _convergence_times_from_scan(
    series: ConvergenceSeries,
    scan_t_ns: np.ndarray,
    *,
    metric: str,
) -> tuple[np.ndarray, float]:
    """
    For each scan time, compute convergence time as:
      time from scan to the minimum spread among particle messages until next scan.
    """
    scan_t_ns = np.asarray(scan_t_ns, dtype=np.int64)
    scan_t_ns = np.unique(scan_t_ns)
    scan_t_ns.sort()

    particle_t = series.t_ns.astype(np.int64)
    values = _metric(series, metric)

    conv_times: list[float] = []
    for i, t0 in enumerate(scan_t_ns):
        t1 = scan_t_ns[i + 1] if i + 1 < scan_t_ns.size else np.iinfo(np.int64).max

        mask = (particle_t >= t0) & (particle_t < t1)
        if not np.any(mask):
            continue
        idx = np.nonzero(mask)[0]
        local_vals = values[idx]
        if local_vals.size == 0:
            continue
        j = int(np.argmin(local_vals))
        t_min = int(particle_t[idx[j]])
        conv_times.append((t_min - int(t0)) / 1e9)

    if not conv_times:
        return np.asarray([], dtype=np.float64), float("nan")
    conv_arr = np.asarray(conv_times, dtype=np.float64)
    return conv_arr, float(np.mean(conv_arr))


def _save_plots(series: ConvergenceSeries, out_dir: Path, *, dpi: int) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = series.bag.name.replace(" ", "_")
    topic_tag = series.topic.strip("/").replace("/", "_")

    spread_path = out_dir / f"{stem}__{topic_tag}__spread.png"
    yaw_path = out_dir / f"{stem}__{topic_tag}__yaw_std.png"

    fig, ax = plt.subplots(figsize=(10, 4), dpi=dpi)
    ax.plot(series.t_sec, series.std_x, label="std_x [m]", linewidth=1.2)
    ax.plot(series.t_sec, series.std_y, label="std_y [m]", linewidth=1.2)
    ax.plot(series.t_sec, series.pos_std, label="pos_std [m]", linewidth=1.8)
    ax.set_title(f"Particle spread (convergence): {series.topic}\n{series.bag}")
    ax.set_xlabel("t [s] (relative)")
    ax.set_ylabel("std [m]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(spread_path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4), dpi=dpi)
    ax.plot(series.t_sec, series.std_yaw, label="std_yaw [rad]", linewidth=1.4, color="tab:purple")
    ax.set_title(f"Yaw spread (convergence): {series.topic}")
    ax.set_xlabel("t [s] (relative)")
    ax.set_ylabel("circular std [rad]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(yaw_path)
    plt.close(fig)

    return [spread_path, yaw_path]


def _save_convergence_time_plot(
    series: ConvergenceSeries,
    scan_t_ns: np.ndarray,
    conv_times: np.ndarray,
    avg_conv_time: float,
    out_dir: Path,
    *,
    metric: str,
    scan_stride: int,
    dpi: int,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = series.bag.name.replace(" ", "_")
    topic_tag = series.topic.strip("/").replace("/", "_")
    out_path = out_dir / f"{stem}__{topic_tag}__conv_time__{metric}__scan_stride{scan_stride}.png"

    if conv_times.size == 0:
        fig, ax = plt.subplots(figsize=(10, 4), dpi=dpi)
        ax.set_title("Convergence time: no matching particle windows found")
        ax.set_xlabel("t [s] (relative)")
        ax.set_ylabel("convergence time [s]")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    scan_t_ns = np.asarray(scan_t_ns, dtype=np.int64)
    scan_t_ns = np.unique(scan_t_ns)
    scan_t_ns.sort()

    # Align x-axis to first particle message.
    t0 = int(series.t_ns[0])
    # conv_times corresponds to scans for which we found a particle window; compute x for those scans.
    # We recompute the same filtering used in _convergence_times_from_scan to keep it consistent.
    xs: list[float] = []
    particle_t = series.t_ns.astype(np.int64)
    values = _metric(series, metric)
    k = 0
    for i, t_scan in enumerate(scan_t_ns):
        if k >= conv_times.size:
            break
        t_next = scan_t_ns[i + 1] if i + 1 < scan_t_ns.size else np.iinfo(np.int64).max
        mask = (particle_t >= t_scan) & (particle_t < t_next)
        if not np.any(mask):
            continue
        # This scan contributed one entry to conv_times.
        xs.append((int(t_scan) - t0) / 1e9)
        k += 1

    x = np.asarray(xs, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10, 4), dpi=dpi)
    ax.plot(x, conv_times, ".", markersize=4, alpha=0.8, label="per-scan convergence time")
    if np.isfinite(avg_conv_time):
        ax.axhline(avg_conv_time, color="tab:red", linewidth=1.5, label=f"mean = {avg_conv_time:.3f}s")
    ax.set_title(f"Convergence time per scan (metric={metric})")
    ax.set_xlabel("t [s] (relative to first particle msg)")
    ax.set_ylabel("time-to-min-spread [s]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Plot particle-filter convergence metrics from rosbag2.")
    parser.add_argument("bag", help="Rosbag2 directory (contains metadata.yaml) or a *.db3 file.")
    parser.add_argument("--topic", default="/particles", help="PoseArray topic to read (default: /particles).")
    parser.add_argument("--scan-topic", default="/scan", help="LaserScan topic used as sensor-update events.")
    parser.add_argument("--out", default="convergence_plots", help="Output directory for saved plots.")
    parser.add_argument("--stride", type=int, default=1, help="Downsample messages by taking every Nth PoseArray.")
    parser.add_argument("--scan-stride", type=int, default=1, help="Downsample scan events by taking every Nth scan.")
    parser.add_argument(
        "--conv-metric",
        default="pos_std",
        choices=["pos_std", "std_x", "std_y", "std_yaw"],
        help="Metric used for convergence time computation (default: pos_std).",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI.")
    args = parser.parse_args(argv)

    bag_dir = _coerce_bag_dir(Path(args.bag).expanduser()).resolve()
    out_dir = Path(args.out).expanduser().resolve()

    series = _read_posearray_rosbags(bag_dir, args.topic, args.stride)
    _save_plots(series, out_dir, dpi=args.dpi)

    scan_t_ns = _read_topic_timestamps(bag_dir, args.scan_topic, args.scan_stride)
    conv_times, avg_conv_time = _convergence_times_from_scan(series, scan_t_ns, metric=args.conv_metric)
    _save_convergence_time_plot(
        series,
        scan_t_ns,
        conv_times,
        avg_conv_time,
        out_dir,
        metric=args.conv_metric,
        scan_stride=int(args.scan_stride),
        dpi=args.dpi,
    )

    if np.isfinite(avg_conv_time):
        print(f"Average convergence time (metric={args.conv_metric}): {avg_conv_time:.4f} s")
    print(f"Saved plots to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
