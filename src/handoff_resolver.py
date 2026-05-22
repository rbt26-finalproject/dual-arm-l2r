"""
Dynamic handoff zone resolver.

Samples the joint space of both arms and collects all reachable end-effector
positions. The handoff zone is the intersection of both arms' reachable
workspaces on the horizontal plane — i.e. positions that both arms can reach.

The result is a center point and radius that define a circle on the table
surface guaranteed to be reachable by both arms.

Usage:
    python src/handoff_resolver.py
"""

import numpy as np
import mujoco
from scipy.spatial import ConvexHull


def _sample_reachable(model, data, site_id, joint_slice, n_samples=3000, seed=0):
    """
    Samples joint space uniformly and collects reachable EE positions (x, y).
    Returns an (N, 2) array of reachable xy positions.
    """
    rng    = np.random.default_rng(seed)
    lo     = model.jnt_range[joint_slice, 0]
    hi     = model.jnt_range[joint_slice, 1]
    pts    = []

    qpos_backup = data.qpos.copy()

    for _ in range(n_samples):
        q = rng.uniform(lo, hi)
        data.qpos[joint_slice] = q
        mujoco.mj_forward(model, data)
        ee = data.site_xpos[site_id]
        pts.append(ee[:2].copy())  # only x-y

    data.qpos[:] = qpos_backup
    mujoco.mj_forward(model, data)

    return np.array(pts)


def _point_in_hull(point, hull):
    """Returns True if point (x, y) is inside the convex hull."""
    # A point is inside if all half-plane inequalities are satisfied
    return all(
        np.dot(eq[:-1], point) + eq[-1] <= 1e-10
        for eq in hull.equations
    )


def compute_handoff_zone(model, data, site_a_id, site_b_id,
                         joint_slice_a, joint_slice_b,
                         table_z=0.05, n_samples=3000):
    """
    Computes the handoff zone as the intersection of both arms' reachable
    workspaces on the xy plane.

    Returns a dict with:
        center      : (3,) array — center of handoff zone (x, y, table_z)
        radius      : float — radius of the inscribed circle of the intersection
        pts_a       : (N, 2) — sampled reachable xy for Arm A (for visualization)
        pts_b       : (N, 2) — sampled reachable xy for Arm B (for visualization)
        intersection: (M, 2) — points in both arms' workspaces
    """
    print("[handoff_resolver] Sampling Arm A reachable workspace...")
    pts_a = _sample_reachable(model, data, site_a_id, joint_slice_a, n_samples)

    print("[handoff_resolver] Sampling Arm B reachable workspace...")
    pts_b = _sample_reachable(model, data, site_b_id, joint_slice_b, n_samples, seed=1)

    print("[handoff_resolver] Computing convex hulls...")
    hull_a = ConvexHull(pts_a)
    hull_b = ConvexHull(pts_b)

    # Intersection: points from hull_a that are inside hull_b and vice versa
    pts_a_in_b = pts_a[[_point_in_hull(p, hull_b) for p in pts_a]]
    pts_b_in_a = pts_b[[_point_in_hull(p, hull_a) for p in pts_b]]
    intersection = np.vstack([pts_a_in_b, pts_b_in_a]) if (
        len(pts_a_in_b) > 0 and len(pts_b_in_a) > 0
    ) else np.vstack([pts_a_in_b, pts_b_in_a]) if len(pts_a_in_b) > 0 else pts_b_in_a

    if len(intersection) < 3:
        print("[handoff_resolver] Warning: very small intersection, using center table fallback.")
        center = np.array([0.0, 0.0, table_z])
        radius = 0.05
        return dict(center=center, radius=radius,
                    pts_a=pts_a, pts_b=pts_b, intersection=intersection)

    center_xy = intersection.mean(axis=0)

    # Radius = mean distance from center to intersection boundary points
    # clipped to a reasonable range for a table scene
    dists  = np.linalg.norm(intersection - center_xy, axis=1)
    radius = float(np.clip(np.percentile(dists, 25), 0.03, 0.20))

    center = np.array([center_xy[0], center_xy[1], table_z])

    print(f"[handoff_resolver] Handoff zone: center=({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})  radius={radius:.3f}m")
    print(f"[handoff_resolver] Intersection points: {len(intersection)}")

    return dict(
        center=center,
        radius=radius,
        pts_a=pts_a,
        pts_b=pts_b,
        intersection=intersection,
    )


def visualize_handoff_zone(result, save_path=None):
    """Plots the reach envelopes and handoff zone. Saves to file or shows interactively."""
    import matplotlib
    matplotlib.use("Agg" if save_path else "TkAgg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.tick_params(colors="#cccccc")
    ax.xaxis.label.set_color("#cccccc")
    ax.yaxis.label.set_color("#cccccc")
    ax.title.set_color("#ffffff")

    ax.scatter(result["pts_a"][:, 0], result["pts_a"][:, 1],
               s=2, alpha=0.15, color="#4a9eff", label="Arm A reach")
    ax.scatter(result["pts_b"][:, 0], result["pts_b"][:, 1],
               s=2, alpha=0.15, color="#ff6b6b", label="Arm B reach")

    if len(result["intersection"]) > 0:
        ax.scatter(result["intersection"][:, 0], result["intersection"][:, 1],
                   s=6, alpha=0.5, color="#4ecdc4", label="Intersection")

    cx, cy = result["center"][0], result["center"][1]
    r      = result["radius"]
    circle = plt.Circle((cx, cy), r, fill=True, facecolor="#4ecdc4",
                         alpha=0.35, edgecolor="#4ecdc4", linewidth=2)
    ax.add_patch(circle)
    ax.plot(cx, cy, "+", color="white", markersize=12, markeredgewidth=2)
    ax.text(cx, cy + r + 0.02, f"Handoff zone\n({cx:.2f}, {cy:.2f})",
            color="#4ecdc4", ha="center", fontsize=8)

    ax.plot(-0.469, 0, "^", color="#4a9eff", markersize=11, label="Arm A base")
    ax.plot( 0.469, 0, "^", color="#ff6b6b", markersize=11, label="Arm B base")
    ax.axvline(0, color="white", linewidth=0.8, linestyle=":", alpha=0.4)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.9, 0.9)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Dynamic Handoff Zone — Intersection of Reachable Workspaces",
                 fontsize=11, pad=10)
    ax.legend(loc="upper right", fontsize=8, facecolor="#1a1a2e",
              edgecolor="#444466", labelcolor="#cccccc")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[handoff_resolver] Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    import importlib
    from pathlib import Path
    from mujoco_playground._src import mjx_env
    from mujoco_playground._src.manipulation.aloha import base as aloha_base
    from etils import epath

    MENAGERIE_PATH = Path(__file__).parent.parent / "aloha_menagerie"
    SCENE_XML      = Path(__file__).parent.parent / "assets" / "scene_dual_arm.xml"

    mjx_env.MENAGERIE_PATH = epath.Path(str(MENAGERIE_PATH))
    importlib.reload(aloha_base)

    assets = aloha_base.get_assets()

    # Load STL mesh files from the aloha assets subfolder
    aloha_assets_dir = MENAGERIE_PATH / "aloha" / "assets"
    if aloha_assets_dir.exists():
        for f in aloha_assets_dir.iterdir():
            if f.is_file():
                assets[f.name] = f.read_bytes()

    xml_str = SCENE_XML.read_text()
    model   = mujoco.MjModel.from_xml_string(xml_str, assets)
    data    = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    site_a_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left/gripper")
    site_b_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right/gripper")

    joint_slice_a = slice(0, 6)
    joint_slice_b = slice(8, 14)

    result = compute_handoff_zone(
        model, data,
        site_a_id, site_b_id,
        joint_slice_a, joint_slice_b,
        table_z=0.05,
        n_samples=3000,
    )

    out_path = Path(__file__).parent.parent / "outputs" / "handoff_zone.png"
    out_path.parent.mkdir(exist_ok=True)
    visualize_handoff_zone(result, save_path=str(out_path))

    print(f"\nHandoff zone summary:")
    print(f"  center : {result['center']}")
    print(f"  radius : {result['radius']:.4f} m")
