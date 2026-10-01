"""Kinematically-constrained counterfactual pitch control optimization for defensive error attribution."""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    from mplsoccer import Pitch
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from numba import njit
except ImportError:
    def njit(func=None, *args, **kwargs):
        if func is None:
            return lambda f: f
        return func

try:
    from .config import (
        A_MAX,
        DANGER_FLOOR_M,
        DANGER_X_MIN,
        DANGER_Y_MAX,
        DANGER_Y_MIN,
        DYNAMIC_ANGLE_WIDTH,
        DYNAMIC_BASE_MIX,
        GRID_CELLS_X,
        GRID_CELLS_Y,
        HEURISTIC_DECAY,
        HEURISTIC_SCALE,
        LAMBDA,
        MA2024_SCALE,
        PARAMETRIC_CENTRAL_ANGLE,
        PARAMETRIC_CENTRAL_DIST,
        PARAMETRIC_HALF_OFFSET,
        PARAMETRIC_HALF_SPREAD,
        PARAMETRIC_HALF_WEIGHT,
        PARAMETRIC_HALF_WIDTH,
        PARAMETRIC_HALF_X,
        PARAMETRIC_SCALE,
        PARAMETRIC_WIDE_OFFSET,
        PARAMETRIC_WIDE_SPREAD,
        PARAMETRIC_WIDE_WEIGHT,
        PARAMETRIC_WIDE_X,
        PITCH_LENGTH,
        PITCH_WIDTH,
        T_REACT,
        V_MAX,
    )
except (ImportError, ValueError):
    from config import (
        A_MAX,
        DANGER_FLOOR_M,
        DANGER_X_MIN,
        DANGER_Y_MAX,
        DANGER_Y_MIN,
        DYNAMIC_ANGLE_WIDTH,
        DYNAMIC_BASE_MIX,
        GRID_CELLS_X,
        GRID_CELLS_Y,
        HEURISTIC_DECAY,
        HEURISTIC_SCALE,
        LAMBDA,
        MA2024_SCALE,
        PARAMETRIC_CENTRAL_ANGLE,
        PARAMETRIC_CENTRAL_DIST,
        PARAMETRIC_HALF_OFFSET,
        PARAMETRIC_HALF_SPREAD,
        PARAMETRIC_HALF_WEIGHT,
        PARAMETRIC_HALF_WIDTH,
        PARAMETRIC_HALF_X,
        PARAMETRIC_SCALE,
        PARAMETRIC_WIDE_OFFSET,
        PARAMETRIC_WIDE_SPREAD,
        PARAMETRIC_WIDE_WEIGHT,
        PARAMETRIC_WIDE_X,
        PITCH_LENGTH,
        PITCH_WIDTH,
        T_REACT,
        V_MAX,
    )


@njit(cache=True)
def _compute_time_grid_njit(
    X: np.ndarray,
    Y: np.ndarray,
    px: float,
    py: float,
    vx: float,
    vy: float,
    t_react: float,
    v_max: float,
    a_max: float,
    use_kinematics: bool,
) -> np.ndarray:
    if not use_kinematics:
        dist = np.sqrt((X - px) ** 2 + (Y - py) ** 2)
        return dist / v_max

    rx = px + vx * t_react
    ry = py + vy * t_react

    time_to_reach = np.empty_like(X)
    s0 = np.sqrt(vx**2 + vy**2)

    d_acc_from_zero = (v_max**2) / (2.0 * a_max)
    t_acc_from_zero = v_max / a_max

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            dx = X[i, j] - rx
            dy = Y[i, j] - ry
            dist = np.sqrt(dx**2 + dy**2)

            if dist < 1e-5:
                time_to_reach[i, j] = t_react
                continue

            if s0 < 1e-4:
                if dist <= d_acc_from_zero:
                    t_move = np.sqrt(2.0 * a_max * dist) / a_max
                else:
                    t_move = t_acc_from_zero + (dist - d_acc_from_zero) / v_max
                time_to_reach[i, j] = t_react + t_move
                continue

            ux = dx / dist
            uy = dy / dist
            v_par = vx * ux + vy * uy

            if v_par >= 0.0:
                u0 = min(v_par, v_max)
                d_acc = (v_max**2 - u0**2) / (2.0 * a_max)
                t_acc = (v_max - u0) / a_max

                if dist <= d_acc:
                    t_move = (np.sqrt(u0**2 + 2.0 * a_max * dist) - u0) / a_max
                else:
                    t_move = t_acc + (dist - d_acc) / v_max
                time_to_reach[i, j] = t_react + t_move
            else:
                v_opp = -v_par
                t_decel = v_opp / a_max
                d_drift = (v_opp**2) / (2.0 * a_max)
                d_eff = dist + d_drift

                if d_eff <= d_acc_from_zero:
                    t_accel = np.sqrt(2.0 * a_max * d_eff) / a_max
                else:
                    t_accel = t_acc_from_zero + (d_eff - d_acc_from_zero) / v_max
                time_to_reach[i, j] = t_react + t_decel + t_accel

    return time_to_reach


class KinematicPitchOptimizer:
    """Evaluates spatial pitch control under physiological reaction and acceleration constraints."""

    def __init__(
        self,
        pitch_length: float = PITCH_LENGTH,
        pitch_width: float = PITCH_WIDTH,
        grid_cells_x: int = GRID_CELLS_X,
        grid_cells_y: int = GRID_CELLS_Y,
        reaction_time: float = T_REACT,
        max_velocity: float = V_MAX,
        max_acceleration: float = A_MAX,
        lambda_param: float = LAMBDA,
        use_kinematics: bool = True,
        kernel_type: str = "ma2024",
    ):
        self.L = pitch_length
        self.W = pitch_width
        self.nx = grid_cells_x
        self.ny = grid_cells_y

        self.t_react = reaction_time
        self.v_max = max_velocity
        self.a_max = max_acceleration
        self.lambda_p = lambda_param
        self.use_kinematics = use_kinematics
        self.kernel_type = kernel_type

        self.x_coords = np.linspace(0, self.L, self.nx)
        self.y_coords = np.linspace(0, self.W, self.ny)
        self.X, self.Y = np.meshgrid(self.x_coords, self.y_coords)

        if self.kernel_type == "distance_to_goal":
            goal_x, goal_y = self.L, self.W / 2.0
            dist_to_goal = np.sqrt((self.X - goal_x) ** 2 + (self.Y - goal_y) ** 2)
            dist_to_goal = np.maximum(dist_to_goal, DANGER_FLOOR_M)
            in_danger_zone = (self.X >= DANGER_X_MIN) & (self.Y >= DANGER_Y_MIN) & (self.Y <= DANGER_Y_MAX)
            self.threat_kernel = np.where(in_danger_zone, (HEURISTIC_SCALE / (dist_to_goal ** HEURISTIC_DECAY)), 0.0)
        elif self.kernel_type in ("parametric_xt", "tactical_prior"):
            dist_to_goal = np.sqrt((self.X - self.L) ** 2 + (self.Y - self.W / 2.0) ** 2)
            angle_to_goal = np.abs(np.arctan2(self.Y - self.W / 2.0, self.L - self.X))
            central_threat = np.exp(- (dist_to_goal / PARAMETRIC_CENTRAL_DIST) ** 2) * np.exp(- (angle_to_goal / PARAMETRIC_CENTRAL_ANGLE) ** 2)
            wide_threat = np.exp(- ((self.X - PARAMETRIC_WIDE_X) / PARAMETRIC_WIDE_SPREAD) ** 2) * np.exp(- ((np.abs(self.Y - self.W / 2.0) - PARAMETRIC_WIDE_OFFSET) / PARAMETRIC_WIDE_SPREAD) ** 2) * PARAMETRIC_WIDE_WEIGHT
            half_space_threat = np.exp(- ((self.X - PARAMETRIC_HALF_X) / PARAMETRIC_HALF_SPREAD) ** 2) * np.exp(- ((np.abs(self.Y - self.W / 2.0) - PARAMETRIC_HALF_OFFSET) / PARAMETRIC_HALF_WIDTH) ** 2) * PARAMETRIC_HALF_WEIGHT
            self.threat_kernel = (central_threat + wide_threat + half_space_threat) * PARAMETRIC_SCALE
        elif self.kernel_type in ("ma2024", "dynamic_xt"):
            import pandas as pd
            from scipy.interpolate import RectBivariateSpline

            candidate_paths = [
                os.path.join(os.path.dirname(__file__), "assets", "ma2024.xlsx"),
                os.path.join(os.path.dirname(__file__), "..", "..", "assets", "ma2024.xlsx"),
            ]
            path = next((p for p in candidate_paths if os.path.exists(p)), candidate_paths[0])
            ma2024_raw = pd.read_excel(path, header=None).values

            rows, cols = ma2024_raw.shape
            x_centers = np.linspace(0, self.L, cols)
            y_centers = np.linspace(0, self.W, rows)

            spline = RectBivariateSpline(y_centers, x_centers, ma2024_raw)
            base_xt = spline(self.Y.flatten(), self.X.flatten(), grid=False).reshape(self.X.shape) * MA2024_SCALE

            if self.kernel_type == "dynamic_xt":
                goal_x, goal_y = self.L, self.W / 2.0
                angle_to_goal = np.abs(np.arctan2(self.Y - goal_y, np.maximum(1e-3, self.L - self.X)))
                angle_factor = np.exp(- (angle_to_goal / DYNAMIC_ANGLE_WIDTH) ** 2)
                self.threat_kernel = base_xt * (DYNAMIC_BASE_MIX + DYNAMIC_BASE_MIX * angle_factor)
            else:
                self.threat_kernel = base_xt
        else:
            raise ValueError(f"Unknown kernel_type: {self.kernel_type}")

    def _compute_time_to_intercept(
        self,
        player_pos: np.ndarray,
        player_vel: np.ndarray,
        t_react_penalty: float = 0.0,
        v_max: Optional[float] = None,
        a_max: Optional[float] = None,
        t_react: Optional[float] = None,
    ) -> np.ndarray:
        """Compute time-to-intercept grid for a single agent."""
        eff_t_react = (self.t_react if t_react is None else t_react) + t_react_penalty
        return _compute_time_grid_njit(
            self.X,
            self.Y,
            player_pos[0],
            player_pos[1],
            player_vel[0],
            player_vel[1],
            eff_t_react,
            self.v_max if v_max is None else v_max,
            self.a_max if a_max is None else a_max,
            self.use_kinematics,
        )

    def compute_baseline_surfaces(
        self,
        attacking_players: List[Dict],
        defending_players: List[Dict],
        target_defender_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute minimum arrival time surfaces for attackers and auxiliary defenders."""
        t_att_all = [
            self._compute_time_to_intercept(
                p["pos"], p["vel"],
                v_max=p.get("vmax"), a_max=p.get("amax"), t_react=p.get("t_react"),
            )
            for p in attacking_players
        ]
        min_t_att = np.min(np.stack(t_att_all, axis=0), axis=0)

        other_defs = [
            p for i, p in enumerate(defending_players) if i != target_defender_idx
        ]
        if other_defs:
            t_def_other_all = [
                self._compute_time_to_intercept(
                    p["pos"], p["vel"],
                    v_max=p.get("vmax"), a_max=p.get("amax"), t_react=p.get("t_react"),
                )
                for p in other_defs
            ]
            min_t_def_other = np.min(np.stack(t_def_other_all, axis=0), axis=0)
        else:
            min_t_def_other = np.full_like(min_t_att, 999.0)

        return min_t_att, min_t_def_other

    def evaluate_threat(
        self,
        candidate_pos: np.ndarray,
        candidate_vel: np.ndarray,
        min_t_att: np.ndarray,
        min_t_def_other: np.ndarray,
        t_react_penalty: float = 0.0,
    ) -> float:
        """Evaluate pitch control threat integral for a candidate defender coordinate."""
        t_target_def = self._compute_time_to_intercept(
            candidate_pos, candidate_vel, t_react_penalty
        )
        min_t_def = np.minimum(min_t_def_other, t_target_def)

        delta_t = min_t_att - min_t_def
        p_att = 1.0 / (1.0 + np.exp(np.clip(delta_t / self.lambda_p, -20.0, 20.0)))
        return float(np.sum(p_att * self.threat_kernel))

    def get_kinematic_reachable_set(
        self,
        p0: np.ndarray,
        v0: np.ndarray,
        delta_t: float,
    ) -> Tuple[np.ndarray, float]:
        """Compute reachable set center and maximum sprint radius over horizon delta_t."""
        if not self.use_kinematics:
            return p0.copy(), float(self.v_max * delta_t)

        eff_dt = max(0.0, delta_t - self.t_react)
        center = p0 + v0 * min(delta_t, self.t_react)
        s0 = float(np.linalg.norm(v0))

        if eff_dt <= 0.0:
            max_dist = 0.0
        elif s0 >= self.v_max:
            max_dist = self.v_max * eff_dt
        else:
            t_acc = (self.v_max - s0) / self.a_max
            if eff_dt <= t_acc:
                max_dist = s0 * eff_dt + 0.5 * self.a_max * (eff_dt ** 2)
            else:
                d_acc = s0 * t_acc + 0.5 * self.a_max * (t_acc ** 2)
                d_cruise = self.v_max * (eff_dt - t_acc)
                max_dist = d_acc + d_cruise

        return center, float(max_dist)

    def get_reachable_boundary(
        self,
        center: np.ndarray,
        max_radius: float,
        v0: np.ndarray,
        n_points: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute polygon coordinates for the momentum-adjusted reachable boundary."""
        center = np.asarray(center, dtype=float)
        v0 = np.asarray(v0, dtype=float)
        s0 = float(np.linalg.norm(v0))
        v_angle = float(np.arctan2(v0[1], v0[0])) if s0 > 1e-9 else 0.0

        thetas = np.linspace(0.0, 2.0 * np.pi, n_points)
        xs = np.empty(n_points)
        ys = np.empty(n_points)
        for i, theta in enumerate(thetas):
            phi = np.abs(theta - v_angle)
            while phi > np.pi:
                phi -= 2 * np.pi
            while phi < -np.pi:
                phi += 2 * np.pi
            phi = np.abs(phi)
            if self.use_kinematics:
                k = 0.25
                r_reduction = k * (s0 / self.v_max) * (1.0 - np.cos(phi))
            else:
                r_reduction = 0.0
            r = max_radius * (1.0 - r_reduction)
            xs[i] = center[0] + r * np.cos(theta)
            ys[i] = center[1] + r * np.sin(theta)
        return xs, ys

    def optimize_defender_position(
        self,
        attacking_players: List[Dict],
        defending_players: List[Dict],
        target_defender_idx: int,
        delta_t: float = 1.0,
        radial_samples: int = 15,
        angular_samples: int = 36,
    ) -> Dict:
        """Find the threat-minimizing counterfactual coordinate p* for a target defender."""
        target_def = defending_players[target_defender_idx]
        p0 = np.array(target_def["pos"], dtype=float)
        v0 = np.array(target_def["vel"], dtype=float)
        s0 = float(min(np.linalg.norm(v0), self.v_max))

        min_t_att, min_t_def_other = self.compute_baseline_surfaces(
            attacking_players, defending_players, target_defender_idx
        )
        actual_threat = self.evaluate_threat(p0, v0, min_t_att, min_t_def_other)
        center, max_radius = self.get_kinematic_reachable_set(p0, v0, delta_t)
        reaction_drift = float(np.linalg.norm(center - p0))

        best_p = p0.copy()
        min_threat = actual_threat
        angles = np.linspace(0.0, 2.0 * np.pi, angular_samples, endpoint=False)

        for r_norm in np.linspace(0.0, 1.0, radial_samples):
            for theta in angles:
                phi = np.abs(theta - np.arctan2(v0[1], v0[0]))
                phi = np.unwrap([phi])[0]
                while phi > np.pi:
                    phi -= 2 * np.pi
                while phi < -np.pi:
                    phi += 2 * np.pi
                phi = np.abs(phi)

                if self.use_kinematics:
                    k = 0.25
                    r_reduction = k * (s0 / self.v_max) * (1.0 - np.cos(phi))
                    dynamic_radius = max_radius * (1.0 - r_reduction)
                else:
                    dynamic_radius = max_radius

                r = r_norm * dynamic_radius
                cand_x = np.clip(center[0] + r * np.cos(theta), 0.0, self.L)
                cand_y = np.clip(center[1] + r * np.sin(theta), 0.0, self.W)
                cand_pos = np.array([cand_x, cand_y])

                displacement = cand_pos - p0
                disp_norm = float(np.linalg.norm(displacement))

                t_penalty = 0.0
                if self.use_kinematics and disp_norm > 0.1 and s0 > 0.1:
                    move_angle = np.arctan2(displacement[1], displacement[0])
                    face_angle = np.arctan2(v0[1], v0[0])
                    view_diff = np.abs(move_angle - face_angle)
                    view_diff = np.unwrap([view_diff])[0]
                    while view_diff > np.pi:
                        view_diff -= 2 * np.pi
                    while view_diff < -np.pi:
                        view_diff += 2 * np.pi
                    view_diff = np.abs(view_diff)

                    if view_diff > (np.pi / 3.0):
                        t_penalty = 0.25

                cand_threat = self.evaluate_threat(
                    cand_pos, v0, min_t_att, min_t_def_other, t_react_penalty=t_penalty
                )

                if cand_threat < min_threat:
                    min_threat = cand_threat
                    best_p = cand_pos

        threat_reduction_pct = ((actual_threat - min_threat) / max(actual_threat, 1e-5)) * 100.0
        total_displacement = float(np.linalg.norm(best_p - p0))

        return {
            "target_id": target_def.get("id", f"Defender_{target_defender_idx}"),
            "actual_position": p0,
            "v0": v0,
            "optimal_position": best_p,
            "actual_threat": actual_threat,
            "optimal_threat": min_threat,
            "threat_reduction_pct": threat_reduction_pct,
            "reachable_center": center,
            "reaction_drift": reaction_drift,
            "sprint_radius": max_radius,
            "total_displacement": total_displacement,
            "delta_t": delta_t,
            "use_kinematics": self.use_kinematics,
        }

    def optimize_defensive_unit(
        self,
        attacking_players: List[Dict],
        defending_players: List[Dict],
        target_defenders_indices: List[int],
        delta_t: float = 1.0,
        max_iterations: int = 3,
    ) -> Dict:
        """Perform coordinate descent optimization across a designated group of defenders."""
        import copy
        current_defs = copy.deepcopy(defending_players)

        initial_threat = None
        current_threat = None

        for _ in range(max_iterations):
            for target_idx in target_defenders_indices:
                res = self.optimize_defender_position(
                    attacking_players,
                    current_defs,
                    target_idx,
                    delta_t=delta_t,
                    radial_samples=10,
                    angular_samples=24,
                )
                if initial_threat is None:
                    initial_threat = res["actual_threat"]

                current_defs[target_idx]["pos"] = res["optimal_position"]
                current_threat = res["optimal_threat"]

        threat_reduction_pct = (
            ((initial_threat - current_threat) / max(initial_threat, 1e-5)) * 100.0
            if initial_threat
            else 0.0
        )

        return {
            "initial_threat": initial_threat,
            "optimal_threat": current_threat,
            "threat_reduction_pct": threat_reduction_pct,
            "optimized_defenders": current_defs,
        }

    def plot_frame_analysis(
        self,
        res: Dict,
        attacking_players: List[Dict],
        defending_players: List[Dict],
        save_path: Optional[str] = None,
    ) -> None:
        """Plot tactical pitch, player velocities, reachable boundary, and optimal coordinate p*."""
        if not HAS_MATPLOTLIB:
            return

        pitch = Pitch(
            pitch_type="custom",
            pitch_length=self.L,
            pitch_width=self.W,
            pitch_color="#1a3c1a",
            line_color="white",
        )
        fig, ax = pitch.draw(figsize=(10, 6.5))

        for p in attacking_players:
            pos = p["pos"]
            vel = p["vel"]
            pitch.scatter(pos[0], pos[1], ax=ax, color="#3b82f6", s=120, edgecolors="white", zorder=4)
            if np.linalg.norm(vel) > 0.1:
                pitch.arrows(
                    pos[0], pos[1], pos[0] + vel[0] * 0.5, pos[1] + vel[1] * 0.5,
                    ax=ax, color="#60a5fa", width=1.0, zorder=3, alpha=0.8,
                )
            ax.text(pos[0], pos[1] - 1.5, p.get("id", "Att"), color="white", fontsize=8, ha="center")

        for p in defending_players:
            pos = p["pos"]
            vel = p["vel"]
            if p.get("id") != res["target_id"]:
                pitch.scatter(pos[0], pos[1], ax=ax, color="#ef4444", s=120, edgecolors="white", zorder=4)
                if np.linalg.norm(vel) > 0.1:
                    pitch.arrows(
                        pos[0], pos[1], pos[0] + vel[0] * 0.5, pos[1] + vel[1] * 0.5,
                        ax=ax, color="#f87171", width=1.0, zorder=3, alpha=0.8,
                    )
                ax.text(pos[0], pos[1] - 1.5, p.get("id", "Def"), color="white", fontsize=8, ha="center")

        act_p = res["actual_position"]
        opt_p = res["optimal_position"]
        r_center = res["reachable_center"]
        r_radius = res["sprint_radius"]
        total_disp = res["total_displacement"]

        if res["reaction_drift"] > 0.1:
            pitch.lines(act_p[0], act_p[1], r_center[0], r_center[1], ax=ax, color="#f59e0b", ls="--", lw=1.5)
            pitch.scatter(r_center[0], r_center[1], ax=ax, color="#f59e0b", s=60, marker="o", alpha=0.7, zorder=5)

        boundary_xs, boundary_ys = self.get_reachable_boundary(
            r_center, r_radius, res.get("v0", [0, 0])
        )
        boundary_poly = plt.Polygon(
            np.column_stack((boundary_xs, boundary_ys)),
            color="#22c55e",
            fill=True,
            alpha=0.2,
            ls="--",
            lw=1.5,
        )
        ax.add_patch(boundary_poly)

        pitch.scatter(act_p[0], act_p[1], ax=ax, color="#ef4444", s=180, marker="X", edgecolors="white", lw=1.5, zorder=6)
        pitch.scatter(opt_p[0], opt_p[1], ax=ax, color="#22c55e", s=180, marker="o", edgecolors="white", lw=1.5, zorder=6)

        if total_disp > 0.1:
            ax.annotate("", xy=(opt_p[0], opt_p[1]), xytext=(act_p[0], act_p[1]),
                        arrowprops=dict(arrowstyle="->", color="yellow", lw=2, ls=":"))

        ax.set_title(
            f"Counterfactual Positioning: {res['target_id']} (ΔThreat: -{res['threat_reduction_pct']:.1f}%, Displacement: {total_disp:.2f}m)",
            color="white", fontsize=11, pad=12,
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=200, facecolor="#111827")
        plt.close()


if __name__ == "__main__":
    opt = KinematicPitchOptimizer()
    attackers = [{"pos": [78.0, 30.0], "vel": [5.5, 1.2]}, {"pos": [92.0, 36.0], "vel": [6.8, -0.5]}]
    defenders = [{"pos": [102.0, 34.0], "vel": [0.0, 0.0]}, {"pos": [84.0, 38.0], "vel": [4.0, 2.5]}]
    res = opt.optimize_defender_position(attackers, defenders, target_defender_idx=1, delta_t=1.0)
    print(f"K-PACE initialized ({opt.nx}x{opt.ny} grid, {opt.kernel_type} threat kernel).")
    print(f"Sample optimization: threat delta = -{res['threat_reduction_pct']:.1f}%, displacement = {res['total_displacement']:.2f}m")
