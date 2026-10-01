"""Data loader and telemetry extraction for Metrica Sports tracking data."""

import os
from typing import Dict, List, Optional, Tuple

from kloppy import metrica
import numpy as np
from scipy.signal import savgol_filter


class MetricaDataLoader:
    """Loads and preprocesses tracking telemetry from Metrica Sports datasets."""

    def __init__(
        self,
        game_id: str,
        data_dir: str,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
    ):
        self.game_id = game_id
        self.L = pitch_length
        self.W = pitch_width

        tracking_txt = os.path.join(data_dir, f"{game_id}_tracking.txt")
        meta_xml = os.path.join(data_dir, f"{game_id}_metadata.xml")

        if os.path.exists(tracking_txt) and os.path.exists(meta_xml):
            self.dataset = metrica.load_tracking_epts(
                meta_data=meta_xml,
                raw_data=tracking_txt,
            )
        else:
            home_csv = os.path.join(data_dir, f"{game_id}_RawTrackingData_Home_Team.csv")
            away_csv = os.path.join(data_dir, f"{game_id}_RawTrackingData_Away_Team.csv")
            self.dataset = metrica.load_tracking_csv(
                home_data=home_csv,
                away_data=away_csv,
            )

        self.frames_dict = {f.frame_id: i for i, f in enumerate(self.dataset.frames)}
        self.frames_list = self.dataset.frames
        self.keeper_ids = self._identify_goalkeepers()

    def _identify_goalkeepers(self, n_frames: int = 25) -> Dict[str, str]:
        """Identify goalkeeper IDs based on proximity to goal centers in kickoff frames."""
        best: Dict[tuple, float] = {}
        for f in self.frames_list[:n_frames]:
            for player, coords in f.players_coordinates.items():
                if coords is None or np.isnan(coords.x) or np.isnan(coords.y):
                    continue
                team = str(player.team.ground).split(".")[-1].lower()
                px = coords.x * self.L
                py = (1.0 - coords.y) * self.W
                d = min(
                    float(np.hypot(px - 0.0, py - self.W / 2.0)),
                    float(np.hypot(px - self.L, py - self.W / 2.0)),
                )
                key = (team, player.jersey_no)
                if key not in best or d < best[key]:
                    best[key] = d

        keepers: Dict[str, str] = {}
        for team in ("home", "away"):
            cands = [(d, j) for (t, j), d in best.items() if t == team]
            if cands:
                _, jersey = min(cands)
                prefix = "Home" if team == "home" else "Away"
                keepers[team] = f"{prefix}_{jersey}"
        return keepers

    def get_frame_telemetry(
        self, target_frame: int = 120259
    ) -> Tuple[List[Dict], List[Dict], Dict]:
        """Extract smoothed player positions and velocities at target frame."""
        if target_frame not in self.frames_dict:
            avail = sorted(list(self.frames_dict.keys()))
            target_frame = avail[len(avail) // 2]

        idx = self.frames_dict[target_frame]
        window_size = 7
        half_window = window_size // 2

        f_curr = self.frames_list[idx]
        idx_next = min(len(self.frames_list) - 1, idx + 1)
        dt = (self.frames_list[idx_next].timestamp - f_curr.timestamp).total_seconds()
        if dt <= 0.0 or dt > 0.1:
            dt = 0.04

        def extract_team_players(team_str: str) -> List[Dict]:
            players = []
            for player, coords_curr in f_curr.players_coordinates.items():
                ground_str = str(player.team.ground).split(".")[-1].lower()
                if ground_str != team_str.lower() and str(player.team.ground).lower() != team_str.lower():
                    continue

                if coords_curr is None or np.isnan(coords_curr.x) or np.isnan(coords_curr.y):
                    continue

                px = coords_curr.x * self.L
                py = (1.0 - coords_curr.y) * self.W

                hist_x = []
                hist_y = []
                for offset in range(-half_window, half_window + 1):
                    f_idx = max(0, min(len(self.frames_list) - 1, idx + offset))
                    coords_hist = self.frames_list[f_idx].players_coordinates.get(player)
                    if coords_hist is not None and not np.isnan(coords_hist.x):
                        hist_x.append(coords_hist.x * self.L)
                        hist_y.append((1.0 - coords_hist.y) * self.W)
                    else:
                        hist_x.append(px)
                        hist_y.append(py)

                try:
                    vx_arr = savgol_filter(hist_x, window_length=window_size, polyorder=2, deriv=1, delta=dt)
                    vy_arr = savgol_filter(hist_y, window_length=window_size, polyorder=2, deriv=1, delta=dt)
                    vx = vx_arr[half_window]
                    vy = vy_arr[half_window]

                    speed = np.sqrt(vx**2 + vy**2)
                    if speed > 12.0:
                        vx = (vx / speed) * 12.0
                        vy = (vy / speed) * 12.0
                except Exception:
                    vx, vy = 0.0, 0.0

                team_prefix = "Home" if team_str == "home" else "Away"
                players.append({
                    "id": f"{team_prefix}_{player.jersey_no}",
                    "pos": np.array([px, py]),
                    "vel": np.array([vx, vy]),
                })
            return players

        home_players = extract_team_players("home")
        away_players = extract_team_players("away")

        b_coords = f_curr.ball_coordinates
        if b_coords is None or np.isnan(b_coords.x) or np.isnan(b_coords.y):
            ball_info = {"pos": np.array([self.L / 2, self.W / 2])}
        else:
            ball_info = {"pos": np.array([b_coords.x * self.L, (1.0 - b_coords.y) * self.W])}

        return home_players, away_players, ball_info
