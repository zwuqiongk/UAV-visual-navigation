import numpy as np
import math
from enum import Enum

class TrackState(Enum):
    DETECTED = 0
    CONFIRMED = 1
    BOUND = 2

class LandmarkNavSystem:
    def __init__(self):
        # Hyperparameters from Table 2
        # Lifecycle Management
        self.alpha = 30.0      # Binding distance threshold (m)
        self.P = 3             # Consecutive frames -> Confirmed
        self.Q = 5             # Consecutive frames -> Bound
        self.T = 30            # Timeout to degrade status
        self.K = 60            # Frames to trigger ID deletion
        
        # Adaptive Weighting
        self.sigma_d = 40.0    # Distance sensitivity
        self.sigma_p = 200.0   # Visual distortion sensitivity (pixel)
        self.lam = 2.0         # Softmax temperature

        # State Priors (Eq. 10)
        self.prob_prior = {
            TrackState.BOUND: 1.0,
            TrackState.CONFIRMED: 0.8,
            TrackState.DETECTED: 0.5
        }

        # database: track_id -> {state, consecutive_hits, lost_frames, is_active}
        self.tracks = {} 

    def update_lifecycle(self, track_id, distance_to_target):
        """
        Implements Landmark Binding and Lifecycle Management (Fig 3).
        """
        # Initialize new track
        if track_id not in self.tracks:
            self.tracks[track_id] = {
                'state': TrackState.DETECTED,
                'consecutive_hits': 1,
                'lost_frames': 0,
                'is_active': True 
            }
            return TrackState.DETECTED

        track = self.tracks[track_id]
        track['is_active'] = True 

        # 1. Geometric constraint check
        if distance_to_target > self.alpha:
            return self._handle_lost_frame(track_id)

        # 2. State transition logic
        track['lost_frames'] = 0
        track['consecutive_hits'] += 1

        if track['state'] == TrackState.DETECTED:
            if track['consecutive_hits'] >= self.P:
                track['state'] = TrackState.CONFIRMED
                
        elif track['state'] == TrackState.CONFIRMED:
            if track['consecutive_hits'] >= (self.P + self.Q):
                track['state'] = TrackState.BOUND
        
        return track['state']

    def predict_lifecycle(self):
        """
        Handles status degradation and deletion for missing tracks.
        Should be called at the end of each frame.
        """
        ids_to_delete = []
        
        for tid, track in self.tracks.items():
            if not track['is_active']:
                track['lost_frames'] += 1
                track['consecutive_hits'] = 0 

                # Degrade status based on timeout T
                if track['lost_frames'] > self.T:
                    if track['state'] == TrackState.BOUND:
                        track['state'] = TrackState.CONFIRMED
                    elif track['state'] == TrackState.CONFIRMED:
                        track['state'] = TrackState.DETECTED
                
                # Mark for deletion
                if track['lost_frames'] > self.K:
                    ids_to_delete.append(tid)
            
            # Reset flag for next frame
            track['is_active'] = False

        for tid in ids_to_delete:
            del self.tracks[tid]

    def _handle_lost_frame(self, track_id):
        return self.tracks[track_id]['state']

    def adaptive_weighted_fusion(self, observations):
        """
        Implements Algorithm 1: Adaptive Spatio-Temporal Landmark Weighting.
        observations: list of dicts {'pos', 'D', 'E', 'state'}
        """
        if not observations:
            return None, []

        omega_list = [] 
        
        # Step 1: Calculate raw reliability scores
        for obs in observations:
            D_i = obs['D']
            E_i = obs['E']
            S_i = obs['state']

            # Eq. 8: Spatial Reliability
            W_d = math.exp(-(D_i**2) / (2 * self.sigma_d**2))
            
            # Eq. 9: Visual Fidelity
            W_p = math.exp(-(E_i**2) / (2 * self.sigma_p**2))
            
            # Eq. 11: Joint Information Score
            Omega = self.prob_prior[S_i] * W_d * W_p
            omega_list.append(Omega)

        # Step 2: Softmax Normalization (Eq. 12)
        omega_arr = np.array(omega_list)
        # Subtract max for numerical stability
        exp_vals = np.exp(self.lam * (omega_arr - np.max(omega_arr)))
        weights = exp_vals / np.sum(exp_vals)

        # Step 3: Weighted average
        final_pos = np.zeros(3)
        for i, obs in enumerate(observations):
            final_pos += weights[i] * obs['pos']

        return final_pos, weights