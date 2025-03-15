# backend/feature_engineer.py
import numpy as np

class BehavioralFeatureEngineer:
    @staticmethod
    def extract_features(raw_data):
        """Convert raw behavioral data into meaningful features"""
        features = {}
        
        # Mouse dynamics
        mouse = raw_data['mouseMovements']
        if len(mouse) > 1:
            dx = np.diff([m['x'] for m in mouse])
            dy = np.diff([m['y'] for m in mouse])
            dt = np.diff([m['time'] for m in mouse])
            
            velocity = np.sqrt(dx**2 + dy**2) / (np.array(dt) + 1e-6)
            
            # Handle edge case for mouse_jerk
            if len(velocity) > 1:
                mouse_jerk = np.mean(np.abs(np.diff(velocity)))
            else:
                mouse_jerk = 0  # Default value if velocity has only one element
            
            features.update({
                'mouse_velocity_mean': np.mean(velocity),
                'mouse_velocity_std': np.std(velocity),
                'mouse_jerk': mouse_jerk,
                'mouse_entropy': np.log(np.var(dx) * np.var(dy) + 1e-6)
            })
        
        # Keystroke dynamics
        keys = raw_data['keyPresses']
        if len(keys) > 1:
            times = [k['time'] for k in keys]
            intervals = np.diff(times)
            features.update({
                'key_interval_mean': np.mean(intervals),
                'key_interval_std': np.std(intervals),
                'key_entropy': -np.sum(np.histogram(intervals, bins=5)[0] * 
                                     np.log(np.histogram(intervals, bins=5)[0] + 1e-6))
            })
        
        # Scroll dynamics
        scrolls = raw_data['scrollPatterns']
        if len(scrolls) > 1:
            dy = np.diff([s['scrollY'] for s in scrolls])
            dt = np.diff([s['time'] for s in scrolls])
            scroll_speed = np.abs(dy) / (np.array(dt) + 1e-6)
            features.update({
                'scroll_speed_mean': np.mean(scroll_speed),
                'scroll_speed_std': np.std(scroll_speed),
                'scroll_direction_changes': np.sum(np.diff(np.sign(dy)) != 0)
            })
        
        # Browser features
        features.update({
            'automated': raw_data['isAutomated'],
            'cpu_cores': raw_data['cpuCores'],
            'session_duration': mouse[-1]['time'] - mouse[0]['time'] if mouse else 0
        })
        
        return features