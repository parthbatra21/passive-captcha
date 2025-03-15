import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import warnings
import time  # For timestamping behavior data
warnings.filterwarnings('ignore')

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

def generate_human_mouse(num_points=100):
    """Generate realistic human-like mouse movements"""
    x, y = 0, 0  # Starting position
    movements = []
    
    for _ in range(num_points):
        # Simulate acceleration/deceleration
        speed = np.random.uniform(0.5, 2.0)  # Random speed
        dx = np.random.normal(5, 2)  # Small random steps in x
        dy = np.random.normal(5, 2)  # Small random steps in y
        
        # Add slight curvature
        if len(movements) > 1:
            dx += 0.1 * (movements[-1]['x'] - movements[-2]['x'])
            dy += 0.1 * (movements[-1]['y'] - movements[-2]['y'])
        
        x += dx * speed
        y += dy * speed
        
        # Add occasional pauses
        if np.random.rand() < 0.1:  # 10% chance of pause
            movements.append({'x': x, 'y': y, 'time': time.time() * 1000})  # Milliseconds
            continue
        
        movements.append({'x': x, 'y': y, 'time': time.time() * 1000})  # Milliseconds
    
    return movements

def generate_bot_mouse(num_points=100):
    """Generate bot-like mouse movements"""
    x, y = 0, 0  # Starting position
    movements = []
    
    for _ in range(num_points):
        # Linear movement with fixed step size
        dx = np.random.uniform(5, 10)  # Fixed step size in x
        dy = np.random.uniform(5, 10)  # Fixed step size in y
        
        x += dx
        y += dy
        
        movements.append({'x': x, 'y': y, 'time': time.time() * 1000})  # Milliseconds
    
    return movements

def generate_human_keystrokes(num_keys=50):
    """Generate realistic human-like keystrokes"""
    keys = []
    last_time = time.time() * 1000  # Milliseconds
    
    for _ in range(num_keys):
        # Random key (alphanumeric + some special keys)
        key = np.random.choice(list("abcdefghijklmnopqrstuvwxyz0123456789") + ["Backspace", "Shift"])
        
        # Variable timing between keystrokes
        interval = np.random.exponential(200)  # Exponential distribution for intervals
        last_time += interval
        
        # Simulate occasional errors (backspaces)
        if key == "Backspace" and len(keys) > 0:
            keys.pop()  # Remove last key
            keys.append({'key': 'Backspace', 'time': last_time})
        else:
            keys.append({'key': key, 'time': last_time})
    
    return keys

def generate_bot_keystrokes(num_keys=50):
    """Generate bot-like keystrokes"""
    keys = []
    last_time = time.time() * 1000  # Milliseconds
    
    for _ in range(num_keys):
        # Fixed interval between keystrokes
        interval = 100  # Constant interval (ms)
        last_time += interval
        
        # Random key (alphanumeric)
        key = np.random.choice(list("abcdefghijklmnopqrstuvwxyz0123456789"))
        keys.append({'key': key, 'time': last_time})
    
    return keys

def generate_human_scroll(num_steps=50):
    """Generate realistic human-like scroll patterns"""
    scrolls = []
    scroll_y = 0
    last_time = time.time() * 1000  # Milliseconds
    
    for _ in range(num_steps):
        # Variable scroll speed
        speed = np.random.uniform(1, 10)  # Random speed
        direction = np.random.choice([-1, 1])  # Random direction (up/down)
        
        scroll_y += direction * speed
        interval = np.random.exponential(200)  # Exponential distribution for intervals
        last_time += interval
        
        # Add occasional pauses
        if np.random.rand() < 0.2:  # 20% chance of pause
            scrolls.append({'scrollY': scroll_y, 'time': last_time})
            continue
        
        scrolls.append({'scrollY': scroll_y, 'time': last_time})
    
    return scrolls

def generate_bot_scroll(num_steps=50):
    """Generate bot-like scroll patterns"""
    scrolls = []
    scroll_y = 0
    last_time = time.time() * 1000  # Milliseconds
    
    for _ in range(num_steps):
        # Fixed scroll speed
        speed = 5  # Constant speed
        direction = 1  # Always scroll down
        
        scroll_y += direction * speed
        interval = 100  # Constant interval (ms)
        last_time += interval
        
        scrolls.append({'scrollY': scroll_y, 'time': last_time})
    
    return scrolls

def generate_realistic_dataset(n_samples=10000):
    """Create realistic human/bot behavior data"""
    data = []
    for _ in range(n_samples):
        is_bot = np.random.choice([0, 1], p=[0.7, 0.3])  # Class imbalance
        
        if is_bot:
            # Bot behavior patterns
            mouse_movements = generate_bot_mouse()
            key_presses = generate_bot_keystrokes()
            scroll_patterns = generate_bot_scroll()
        else:
            # Human behavior patterns
            mouse_movements = generate_human_mouse()
            key_presses = generate_human_keystrokes()
            scroll_patterns = generate_human_scroll()
            
        raw_data = {
            'mouseMovements': mouse_movements,
            'keyPresses': key_presses,
            'scrollPatterns': scroll_patterns,
            'isAutomated': is_bot,
            'cpuCores': np.random.choice([2, 4, 8], p=[0.2, 0.5, 0.3])
        }
        
        features = BehavioralFeatureEngineer.extract_features(raw_data)
        features['is_human'] = 1 - is_bot
        data.append(features)
    
    return pd.DataFrame(data)

if __name__ == '__main__':
    # Generate and save dataset
    df = generate_realistic_dataset(50000)
    df.to_parquet('behavior_dataset.parquet', index=False, engine='pyarrow')  # Use 'pyarrow' or 'fastparquet'
    
    # Train-test split
    X = df.drop(columns=['is_human'])
    y = df['is_human']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    
    # Preprocessing
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model training with hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    }
    
    model = GridSearchCV(GradientBoostingClassifier(subsample=0.8),
                        param_grid,
                        scoring='roc_auc',
                        cv=3,
                        n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Evaluation
    print("Best Parameters:", model.best_params_)
    y_pred = model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]))
    
    # Save artifacts
    joblib.dump(model.best_estimator_, 'bot_detector_gb.pkl')
    joblib.dump(scaler, 'scaler_gb.pkl')