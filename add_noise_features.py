import pandas as pd
import numpy as np
import os

def add_noise_features(df, is_train=True):
    n_samples = len(df)
    np.random.seed(42 if is_train else 43)
    
    print(f"Adding noise features to {'train' if is_train else 'test'} set ({n_samples} samples)...")
    
    # rand_1 ~ rand_9: Simple Uniform Noise [0, 1]
    for i in range(1, 10):
        df[f'rand_{i}'] = np.random.rand(n_samples)
        
    # rand_10 ~ rand_30: "Plausible" feature distributions
    
    # 10-14: Normal distributions with different means/stds
    df['rand_10'] = np.random.normal(0, 1, n_samples)
    df['rand_11'] = np.random.normal(100, 20, n_samples)
    df['rand_12'] = np.random.normal(-50, 5, n_samples)
    df['rand_13'] = np.random.normal(0.5, 0.1, n_samples)
    df['rand_14'] = np.random.normal(1000, 500, n_samples)
    
    # 15-19: Log-normal / Exponential (Skewed)
    df['rand_15'] = np.random.lognormal(0, 1, n_samples)
    df['rand_16'] = np.random.exponential(1.0, n_samples)
    df['rand_17'] = np.random.chisquare(2, n_samples)
    df['rand_18'] = np.random.beta(0.5, 0.5, n_samples) # U-shaped
    df['rand_19'] = np.random.gamma(2, 2, n_samples)
    
    # 20-24: Integer-like / Discrete-like but continuous noise
    df['rand_20'] = np.random.randint(0, 100, n_samples).astype(float) + np.random.rand(n_samples) * 0.1
    df['rand_21'] = np.random.poisson(5, n_samples).astype(float)
    df['rand_22'] = np.round(np.random.normal(50, 10, n_samples))
    df['rand_23'] = np.random.choice([0, 1, 2], n_samples, p=[0.7, 0.2, 0.1]) + np.random.normal(0, 0.05, n_samples)
    df['rand_24'] = np.floor(np.random.uniform(0, 10, n_samples))
    
    # 25-30: Bimodal / Correlated / Structured noise
    # Bimodal
    mask = np.random.rand(n_samples) > 0.5
    df['rand_25'] = np.where(mask, np.random.normal(0, 1, n_samples), np.random.normal(10, 1, n_samples))
    
    # Correlated with other rand features (synthetic collinearity)
    df['rand_26'] = df['rand_10'] * 2 + np.random.normal(0, 0.5, n_samples)
    df['rand_27'] = df['rand_15'] / (df['rand_1'] + 0.1)
    df['rand_28'] = np.sin(df['rand_11'] / 10)
    df['rand_29'] = df['rand_12'] ** 2
    
    # Outlier-heavy
    df['rand_30'] = np.random.normal(0, 1, n_samples)
    outlier_idx = np.random.choice(n_samples, int(n_samples * 0.01), replace=False)
    df.loc[outlier_idx, 'rand_30'] = df.loc[outlier_idx, 'rand_30'] * 10
    
    return df

def main():
    root = '/workspace/data/loan'
    train_path = os.path.join(root, 'train.csv')
    test_path = os.path.join(root, 'test.csv')
    
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found.")
        return

    # Load
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path) if os.path.exists(test_path) else None
    
    # Process
    train_df = add_noise_features(train_df, is_train=True)
    if test_df is not None:
        test_df = add_noise_features(test_df, is_train=False)
    
    # Save
    train_out = os.path.join(root, 'train_noisy.csv')
    train_df.to_csv(train_out, index=False)
    print(f"Saved {train_out}")
    
    if test_df is not None:
        test_out = os.path.join(root, 'test_noisy.csv')
        test_df.to_csv(test_out, index=False)
        print(f"Saved {test_out}")

if __name__ == '__main__':
    main()
