import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.metrics import r2_score

def ts_to_minutes(ts):
    h, m = map(int, ts.split(':'))
    return h * 60 + m

def run_pipeline():
    print("Loading data...")
    train = pd.read_csv('dataset/train.csv')
    test = pd.read_csv('dataset/test.csv')

    df = pd.concat([train, test], sort=False).reset_index(drop=True)

    print("Preprocessing...")
    df['RoadType'] = df.groupby('geohash')['RoadType'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan))
    df['RoadType'] = df['RoadType'].fillna(df['RoadType'].mode()[0])
    df['Temperature'] = df['Temperature'].fillna(df['Temperature'].median())
    df['Weather'] = df['Weather'].fillna(df['Weather'].mode()[0])

    df['total_minutes'] = df['timestamp'].apply(ts_to_minutes)
    df['hour'] = df['total_minutes'] // 60
    df['minute'] = df['total_minutes'] % 60

    cat_cols = ['geohash', 'RoadType', 'LargeVehicles', 'Landmarks', 'Weather']
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    print("Feature engineering...")
    raw_geohashes = pd.concat([train['geohash'], test['geohash']]).reset_index(drop=True)
    for i in range(3, 6):
        df[f'geohash_{i}'] = raw_geohashes.str[:i]
        le = LabelEncoder()
        df[f'geohash_{i}'] = le.fit_transform(df[f'geohash_{i}'].astype(str))

    # Day 48 and Day 49 (train) data
    train_indices = df['demand'].notnull()
    train_df = df[train_indices].copy()

    # Target encoding for geohash and hour
    gh_mean = train_df.groupby('geohash')['demand'].mean()
    df['gh_mean'] = df['geohash'].map(gh_mean)

    h_mean = train_df.groupby('hour')['demand'].mean()
    df['h_mean'] = df['hour'].map(h_mean)

    gh_h_mean = train_df.groupby(['geohash', 'hour'])['demand'].mean().reset_index().rename(columns={'demand': 'gh_h_mean'})
    df = df.merge(gh_h_mean, on=['geohash', 'hour'], how='left')

    # Fill NAs
    df['gh_mean'] = df['gh_mean'].fillna(train_df['demand'].mean())
    df['h_mean'] = df['h_mean'].fillna(train_df['demand'].mean())
    df['gh_h_mean'] = df['gh_h_mean'].fillna(df['gh_mean'])

    # Split
    train_final = df[df['demand'].notnull()].copy()
    test_final = df[df['demand'].isnull()].copy()

    features = [col for col in train_final.columns if col not in ['Index', 'day', 'timestamp', 'demand']]

    X_train = train_final[train_final['day'] == 48][features]
    y_train = train_final[train_final['day'] == 48]['demand']
    X_val = train_final[train_final['day'] == 49][features]
    y_val = train_final[train_final['day'] == 49]['demand']

    print("Training model...")
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'random_state': 42,
        'learning_rate': 0.01,
        'num_leaves': 255,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=5000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=200)]
    )

    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)
    print(f"Validation R2: {r2}")

    # Train on full data for final prediction
    print("Training on full data...")
    full_train_data = lgb.Dataset(train_final[features], label=train_final['demand'])
    final_model = lgb.train(
        params,
        full_train_data,
        num_boost_round=model.best_iteration
    )

    print("Predicting for test set...")
    test_preds = final_model.predict(test_final[features])

    submission = pd.DataFrame({
        'Index': test['Index'],
        'demand': test_preds
    })

    submission.to_csv('submission.csv', index=False)
    print("Submission saved to submission.csv")

if __name__ == "__main__":
    run_pipeline()
