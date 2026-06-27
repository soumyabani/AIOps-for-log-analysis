"""
My AIOps Log Analysis System
============================

What it does:
- Automatically learns suspicious patterns from log data
- Uses multiple ML algorithms to reduce false positives  
- Explains why each log entry was flagged as anomalous
- Saves results in easy-to-review CSV files

"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import json
import os
import re
import warnings
warnings.filterwarnings('ignore')  # Got tired of sklearn warnings during development


DEFAULT_CONFIG = {
    "log_file_path": "data/Hadoop_2k.log",
    "contamination": 0.08,
    "train_ratio": 0.7,
    "zscore_threshold": 3.0,
    "severity_boost": 1,
    "max_features": 120,
    "level_mapping": {
        "DEBUG": 0,
        "INFO": 1,
        "WARN": 2,
        "WARNING": 2,
        "ERROR": 3,
        "CRITICAL": 4,
        "FATAL": 5
    }
}


def load_config(config_path="config.json"):
    """Load optional config and fall back to sane defaults for local runs."""
    config = DEFAULT_CONFIG.copy()
    config["level_mapping"] = DEFAULT_CONFIG["level_mapping"].copy()

    try:
        with open(config_path, "r", encoding="utf-8") as config_file:
            user_config = json.load(config_file)
    except FileNotFoundError:
        print(f"Config file '{config_path}' not found. Using defaults.")
        return config
    except json.JSONDecodeError as exc:
        print(f"Config file '{config_path}' is invalid JSON: {exc}. Using defaults.")
        return config

    for key, value in user_config.items():
        if key == "level_mapping" and isinstance(value, dict):
            config["level_mapping"].update(value)
        else:
            config[key] = value

    return config


config = load_config()
log_file_path = config["log_file_path"]
level_mapping = config["level_mapping"]
contamination = config["contamination"]
train_ratio = config["train_ratio"]
zscore_threshold = config["zscore_threshold"]
severity_boost = config["severity_boost"]
max_features = config["max_features"]
output_dir = config.get("output_dir", "output")
os.makedirs(output_dir, exist_ok=True)

print(f"Loading logs from: {log_file_path}")

try:
    with open(log_file_path, "r", encoding="utf-8") as file:
        logs = file.readlines()
    print(f"Successfully loaded {len(logs)} log entries")
except FileNotFoundError:
    print("Error: Log file not found!")
    exit(1)

# my custom log parser - took a while to get this right
def parse_my_logs(log_lines):
    """Parse Hadoop-style logs with timestamp, level, thread, logger and message."""
    parsed_data = []
    failed_count = 0
    log_pattern = re.compile(
        r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\s+"
        r"(?P<level>[A-Z]+)\s+"
        r"\[(?P<thread>.*?)\]\s+"
        r"(?P<logger>[\w.$]+):\s+"
        r"(?P<message>.*)$"
    )
    
    for line_num, log in enumerate(log_lines):
        try:
            match = log_pattern.match(log.strip())
            if not match:
                failed_count += 1
                continue

            parsed_data.append([
                match.group("timestamp"),
                match.group("level"),
                match.group("thread"),
                match.group("logger"),
                match.group("message")
            ])
            
        except Exception as e:
            # print(f"Debug: Failed to parse line {line_num}: {e}")  # Used for debugging
            failed_count += 1
            continue
    
    if failed_count > 0:
        print(f"Warning: Couldn't parse {failed_count} lines (probably malformed)")
    
    return parsed_data

# Parse all the logs
print("Parsing log entries...")
data = parse_my_logs(logs)
df = pd.DataFrame(data, columns=["timestamp", "level", "thread", "logger", "message"])

if df.empty:
    print("Error: No log entries could be parsed. Check the log format and parser pattern.")
    exit(1)

print(f"Successfully parsed {len(df)} log entries")

# Convert timestamps - pandas is pretty good at this
df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')

# My scoring system for log levels - learned this from experience
# INFO is usually fine, CRITICAL means wake me up at 3am!
df["level_score"] = df["level"].map(level_mapping).fillna(1)

# Message length might be important - longer messages often mean trouble
df["msg_len"] = df["message"].apply(len)
df["logger_depth"] = df["logger"].str.count(r"\.") + 1
df["thread_len"] = df["thread"].apply(len)
df["has_digits"] = df["message"].str.contains(r"\d", regex=True).astype(int)
df["has_job_id"] = df["message"].str.contains(r"job_\d+|task_\d+|attempt_\d+", regex=True).astype(int)

# quick check of what we're working with
print(f"Log levels found: {df['level'].value_counts().to_dict()}")
print(f"Average message length: {df['msg_len'].mean():.1f} characters")
print(f"Top logger namespaces: {df['logger'].value_counts().head(5).to_dict()}")

numeric_columns = [
    "level_score",
    "msg_len",
    "logger_depth",
    "thread_len",
    "has_digits",
    "has_job_id"
]


def build_text_corpus(data_df):
    """Combine text-heavy columns so TF-IDF can learn operational context."""
    return (
        data_df["logger"].fillna("") + " " +
        data_df["thread"].fillna("") + " " +
        data_df["message"].fillna("")
    )


def get_split_index(total_rows, ratio):
    """Compute safe split index and keep at least one row on each side."""
    split_idx = int(total_rows * ratio)
    split_idx = max(1, min(total_rows - 1, split_idx))
    return split_idx


def train_test_anomaly_detection(log_df):
    """Train on early logs and evaluate on later logs to reduce false positives."""
    print("Analyzing log patterns with time-based train/test split...")

    sorted_df = log_df.sort_values("timestamp").reset_index(drop=True)
    split_idx = get_split_index(len(sorted_df), train_ratio)
    train_df = sorted_df.iloc[:split_idx].copy()
    test_df = sorted_df.iloc[split_idx:].copy()

    print(f"Train/Test split: {len(train_df)} / {len(test_df)}")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2)
    )

    train_text = build_text_corpus(train_df)
    test_text = build_text_corpus(test_df)
    train_text_matrix = vectorizer.fit_transform(train_text).toarray()
    test_text_matrix = vectorizer.transform(test_text).toarray()

    scaler = StandardScaler()
    train_numeric = scaler.fit_transform(train_df[numeric_columns].values)
    test_numeric = scaler.transform(test_df[numeric_columns].values)

    train_features = np.hstack([train_numeric, train_text_matrix])
    test_features = np.hstack([test_numeric, test_text_matrix])

    detector = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=200
    )
    detector.fit(train_features)

    train_if_score = -detector.score_samples(train_features)
    test_if_score = -detector.score_samples(test_features)

    # Threshold learned from train distribution for stable out-of-sample scoring.
    train_if_threshold = np.quantile(train_if_score, 1 - contamination)

    train_z = np.abs((train_numeric - train_numeric.mean(axis=0)) / (train_numeric.std(axis=0) + 1e-9))
    test_z = np.abs((test_numeric - train_numeric.mean(axis=0)) / (train_numeric.std(axis=0) + 1e-9))
    train_z_max = train_z.max(axis=1)
    test_z_max = test_z.max(axis=1)

    train_if_flag = train_if_score >= train_if_threshold
    test_if_flag = test_if_score >= train_if_threshold
    train_z_flag = train_z_max > zscore_threshold
    test_z_flag = test_z_max > zscore_threshold

    train_severity_flag = train_df["level"].isin(["ERROR", "CRITICAL", "FATAL"]).values
    test_severity_flag = test_df["level"].isin(["ERROR", "CRITICAL", "FATAL"]).values

    train_votes = train_if_flag.astype(int) + train_z_flag.astype(int) + (train_severity_flag.astype(int) * severity_boost)
    test_votes = test_if_flag.astype(int) + test_z_flag.astype(int) + (test_severity_flag.astype(int) * severity_boost)

    train_pred = np.where(train_votes >= 2, -1, 1)
    test_pred = np.where(test_votes >= 2, -1, 1)

    feature_names = vectorizer.get_feature_names_out()
    importance_scores = np.abs(train_text_matrix).mean(axis=0)
    top_indices = np.argsort(importance_scores)[-8:]
    suspicious_patterns = [feature_names[i] for i in top_indices]

    train_df["split"] = "train"
    test_df["split"] = "test"
    train_df["anomaly_score"] = train_pred
    test_df["anomaly_score"] = test_pred

    train_df["if_score"] = train_if_score
    test_df["if_score"] = test_if_score
    train_df["zscore_max"] = train_z_max
    test_df["zscore_max"] = test_z_max

    train_df["if_flag"] = train_if_flag.astype(int)
    test_df["if_flag"] = test_if_flag.astype(int)
    train_df["zscore_flag"] = train_z_flag.astype(int)
    test_df["zscore_flag"] = test_z_flag.astype(int)

    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined_df["status"] = combined_df["anomaly_score"].apply(lambda x: "❌ Anomaly" if x == -1 else "✅ Normal")

    # Build aligned text feature matrix for explanation using global row indexes.
    all_text_matrix = np.vstack([train_text_matrix, test_text_matrix])

    print(f"Found these suspicious patterns from train baseline: {', '.join(suspicious_patterns)}")
    print(
        "Anomaly rates -> "
        f"train: {(train_df['anomaly_score'].eq(-1).mean() * 100):.1f}%, "
        f"test: {(test_df['anomaly_score'].eq(-1).mean() * 100):.1f}%"
    )

    return combined_df, suspicious_patterns, all_text_matrix, feature_names


# Run train/test anomaly detection
df, suspicious_words, text_features, all_features = train_test_anomaly_detection(df)

# My reasoning system - explains why something looks suspicious
logger_counts = df["logger"].value_counts()
top_logger_count = logger_counts.iloc[0] if len(logger_counts) else 1


def explain_why_anomaly(log_row, row_idx):
    """
    Figure out why this log entry was flagged
    This helps me understand what the system is thinking
    """
    if log_row["status"] == "✅ Normal":
        return "Looks normal to me"
    
    explanations = []
    
    # Check what text patterns triggered it
    row_features = text_features[row_idx]
    top_feature_idx = np.argsort(row_features)[-4:]  # Top 4 patterns
    triggered_patterns = [all_features[i] for i in top_feature_idx if row_features[i] > 0]
    
    if triggered_patterns:
        explanations.append(f"Suspicious patterns found: {', '.join(triggered_patterns)}")
    
    # Cheeck if message length is weird
    split_df = df[df["split"] == log_row["split"]]
    avg_len = split_df["msg_len"].mean()
    std_len = split_df["msg_len"].std()
    if abs(log_row["msg_len"] - avg_len) > 2 * std_len:
        explanations.append("Message length is unusual")
    
    # High severity levels are always suspicious
    if log_row["level"] in ["ERROR", "CRITICAL", "FATAL"]:
        explanations.append(f"High severity: {log_row['level']}")

    if "Exception" in log_row["message"] or "failed" in log_row["message"].lower():
        explanations.append("Failure-oriented message pattern")

    if log_row["has_job_id"]:
        explanations.append("References Hadoop job or task identifiers")

    if logger_counts.get(log_row["logger"], 0) < max(2, top_logger_count * 0.005):
        explanations.append("Rare logger namespace")

    if log_row.get("if_flag", 0) == 1:
        explanations.append("IsolationForest score above train baseline threshold")

    if log_row.get("zscore_flag", 0) == 1:
        explanations.append("Numeric behavior deviates from train baseline")
    
    # Check for overlap with known suspicious words
    msg_words = set(log_row["message"].lower().split())
    matching_suspicious = msg_words.intersection(set(suspicious_words))
    if matching_suspicious:
        explanations.append(f"Contains suspicious terms: {', '.join(matching_suspicious)}")
    
    # Look for performance metrics (usually indicate problems)
    if "%" in log_row["message"] and any(c.isdigit() for c in log_row["message"]):
        explanations.append("Contains performance metrics")
    
    # Fallback explanation
    if not explanations:
        explanations.append("Multiple detection methods flagged this")
    
    return "; ".join(explanations)

# Generate explanations for each log entry
print("Generating explanations for detected anomalies...")
df["explanation"] = [explain_why_anomaly(row, idx) for idx, row in df.iterrows()]

# Sort by time to see the sequence of events
df = df.sort_values("timestamp").reset_index(drop=True)

# Show me what we found
anomaly_logs = df[df["status"] == "❌ Anomaly"]
print(f"\n🔍 Found {len(anomaly_logs)} suspicious log entries:")
if len(anomaly_logs) > 0:
    # Show first few anomalies with key info
    display_cols = ['timestamp', 'level', 'logger', 'message', 'explanation']
    print(anomaly_logs[display_cols].head(8).to_string(index=False))
else:
    print("No anomalies detected - system looks healthy!")

for split_name in ["train", "test"]:
    split_slice = df[df["split"] == split_name]
    split_anomalies = split_slice[split_slice["status"] == "❌ Anomaly"]
    print(
        f"{split_name.upper()} summary: {len(split_slice)} total, "
        f"{len(split_anomalies)} anomalies ({(len(split_anomalies) / len(split_slice)) * 100:.1f}%)"
    )

# My file saving functions - keep both detailed and summary reports
def save_full_analysis(data_df, filename="my_log_analysis.csv"):
    """Save everything - I like having all the data for later analysis"""
    try:
        filepath = os.path.join(output_dir, filename)
        data_df.to_csv(filepath, index=False)
        
        total = len(data_df)
        anomalies = len(data_df[data_df["status"] == "❌ Anomaly"])
        normal = total - anomalies
        
        print(f"\n💾 Full analysis saved to '{filepath}'")
        print(f"📊 Processed {total} log entries")
        print(f"✅ Normal: {normal} entries")
        print(f"❌ Suspicious: {anomalies} entries")
        
        if anomalies > 0:
            print(f"   Anomaly rate: {(anomalies/total)*100:.1f}%")

        for split_name in ["train", "test"]:
            split_slice = data_df[data_df["split"] == split_name]
            if len(split_slice) == 0:
                continue
            split_anomalies = len(split_slice[split_slice["status"] == "❌ Anomaly"])
            print(
                f"   {split_name}: {split_anomalies}/{len(split_slice)} "
                f"({(split_anomalies/len(split_slice))*100:.1f}%)"
            )
        
    except Exception as e:
        print(f"Error saving full report: {e}")

def save_just_anomalies(data_df, filename="suspicious_logs.csv"):
    """Save only the problematic entries - easier for quick review"""
    try:
        problem_logs = data_df[data_df["status"] == "❌ Anomaly"].copy()
        
        if len(problem_logs) > 0:
            # Keep the important columns for incident response
            important_cols = ['timestamp', 'level', 'thread', 'logger', 'message', 'status', 'explanation']
            filepath = os.path.join(output_dir, filename)
            problem_logs[important_cols].to_csv(filepath, index=False)
            
            print(f"🚨 Suspicious entries saved to '{filepath}'")
            print(f"   📋 {len(problem_logs)} entries need attention")
            
            # Quick summary of what we found
            level_counts = problem_logs['level'].value_counts()
            print(f"   Breakdown: {level_counts.to_dict()}")
        else:
            print("✅ No suspicious entries found - nothing to save")
            
    except Exception as e:
        print(f"Error saving anomaly report: {e}")

# Save both reports - I find both useful
print("\nSaving analysis results...")
save_full_analysis(df)
save_just_anomalies(df)
save_full_analysis(df[df["split"] == "train"], filename="train_log_analysis.csv")
save_full_analysis(df[df["split"] == "test"], filename="test_log_analysis.csv")
save_just_anomalies(df[df["split"] == "test"], filename="test_suspicious_logs.csv")

print("\n" + "="*50)
print("Analysis complete! 🎉")
print("Check the CSV files for detailed results.")
print("="*50)
