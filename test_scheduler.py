from datetime import datetime
import os

log_dir = r"C:\btc_bot\logs"
os.makedirs(log_dir, exist_ok=True)

log_path = os.path.join(log_dir, "test_scheduler.log")

with open(log_path, "a", encoding="utf-8") as f:
    f.write(f"OK | test_scheduler ran at {datetime.now()}\n")

print("Scheduler test OK")