#!/usr/bin/env python3
import subprocess
import sys
import datetime
import os
import re

LOG_PATH = os.path.join(os.path.dirname(__file__), "quartz", "content", "log.md")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "quartz", "content", "index.md")

def run_cmd(command):
    """Runs a shell command and prints its output."""
    print(f"➜ Running: {command}")
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"❌ Error executing '{command}':\n{result.stderr}")
        sys.exit(result.returncode)
    
    if result.stdout.strip():
        print(result.stdout.strip())
    return result

def get_today_log_entries():
    """Read log.md and return all entries for today as a single string."""
    today = datetime.date.today().strftime("%Y-%m-%d")
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return None

    entries = [
        line.strip()
        for line in lines
        if re.match(rf"^{re.escape(today)}", line.strip())
    ]
    return " | ".join(entries) if entries else None

def update_index_latest_update_date(date_str):
    """Update the Latest Update date on index.md."""
    try:
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"⚠️ Could not update Latest Update date: {INDEX_PATH} not found.")
        return

    updated_content, count = re.subn(
        r"(\*\*Latest Update\*\*:\s*)\d{4}-\d{2}-\d{2}",
        rf"\g<1>{date_str}",
        content,
        count=1,
    )
    if count:
        with open(INDEX_PATH, "w", encoding="utf-8") as f:
            f.write(updated_content)
        print(f"Updated Latest Update date to {date_str}.")
    else:
        print("⚠️ Could not find Latest Update date in index.md.")

def main():
    """Automates git add, commit, and push."""
    # Step 1: Use today's log entries or a timestamped fallback
    log_msg = get_today_log_entries()
    if log_msg:
        update_index_latest_update_date(datetime.date.today().strftime("%Y-%m-%d"))
        commit_msg = log_msg
    else:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        commit_msg = f"Auto-sync: {timestamp}"

    # Step 2: Check for changes
    status = subprocess.run("git status --short", shell=True, text=True, capture_output=True)
    if not status.stdout.strip():
        print("✅ No changes found. Everything is already up to date!")
        sys.exit(0)

    print(f"Found changes:\n{status.stdout}")

    # Step 3: Add all changes
    run_cmd("git add .")
    
    # Step 4: Commit with the message (using list format to safely handle quotes/spaces)
    print(f"➜ Running: git commit -m \"{commit_msg}\"")
    commit_result = subprocess.run(["git", "commit", "-m", commit_msg])
    if commit_result.returncode != 0:
        print("❌ Failed to commit. Aborting.")
        sys.exit(commit_result.returncode)

    # Step 5: Push to Github
    print("\nPushing to GitHub (origin main)...")
    run_cmd("git push origin main")
    
    print("\n🚀 Sync complete! Your wiki will rebuild shortly.")

if __name__ == "__main__":
    main()
