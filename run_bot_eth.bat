@echo off
cd /d C:\btc_bot
call .venv\Scripts\activate.bat
set ENV_FILE=.env.ETH
python bot.py >> logs\run.log 2>&1