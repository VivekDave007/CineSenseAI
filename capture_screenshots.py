import os
import time
import subprocess
import sys

# Ensure selenium is installed
try:
    from selenium import webdriver
except ImportError:
    print("Installing selenium...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "selenium", "webdriver-manager"])
    from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.edge.service import Service

print("Starting Edge in headless mode...")
options = webdriver.EdgeOptions()
options.add_argument('--headless')
options.add_argument('--window-size=1920,1080')
options.add_argument('--disable-gpu')

try:
    driver = webdriver.Edge(options=options)
except Exception as e:
    print("Edge driver failed. Trying to install via webdriver-manager...", e)
    from webdriver_manager.microsoft import EdgeChromiumDriverManager
    service = Service(EdgeChromiumDriverManager().install())
    driver = webdriver.Edge(service=service, options=options)

try:
    print("Navigating to local Streamlit app...")
    driver.get("http://localhost:8502")

    print("Waiting 12 seconds for Streamlit Server connection & initial layout to render...")
    time.sleep(12)

    # Capture main empty dashboard
    driver.save_screenshot(r"d:\ML Project for Entertainment\media\dashboard_overview.png")
    print("Saved primary screenshot: dashboard_overview.png")
    
    # Save a duplicated one just in case we can't get interaction to work
    driver.save_screenshot(r"d:\ML Project for Entertainment\media\main_dashboard.png")

    print("Attempting to interact with the chat input...")
    try:
        chat_input = driver.find_element(By.CSS_SELECTOR, '[data-testid="stChatInput"] textarea')
        chat_input.send_keys("Filter the Top 5 Sci-Fi movies for me.")
        time.sleep(1)
        chat_input.send_keys(Keys.ENTER)
        
        print("Sent chat query. Waiting 15 seconds for ML model inference and UI response...")
        time.sleep(15)
        
        driver.save_screenshot(r"d:\ML Project for Entertainment\media\wikipedia_recommender.png")
        print("Saved interaction screenshot: wikipedia_recommender.png")
    except Exception as e:
        print("Could not interact with chat input. Skipping interactive screenshot.", e)

finally:
    driver.quit()
    print("Selenium browser closed.")
