from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from time import sleep

# Create ChromeOptions object and set headless mode
chrome_options = Options()
chrome_options.add_argument('--headless')

driver = webdriver.Chrome(options=chrome_options)
wait = WebDriverWait(driver, 20)  # Wait up to 10 seconds


def pause():
    driver.get('http://mainsailos.local')

    # Wait for the pause button to appear
    pause_button = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="page-container"]/div/div/div[1]/div[1]/div/header/div/div[3]/div/button')))

    pause_button.click()

    driver.quit()
