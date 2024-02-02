from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service

# Create ChromeOptions object and set headless mode
chrome_options = Options()
chrome_options.add_argument("--headless")

driver = webdriver.Chrome(
    options=chrome_options,
    service=Service(executable_path='/usr/lib/chromium-browser/chromedriver')
)

wait = WebDriverWait(driver, 6)  # Wait up to 10 seconds

driver.get("http://mainsailos.local")


def is_printing():
    # Wait for the status to appear
    status = wait.until(
        EC.presence_of_element_located(
            (
                By.XPATH,
                '//*[@id="page-container"]/div/div/div[1]/div[1]/div/header/div/div[1]/span',
            )
        )
    )

    if str(status.text) == "":
        status = wait.until(
            EC.presence_of_element_located(
                (
                    By.XPATH,
                    '//*[@id="page-container"]/div/div/div[1]/div[1]/div/header/div/div[1]/span[2]',
                )
            )
        )

    return "Printing" in status.text
