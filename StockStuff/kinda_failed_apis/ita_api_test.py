import time
from lxml import etree
import selenium.common.exceptions
from bs4 import BeautifulSoup
import mechanicalsoup as soup
import selenium as sl
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import requests


def basic_example():
    browser = soup.StatefulBrowser()
    browser.open("http://httpbin.org/forms/post")
    browser.select_form("form[action='/post']")

    browser["custname"] = "Tommy"
    browser["custtel"] = 12345
    browser["custemail"] = "apli@gmail.com"
    browser["size"] = "medium"
    browser["topping"] = "bacon"
    browser["delivery"] = "15:15"
    browser["comments"] = "Amazing!"

    browser.form.print_summary()

    response = browser.submit_selected()
    print(response.text)


def github_login():
    browser = soup.StatefulBrowser()
    browser.open("https://github.com/login")
    # print(browser.page.find_all('form'))
    browser.select_form()
    browser["login"] = "sabyasachi-choudhury"
    browser["password"] = "Sabya19sachi05"
    browser.form.print_summary()
    response = browser.submit_selected()
    print(browser.url)
    """I'm in!"""


def investopedia_test():
    browser = soup.StatefulBrowser()
    # browser.open("https://www.investopedia.com/auth/realms/investopedia/protocol/openid-connect/auth?client_id=finance-simulator&redirect_uri=https%3A%2F%2Fwww.investopedia.com%2Fsimulator%2Fportfolio&state=b84fedf8-bf8f-48a9-af0e-716c4da2d1c2&response_mode=fragment&response_type=code&scope=openid&nonce=33f3e243-9792-4c08-9d5c-e1fece3433e8")
    browser.open("https://www.investopedia.com/simulator/portfolio#state=f0ababfe-50fe-475c-8d13-4013bca8c5c6&session_state=8db3fc80-3130-46db-b2ae-612e6055a065&code=7d93960f-be1f-436a-8647-c329eb1e4263.8db3fc80-3130-46db-b2ae-612e6055a065.12bddc88-7575-48ef-99f9-f196203e4054")
    # browser.select_form()
    #
    # browser["username"] = "thegooddeatheater"
    # browser["password"] = "Sabya19sachi05"
    #
    # response = browser.submit_selected()
    # print(response.text)
    print(browser.page)
    browser.launch_browser()


def selenium_test(selene=True, msoup=False):
    url = "https://github.com/login"
    if selene:
        driver_path = r"C:\Users\Sabyasachi\Downloads\chromedriver_win32\chromedriver.exe"
        driver = webdriver.Chrome(executable_path=driver_path)
        driver.get(url)
        driver.find_element(By.ID, "password").send_keys("Sabya19sachi05")
        driver.find_element(By.ID, "login_field").send_keys("thegooddeatheater")
        driver.find_element(By.NAME, "commit").click()
        time.sleep(10)
    if msoup:
        browser = soup.StatefulBrowser()
        browser.open(url)
        # print(browser.page)
        print(browser.page.find_all('form'))


def selenium_investopedia():
    url = r"https://www.investopedia.com/auth/realms/investopedia/protocol/openid-connect/auth?client_id=finance-simulator&redirect_uri=https%3A%2F%2Fwww.investopedia.com%2Fsimulator%2Fportfolio&state=c1543adb-13e9-439b-a4e9-e29f01f38859&response_mode=fragment&response_type=code&scope=openid&nonce=139189eb-800a-4682-9b85-9ba2d6d91f21"
    driver_path = r"C:\Users\Sabyasachi\Downloads\chromedriver_win32\chromedriver.exe"
    driver = webdriver.Chrome(executable_path=driver_path)
    browser = soup.StatefulBrowser()

    driver.get(url)
    browser.open(url)

    driver.find_element(By.NAME, "username").send_keys("thegooddeatheater")
    driver.find_element(By.NAME, "password").send_keys("Sabya19sachi05")
    driver.find_element(By.NAME, "login").click()
    print(type(driver.page_source))

    time.sleep(5)


def stockrover(selene=True, msoup=False, **kwargs):
    url = "http://stockrover.com"
    if selene:
        driver_path = r"C:\Users\Sabyasachi\Downloads\chromedriver_win32\chromedriver.exe"
        driver = webdriver.Chrome(executable_path=driver_path)

        driver.get(url)
        driver.find_element(By.XPATH, "/html/body/div[1]/div/section[2]/div/ul/li[2]/a").click()
        driver.find_element(By.CSS_SELECTOR, "input[name='username']").send_keys("thegooddeatheater")
        driver.find_element(By.CSS_SELECTOR, "input[name='password']").send_keys("Sabya19sachi05")
        driver.find_element(By.CSS_SELECTOR, "button[name='Sign In']").click()

        def check_if_exist():
            killed = False
            while not killed:
                try:
                    driver.find_element(By.CSS_SELECTOR, "img[src~='/images/preloader.gif']")
                    print("there")
                    # driver.implicitly_wait(10)

                except selenium.common.exceptions.NoSuchElementException:
                    print("spinner killed")
                    killed = True

        time.sleep(10)
        check_if_exist()

        time.sleep(6)
        if "ticker_name" in kwargs.keys():
            ticker_name = kwargs["ticker_name"]
            driver.find_element(By.CSS_SELECTOR, "input[id='tickersearchbox-3071-inputEl']").send_keys(ticker_name)
            driver.find_element(By.CSS_SELECTOR, "input[id='tickersearchbox-3071-inputEl']").send_keys(Keys.RETURN)
            time.sleep(4)
            page_source = BeautifulSoup(driver.page_source)
            print(page_source.select("span[class='price-perf-price']"))

    if msoup:
        browser = soup.StatefulBrowser()
        browser.open(url)
        for x in browser.page.find_all('form'):
            print(x, '\n')


# stockrover(msoup=True, selene=False)
stockrover(ticker_name="WOLF")

# with open(r"StockStuff\rover_html.txt") as file:
#     html = BeautifulSoup(file.read())
# tables = html.select('table[data-boundview="gridview-1070"]')
# print(len(tables))
# for x in tables:
#     print(x.prettify(), '\n')
# print(html.prettify())
# xml = etree.HTML(str(html))
# print(xml.xpath('//*')[0].xml)

"""TO DO: USE TICKER ENTRANCE TO FIND STOCK PRICESAND STUFF"""