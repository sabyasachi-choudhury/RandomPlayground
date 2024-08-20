"""API for HTMW

Features: View portfolio, home page popular stocks
View details of any stock you want
Trade"""
import time
import warnings
import re
from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
from bs4 import BeautifulSoup
import mechanicalsoup

"""Reload page in case of page_not_found_error"""
def reload():
    try:
        while Account.driver.find_element(By.CSS_SELECTOR, "button[id='reload-button']"):
            Account.driver.find_element(By.CSS_SELECTOR, "button[id='reload-button']").click()
            time.sleep(0.25)
    except NoSuchElementException:
        pass


"""Halt code until a certain element is found"""
def until_elem_found(selector_type, selector):
    try:
        try:
            if Account.driver.find_element(selector_type, selector):
                pass
        except NoSuchElementException:
            until_elem_found(selector_type, selector)
    except RecursionError:
        until_elem_found(selector_type, selector)


"""Helper function to detect if an elem exists or not"""
def elem_exists(selector_type, selector):
    try:
        if Account.driver.find_element(selector_type, selector):
            return True
    except NoSuchElementException:
        return False


"""Helper function to kill sticky ads which might obstruct clicks"""
def kill_sticky_ad():
    try:
        Account.driver.find_element(By.CSS_SELECTOR, "button[onclick=\"$('.stickyad').hide()\"]").click()
    except NoSuchElementException:
        pass


"""to initialize driver"""
def driver_init(do=True):
    if do:
        driver_path = r"C:\Users\Sabyasachi\Downloads\chromedriver_win32\chromedriver.exe"
        driver = Chrome(executable_path=driver_path)
        return driver

"""to find with bs4"""
def bs4_find(source, selector)


"""Primary class, inside which everything happens"""
class Account:

    driver = driver_init()

    """Init, which just logs in to account"""
    def __init__(self, user_name, passcode):
        Account.driver.get("https://www.howthemarketworks.com/login")
        reload()
        Account.driver.find_element(By.CSS_SELECTOR, "input[id='UserName']").send_keys(user_name)
        Account.driver.find_element(By.CSS_SELECTOR, "input[id='Password']").send_keys(passcode)
        Account.driver.find_element(By.CSS_SELECTOR, "input[type='submit']").click()
        reload()

    """list all contests in which you're participating"""
    def list_contests(self):
        until_elem_found(By.CSS_SELECTOR, 'select[id="ddlTournaments"]')
        contest_list = []
        for i, li in enumerate(Account.driver.find_elements(By.XPATH, '//*[@id="ddlTournaments"]/option'), 1):
            contest_list.append(Account.driver.find_element(By.XPATH, '//*[@id="ddlTournaments"]/option'+'['+str(i)+']').text)
        return contest_list

    """Change the current contest"""
    def change_contest(self, new_cont):
        cont_list = self.list_contests()
        option_path = '//*[@id="ddlTournaments"]/option'
        cont_rel_dict = {cont_list[i]: option_path+'['+str(i+1)+']' for i in range(len(cont_list))}
        Account.driver.find_element(By.CSS_SELECTOR, "select[id='ddlTournaments']").click()
        try:
            Account.driver.find_element(By.XPATH, cont_rel_dict[new_cont]).click()
        except KeyError:
            raise Exception("NoSuchContest")

    """Subclass to deal with searches for information on specific stock"""
    class Stock:
        """Just searches for a specified stock"""

        def __init__(self, symbol):
            until_elem_found(selector_type=By.CSS_SELECTOR, selector="input[name='keywords']")
            Account.driver.find_element(By.CSS_SELECTOR, "input[name='keywords']").send_keys(symbol)
            Account.driver.find_element(By.CSS_SELECTOR, "input[name='keywords']").send_keys(Keys.RETURN)

        """Find most useful info of the stock, like current_price, today's change,
        the change percentage, and the time of these observations."""

        def current_stats(self):
            until_elem_found(By.XPATH,
                             '//*[@id="qmodTool"]/div/div[1]/div/div[2]/div/div/div[1]/div[3]/span[1]/span[1]/span[1]')
            price = Account.driver.find_element(By.XPATH,
                                                '//*[@id="qmodTool"]/div/div[1]/div/div[2]/div/div/div[1]/div[1]/span[1]').text
            change = Account.driver.find_element(By.XPATH,
                                                 '//*[@id="qmodTool"]/div/div[1]/div/div[2]/div/div/div[1]/div[1]/span[2]/span[2]').text
            change_percent = Account.driver.find_element(By.XPATH,
                                                         '//*[@id="qmodTool"]/div/div[1]/div/div[2]/div/div/div[1]/div[1]/span[2]/span[4]').text
            date_time = Account.driver.find_element(By.XPATH,
                                                    '//*[@id="qmodTool"]/div/div[1]/div/div[2]/div/div/div[1]/div[3]/span[1]/span[1]/span[1]').text
            return {"current_price": price, "change": change, "change%": change_percent, "date_time": date_time}

        """Finds mentioned chart of the stock. Options for chart - [1d, 5d, 1m, 3m, 1y, 5y]
        Can save as well"""
        def chart(self, period, save=False, **kwargs):
            kill_sticky_ad()
            until_elem_found(selector_type=By.XPATH, selector="button[rv-class-qmod-btn-active*='" + period + "']")
            Account.driver.find_element(By.CSS_SELECTOR, "button[rv-class-qmod-btn-active*='" + period + "']").click()

            until_elem_found(selector_type=By.XPATH, selector='//*[@id="qmodTool"]/div/div[2]/div/div[1]/div/div/img')
            img_url = Account.driver.find_element(By.XPATH,
                                                  '//*[@id="qmodTool"]/div/div[2]/div/div[1]/div/div/img').get_attribute(
                "src")

            temp_driver = Chrome(executable_path=Account.driver_path)
            temp_driver.get(img_url)
            if save:
                with open(kwargs["save_file"], "w") as file:
                    file.write(temp_driver.page_source)
            temp_driver.maximize_window()
            temp_driver.implicitly_wait(kwargs["display_time"])
            temp_driver.close()

        """Finds extra details of the stock, like volume, last price, so on. 
        Returns dictionary containing all this dataa"""
        def details(self):
            pre_root = '//*[@id="qmodTool"]/div/div[2]/div/div[2]/div/div[1]/div'
            until_elem_found(selector_type=By.XPATH, selector=pre_root)
            labels, values = [], []
            for i0, elem0 in enumerate(Account.driver.find_elements(By.XPATH, pre_root + '/div'), 1):
                root_path = pre_root + '/div[' + str(i0) + ']/div'
                for i1, elem1 in enumerate(Account.driver.find_elements(By.XPATH, root_path + '/div'), 1):
                    labels.append(
                        Account.driver.find_element(By.XPATH, root_path + '/div[' + str(i1) + ']/div[1]/div').text)
                    values.append(
                        Account.driver.find_element(By.XPATH, root_path + '/div[' + str(i1) + ']/div[2]/div').text)
            return {labels[i]: values[i] for i in range(len(labels))}

    """Sub class for handling portfolio stuff"""

    class Portfolio:
        """Init, which just takes the driver to the portfolio page"""
        def __init__(self):
            reload()
            Account.driver.find_element(By.XPATH, '//*[@id="responsive-menu"]/div/div/ul/li[2]/a').click()

        """Function to return all open positions and their details in dictionary form"""
        def open_positions(self):
            until_elem_found(By.XPATH, '//*[@id="tOpenPositions_equities"]/tbody')
            details = {}

            for i, row in enumerate(
                    Account.driver.find_elements(By.XPATH, '//*[@id="tOpenPositions_equities"]/tbody/tr'), 1):
                sub_details = {}
                row_path = '//*[@id="tOpenPositions_equities"]/tbody/tr[' + str(i) + ']'
                sub_details["Quantity"] = Account.driver.find_element(By.XPATH, row_path + '/td[3]').text
                sub_details["Price Paid"] = Account.driver.find_element(By.XPATH, row_path + '/td[4]').text
                sub_details["Last Price"] = Account.driver.find_element(By.XPATH, row_path + '/td[5]').text
                sub_details["Day's Change"] = Account.driver.find_element(By.XPATH, row_path + '/td[6]').text
                sub_details["Market Value"] = Account.driver.find_element(By.XPATH, row_path + '/td[7]').text
                sub_details["Profit/Loss"] = Account.driver.find_element(By.XPATH, row_path + '/td[8]/span').text
                sub_details["Proft/Loss%"] = Account.driver.find_element(By.XPATH, row_path + '/td[8]/small').text[1:-2]
                symbol = Account.driver.find_element(By.XPATH, row_path + '/td[1]/a').text
                details[symbol] = sub_details

            return details

        def portfolio_summary(self):
            # until_elem_found(By.XPATH, '/html/body/section[2]/div/div/div/div[1]/div/div/ul')
            portfolio_value = Account.driver.find_element(By.CSS_SELECTOR, 'li[id="portfolioValue"]').get_attribute('innerHTML')
            cash_bal = Account.driver.find_element(By.CSS_SELECTOR, "li[id='cashBalance']").get_attribute('innerHTML')
            buy_power = Account.driver.find_element(By.CSS_SELECTOR, 'li[id="buyingPower"]').get_attribute('innerHTML')

            find_val = lambda html: re.findall("\s*\$(.*)\n", html)[0]
            print(portfolio_value, cash_bal, buy_power)
            portfolio_value = find_val(portfolio_value)
            cash_bal = find_val(cash_bal)
            buy_power = find_val(buy_power)
            return {"portfolio_value": portfolio_value, "cash_balance": cash_bal, "buying_power": buy_power}

    """Sub class to deal with trade stuff"""
    class Trade:
        """Init, which opens trade tab and checks if market is open or not"""

        def __init__(self):
            until_elem_found(By.XPATH, '//*[@id="responsive-menu"]/div/div/ul/li[3]/a')
            Account.driver.find_element(By.XPATH, '//*[@id="responsive-menu"]/div/div/ul/li[3]/a').click()
            try:
                if Account.driver.find_element(By.CSS_SELECTOR, "div[id='market-closed']"):
                    warnings.warn("MarketNotOpen")
                    # self.market_state = "Closed"
                    self.market_state = "Open"
            except NoSuchElementException:
                self.market_state = "Open"
            reload()
            kill_sticky_ad()

        """Function to trade stocks
        Action, symbol, quantity, and trade_type are compulsory args
        Of these four, quantity is a +ve int, while the rest are strings
        
        In kwargs, limit, date, and order_term are strings to be supplied with the appropriate
        trade_types.
        Limit is an int, order_term is a string, and date is a string of form 'mm/dd/yyyy']
        
        def stock_trade(self, action: str, symbol: str, quantity: int, trade_type: str, limit: int, order_term: str, date: str):"""
        def stock_trade(self, action: str, symbol: str, quantity: int, trade_type: str, preview=True, confirm=True,
                        **kwargs):
            if self.market_state == "Closed":
                warnings.warn("""The Market is closed now. Any trade orders will be executed as soon as the market opens
                              tomorrow. Are you sure you want to trade now?""")

            """Dealing with the action slot"""
            Account.driver.find_element(By.CSS_SELECTOR, 'select[id="ddlOrderSide"]').click()
            action_rel_dict = {"buy": '//*[@id="ddlOrderSide"]/option[1]',
                               "sell": '//*[@id="ddlOrderSide"]/option[2]',
                               "short": '//*[@id="ddlOrderSide"]/option[3]',
                               "cover": '//*[@id="ddlOrderSide"]/option[4]'}
            Account.driver.find_element(By.XPATH, action_rel_dict[action.lower()]).click()

            """Dealing with Symbol slot"""
            Account.driver.find_element(By.CSS_SELECTOR, "input[id='tbSymbol']").send_keys(symbol)
            Account.driver.find_element(By.CSS_SELECTOR, "input[id='tbSymbol']").send_keys(Keys.RETURN)

            """Dealing with quantity slot"""
            Account.driver.find_element(By.CSS_SELECTOR, "input[id='tbQuantity']").send_keys(quantity)
            Account.driver.find_element(By.CSS_SELECTOR, "input[id='tbQuantity']").send_keys(Keys.RETURN)

            """Dealing with type slot"""
            Account.driver.find_element(By.CSS_SELECTOR, "select[id='ddlOrderType']").click()
            type_rel_dict = {"market": '//*[@id="ddlOrderType"]/option[1]',
                             "limit": '//*[@id="ddlOrderType"]/option[2]',
                             "stop": '//*[@id="ddlOrderType"]/option[3]',
                             "trailing_stop_$": '//*[@id="ddlOrderType"]/option[4]',
                             "trailing_stop_%": '//*[@id="ddlOrderType"]/option[5]'}
            Account.driver.find_element(By.XPATH, type_rel_dict[trade_type.lower()]).click()

            """Dealing with limit slot"""
            if trade_type.lower() in ["limit", "stop", "trailing_stop_$", "trailing_stop_%"]:
                Account.driver.find_element(By.CSS_SELECTOR, "input[id='tbPrice']").send_keys(kwargs["limit"])
                Account.driver.find_element(By.CSS_SELECTOR, "input[id='tbPrice']").send_keys(Keys.RETURN)

            """Dealing with order_term slot"""
            if trade_type.lower() in ["limit", "stop", "trailing_stop_$", "trailing_stop_%"]:
                Account.driver.find_element(By.CSS_SELECTOR, "select[id='ddlOrderExpiration']").click()
                term_rel_dict = {"good-til-day": '//*[@id="ddlOrderExpiration"]/option[1]',
                                 "good-til-cancel": '//*[@id="ddlOrderExpiration"]/option[2]',
                                 "good-til-date": '//*[@id="ddlOrderExpiration"]/option[3]'}
                Account.driver.find_element(By.XPATH, term_rel_dict[kwargs["order_term"].lower()]).click()

            """Dealing with date slot"""
            if trade_type.lower() in ["limit", "stop", "trailing_stop_$", "trailing_stop_%"] and \
                    kwargs["order_term"].lower() == "good-til-date":
                Account.driver.find_element(By.CSS_SELECTOR, "input[id='pGTDateCalendar']").send_keys(kwargs["date"])
                Account.driver.find_element(By.CSS_SELECTOR, "input[id='pGTDateCalendar']").send_keys(Keys.RETURN)

            """Previews"""
            Account.driver.find_element(By.CSS_SELECTOR, "a[id='btn-preview-order']").click()
            # checks if order is valid, if not, raises exception stating cause.
            if elem_exists(By.XPATH, '//*[@id="vs-trading"]/ul/li[2]'):
                raise Exception(Account.driver.find_element(By.XPATH, '//*[@id="vs-trading"]/ul/li[2]').text)
            # waits till preview found, then appends it to output array.
            until_elem_found(By.XPATH, '//*[@id="tradebuttons-container"]/div/p')
            outputs = [Account.driver.find_element(By.XPATH, '//*[@id="tradebuttons-container"]/div/p').text,
                       "Estimated Cost: " + Account.driver.find_element(By.XPATH,
                                                                        '//*[@id="tradebuttons-container"]/div/h3/span').text]
            # if trade is confirmed, appends confirmation message as well
            if confirm:
                until_elem_found(By.CSS_SELECTOR, "a[id='btn-place-order']")
                Account.driver.find_element(By.CSS_SELECTOR, "a[id='btn-place-order']").click()
                outputs.append("TradeConfirmed")
            else:
                outputs.append("TradeNotConfirmed")
            return outputs


acc = Account(user_name="thegooddeatheater", passcode="#Sabya19sachi05")
port = acc.Portfolio()
print(port.portfolio_summary())

# x = '<b>Portfolio Value:</b>\n                                <i class="fa fa-arrow-circle-up" aria-hidden="true"></i>\n                                $10,000.00\n                            '
# # print(re.findall("<.*>(.*)</.*>", x))
# print(re.findall("\s*\$(.*)\n", x))