import mechanicalsoup
import requests
from bs4 import BeautifulSoup

"""Getting login page form"""
# request = requests.get("https://www.howthemarketworks.com/login")
# soup = BeautifulSoup(request.content)
# print(soup.prettify())

"""Logging in"""
# login_details_request = requests.post("https://www.howthemarketworks.com/login",
#                                       data={"thegooddeatheater": "#Sabya19sachi05"})
# print(login_details_request.status_code)
# soup = BeautifulSoup(login_details_request.content)

"""To test if logged in or not"""


# comp_request = requests.get("https://www.howthemarketworks.com/quotes/quotes?symbol=AAPL")
# print(comp_request.status_code)
# comp_soup = BeautifulSoup(comp_request.content)
# print(comp_soup.select("a[class='account']"))
# print(comp_soup.select("a[class='login active']"))
#
def test_logged_in(req):
    soup = BeautifulSoup(req.content)
    if soup.select("a[class='account']"):
        print("Logged in")
    if soup.select("a[class='login active']"):
        print("Not logged in")


#
#
# session = requests.Session()
# logging_in = session.post("https://www.howthemarketworks.com/login",
#                           data={"UserName": "thegooddeatheater", "Password": "#Sabya19sachi05"})
# print("login status", logging_in.status_code)
# trade_request = session.get("https://www.howthemarketworks.com/trading/equities")
# print("trade status", trade_request.status_code)
# test_logged_in(trade_request)

"""TO do
deal with stock pages, delays, and fuck aroyund with trade system
lack of time, focus on essentials"""

"""Getting to doing shit"""


class Account:
    session = requests.Session()

    def __init__(self, user_name, passcode):
        login = Account.session.post("https://www.howthemarketworks.com/login",
                                     data={"UserName": user_name, "Password": passcode})
        if login.status_code != 200:
            raise Exception(login.status_code)
        else:
            print("logged in")

    def get_stock_data(self, company):
        parameters = {"symbols": company}
        headers = {"origin": "https://www.howthemarketworks.com", "referer": "https://www.howthemarketworks.com/"}
        stock_data = Account.session.get("https://app.quotemedia.com/datatool/getEnhancedQuotes.json",
                                         params=parameters, headers=headers)
        print(stock_data.text)
        print(stock_data.json())


acc = Account("thegooddeatheater", "#Sabya19sachi05")
acc.get_stock_data("AAPL")
# Stock("SNOW")

# https://app.quotemedia.com/datatool/getEnhancedQuotes.json?symbols=MTTR&greek=true&timezone=true&afterhours=true&premarket=true&currencyInd=true&countryInd=true&marketstatus=true&chfill=ee69C1D1&chfill2=69C1D1&chln=69C1D1&chxyc=5F6B6E&newslang=&token=06d7901ac51acb11eaabbfbcd56b00d878f1ae17e8e658aef4500b65c5fbec2d
# https://app.quotemedia.com/datatool/getEnhancedQuotes.json?symbols=MSFT&greek=true&timezone=true&afterhours=true&premarket=true&currencyInd=true&countryInd=true&marketstatus=true&chfill=ee69C1D1&chfill2=69C1D1&chln=69C1D1&chxyc=5F6B6E&newslang=&token=06d7901ac51acb11eaabbfbcd56b00d878f1ae17e8e658aef4500b65c5fbec2d
# https://app.quotemedia.com/datatool/getEnhancedQuotes.json?symbols=AAPL&greek=true&timezone=true&afterhours=true&premarket=true&currencyInd=true&countryInd=true&marketstatus=true&chfill=ee69C1D1&chfill2=69C1D1&chln=69C1D1&chxyc=5F6B6E&newslang=&token=6762a7ae6a6c6ce1b4bb37aa7e340ddba02c23aecf442d0400d5e823ac3d601c
