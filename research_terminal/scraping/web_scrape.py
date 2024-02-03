from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support import wait
from bs4 import BeautifulSoup
from requests.compat import urljoin
from research_terminal.logger.logger import logger

class WebScraper:
    def __init__(self, browser):
        self.driver = self.get_driver(browser)
        logger.debug(f"WebScraper initialized with {browser} browser")
        
    def get_driver(self, browser: str):
        drivers = {
            "chrome": webdriver.Chrome,
            "firefox": webdriver.Firefox
        }
        if browser not in drivers:
            logger.error(f"{browser} is not a supported browser")
            raise Exception(f"{browser} is not a supported browser")
        
        options = ChromeOptions() if browser == "chrome" else FirefoxOptions()
        options.add_argument("--headless")
        options.add_argument("--enable-javascript")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        logger.debug(f"Getting {browser} driver")
        return drivers[browser](options=options)
        
    def scrape(self, url: str):
        soup = self.get_soup(url)
        return self.get_text(soup)
    
    def get_soup(self, url: str):
        logger.debug(f"Getting soup from {url}")
        self.driver.get(url)
        wait.WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))    
        page_source = self.driver.execute_script("return document.body.outerHTML;")
        soup = BeautifulSoup(page_source, 'html.parser')
        logger.debug(f"Got soup from {url}")
        return soup
    def get_text(self, soup: BeautifulSoup):
        """Get the text from the soup
    
        Args:
            soup (BeautifulSoup): The soup to get the text from
    
        Returns:
            str: The text from the soup
        """
        logger.debug("Getting text from soup")
        text = ""
        tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'p']
        for element in soup.find_all(tags):
            text += element.text + "\n\n"
        return text
    
    def extract_hyperlinks(self, soup: BeautifulSoup, url: str):
        """Extract the hyperlinks from the soup
        Args:
            soup (BeautifulSoup): The soup to extract the hyperlinks from
            url (str): The url of the page
        Returns:
            List[Tuple[str, str]]: The hyperlinks
        """
        logger.info("Extracting hyperlinks from soup")
        hyperlinks = []
        for link in soup.find_all("a", href=True):
            link_text = link.text.strip()
            link_url = urljoin(url, link["href"])
            if link_text and link_url:
                hyperlinks.append((link_text, link_url))
        return hyperlinks
    
    def format_hyperlinks(self, hyperlinks: list[tuple[str, str]]) -> list[str]:
        """Format hyperlinks to be displayed to the user
    
        Args:
            hyperlinks (List[Tuple[str, str]]): The hyperlinks to format
    
        Returns:
            List[str]: The formatted hyperlinks
        """
        logger.info("Formatting hyperlinks")
        return [f"{link_text} ({link_url})" for link_text, link_url in hyperlinks]
    
    def scrape_links(self, url: str):
        soup = self.get_soup(url)
        hyperlinks = self.extract_hyperlinks(soup, url)
        return self.format_hyperlinks(hyperlinks)
    
    
    def quit(self):
        logger.info("Quitting driver")
        self.driver.quit()
        


    
    
    