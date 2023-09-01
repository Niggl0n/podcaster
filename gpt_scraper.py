import requests
from bs4 import BeautifulSoup
import re
import os

def scrape_article(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # article_text = soup.get_text()

    # headline = soup.find('h1', class_='c-entry-box--compact__title').get_text()
    body = soup.find('div', class_='c-entry-content').get_text()
    # article_text_ = headline + '\n' + body
    return body

def save_article(article_text, filename):
    with open(filename, 'w') as file:
        file.write(article_text)


def clean_text(text):
    # Remove content after "Sign up for the"
    text = re.split("Sign up for the", text, flags=re.IGNORECASE)[0]

    # Remove multiple line breaks
    text = re.sub(r'\n\s*\n', '\n\n', text)

    return text

def scrape_website(base_url):
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a', href=True)

    article_links = [link['href'] for link in links if re.match(r'https://www.theringer.com/nba/\d+/\d+/\d+/.*/.*', link['href'])]
    article_links_gen = [link['href'] for link in links if re.match(r'https://www.theringer.com/\d+/\d+/\d+/.*/.*', link['href'])]
    article_links_archive = [link['href'] for link in links if re.match(r'https://www.theringer.com/nba/archives/\d+/\d+/\d+/.*/.*', link['href'])]
    article_links = list(set(article_links + article_links_gen + article_links_archive))
    print(f"Found {len(article_links)} articles to scrape.")

    for i, article_link in enumerate(article_links):
        print(article_link)
        article_text = scrape_article(article_link)
        article_text_cleaned = clean_text(article_text)
        # article_id = re.search(r'https://www.theringer.com/nba/\d+/\d+/\d+/([^/]+)/.*', article_link).group(1)
        article_id = re.search(r'https://www.theringer.com/(?:nba/)?\d+/\d+/\d+/([^/]+)/.*', article_link).group(1)
        filename = os.path.join('articles', f'{article_id}.txt')
        save_article(article_text_cleaned, filename)


scrape_website('https://www.theringer.com/nba')