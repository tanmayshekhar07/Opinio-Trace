#source "/Users/riyavirani/NEU/Gen AI/amazon-python-scrapy-scraper-master/venv/bin/activate"

import scrapy
from urllib.parse import urljoin

class AmazonReviewsSpider(scrapy.Spider):
    name = "amazon_reviews"

    custom_settings = {
        'FEEDS': { 
            's3://damg7374-storage-files/reviews.csv': { 'format': 'csv',}
            }
    }

    def start_requests(self):
        asin_list = [self.asin]
        for asin in asin_list:
            amazon_reviews_url = f'https://www.amazon.com/product-reviews/{asin}/'
            yield scrapy.Request(url=amazon_reviews_url, callback=self.parse_reviews, meta={'asin': asin, 'retry_count': 0, 'page_number': 1})

    def parse_reviews(self, response):
        asin = response.meta['asin']
        retry_count = response.meta.get('retry_count', 0)  # Initialize retry_count if it doesn't exist

        next_page_relative_url = response.css('a[rel="nofollow"][class="pn-next"]::attr(href)').get()

        if next_page_relative_url:
            retry_count = 0  # Reset retry_count when the "Next page" link is found

        if retry_count < 15:
            if next_page_relative_url:
                next_page = response.urljoin(next_page_relative_url)
                yield scrapy.Request(url=next_page, callback=self.parse_reviews, meta={'asin': asin, 'retry_count': retry_count})
            else:
                retry_count = retry_count + 1
                yield scrapy.Request(url=response.url, callback=self.parse_reviews, dont_filter=True, meta={'asin': asin, 'retry_count': retry_count})

        # Parse Product Reviews
        review_elements = response.css("#cm_cr-review_list div.review")
        for review_element in review_elements:
            yield {
                "asin": asin,
                "text": "".join(review_element.css("span[data-hook=review-body] ::text").getall()).strip(),
                "title": review_element.css("*[data-hook=review-title]>span::text").get(),
                "location_and_date": review_element.css("span[data-hook=review-date] ::text").get(),
                "verified": bool(review_element.css("span[data-hook=avp-badge] ::text").get()),
                "rating": review_element.css("*[data-hook*=review-star-rating] ::text").re(r"(\d+\.*\d*) out")[0],
            }
