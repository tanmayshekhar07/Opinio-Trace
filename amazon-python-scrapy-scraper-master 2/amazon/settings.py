
BOT_NAME = 'amazon'

SPIDER_MODULES = ['amazon.spiders']
NEWSPIDER_MODULE = 'amazon.spiders'

# Obey robots.txt rules
ROBOTSTXT_OBEY = False

SCRAPEOPS_API_KEY = '3e791168-d3bf-49a2-a858-484ce6c42bee'
SCRAPEOPS_PROXY_ENABLED = True



LOG_LEVEL = 'INFO'

DOWNLOADER_MIDDLEWARES = {

    ## ScrapeOps Monitor
    'scrapeops_scrapy.middleware.retry.RetryMiddleware': 550,
    'scrapy.downloadermiddlewares.retry.RetryMiddleware': None,
    

    ## Proxy Middleware
    'scrapeops_scrapy_proxy_sdk.scrapeops_scrapy_proxy_sdk.ScrapeOpsScrapyProxySdk': 725,
}

# Max Concurrency On ScrapeOps Proxy Free Plan is 1 thread
CONCURRENT_REQUESTS = 1

AWS_ACCESS_KEY_ID = 'AKIAZCUUL3YW7CR3FAH2'
AWS_SECRET_ACCESS_KEY = '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'

AIRFLOW_USERNAME = 'admin'
AIRFLOW_PASSWORD = "say my name"
