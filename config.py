PHARMA_SIC_CODES = [
    "2830", "2833", "2834", "2835", "2836", "8731", "8734"
]

SEC_HEADERS = {
    "User-Agent": "Jyotiraditya Garikipati jyothiraditya.g29@gmail.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

SEC_BASE_URL      = "https://data.sec.gov"
RAW_DATA_DIR      = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
MODELS_DIR        = "data/models"
LOGS_DIR          = "logs"
MIN_MARKET_CAP_M  = 50
CT_BASE_URL       = "https://clinicaltrials.gov/api/v2/studies"
FDA_BASE_URL      = "https://api.fda.gov/drug"