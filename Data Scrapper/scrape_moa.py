"""
scrape_moa.py
-------------
Scrapes daily commodity price data from the Agricultural Service Portal.
"""

import csv
import os
import re
import time
from datetime import date, timedelta

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager


# ============================================================
# CONFIG
# ============================================================

LOCATIONS = [
    # {
    #     "division": "Khulna",
    #     "district": "Khulna",
    #     "upazila": "Khulna City Corporation",
    #     "market": "Khulna Sadar Bazar",
    # },
    # {
    #         "division": "Khulna",
    #         "district": "Khulna",
    #         "upazila": "Dumuria",
    #         "market": "Dumuria Bazar",
    # }
    # {
    #     "division": "Dhaka",
    #     "district": "Dhaka",
    #     "upazila": "Dhaka North City Corporation",
    #     "market": "Mohammadpur Krishi Market",
    # },
    # {
    #         "division": "Dhaka",
    #         "district": "Narayanganj",
    #         "upazila": "Narayanganj Sadar",
    #         "market": "Narayanganj Sadar Bazar",
    # }
    {
            "division": "Dhaka",
            "district": "Shariatpur",
            "upazila": "Shariatpur Sadar",
            "market": "Shariatpur Bazar",
    }
    
    # {
    #     "division": "Dhaka",
    #     "district": "Dhaka",
    #     "upazila": "Savar",
    #     "market": "Savar Bazar",
    # },
    # {
    #     "division": "Rangpur",
    #     "district": "Rangpur",
    #     "upazila": "Rangpur City Corporation",
    #     "market": "City Bazar, Rangpur",
    # },
    # {
    #         "division": "Rangpur",
    #         "district": "Rangpur",
    #         "upazila": "Mithapukur",
    #         "market": "Mithapukur Bazar",
    # }
    #  {
    #     "division": "Barisal",
    #     "district": "Pirojpur",
    #     "upazila": " Pirojpur Sadar",
    #      "market": "Pirojpur Sadar Bazar",
    #  },
    # {
    #     "division": "Sylhet",
    #     "district": "Sylhet",
    #     "upazila": "Beanibazar",
    #     "upazila": "Beanibazar",
    #     "market": "Beanibazar Market",
    # }
    # {
    #     "division": "Rajshahi",
    #     "district": "Rajshahi",
    #     "upazila": "Rajshahi City Corporation",
    #     "market": "Rajshahi Sadar Bazar",
    # },
    # {
    #         "division": "Rajshahi",
    #         "district": "Rajshahi",
    #         "upazila": "Godagari",
    #         "market": "Godagari Sadar Bazar",
    #     }
    # {
    #     "division": "Mymensingh",
    #     "district": "Mymensingh",
    #     "upazila": "Mymensingh Sadar",
    #     "market": "Mechua Bazar, Mymensingh",
    # },
    # {
    #     "division": "Mymensingh",
    #     "district": "Mymensingh",
    #     "upazila": "Fulbaria",
    #     "market": "Fulbaria Bazar",
    # }
    # {
    #     "division": "Chattagram",
    #     "district": "Chattogram",
    #     "upazila": "Chattogram City Corporation",
    #     "market": "Chittagong Sadar Bazar",
    # }
]

COMMODITY_GROUPS = ["Egg", "Fish", "Meat", "Oil", "Milk", "Dal", "Spices", "Poultry", "Vegetable", "Foodgrains"]
COMMODITY_SUB_GROUPS = ["Hen", "Pangash", "Cock / Hen", "Others", "Powder Milk", "Masur", "Onion", "Ginger", "Garlic", "Green Chili", "Potato", "Ata", "Rice"]
COMMODITY_NAMES = ["Egg Farm-Red", "Broiler chicken", "Pangash (big)", "Soybean Oil(loose)", "Palm Oil", "Cow Milk (liquid)", "Lentils - Desi-Whole", "Beef", "Ginger (Imported)", "Green Chili (Local)", "Farm-raised Hen", "Potato (Holland) - Red", "Ata (loose) - White", "Ata (Packet)", "Rice - Boro - Hybrid - Medium", "Rice-Boro-Hybrid -Fine", "Garlic (local) - Big Size", "Garlic (Imported)", "Onion (local)"]

# COMMODITY_GROUPS = ["Poultry"] #, ["Egg", "Fish", "Meat", "Oil", "Milk", "Dal", "Spices", "Vegetable", "Foodgrains"]
# COMMODITY_SUB_GROUPS = ["Hen"] #, "Pangash", "Cock / Hen", "Others", "Powder Milk", "Masur", "Onion", "Ginger", "Garlic", "Green Chili", "Potato", "Ata", "Rice"]

# COMMODITY_NAMES = ["Broiler chicken"] #, "Pangash (big)", "Soybean Oil(loose)", "Palm Oil", "Cow Milk (liquid)", "Lentils - Desi-Whole", "Beef", "Ginger (Imported)", "Green Chili (Local)", "Farm-raised Hen", "Potato (Holland) - Red", "Ata (loose) - White", "Ata (Packet)", "Rice - Boro - Hybrid - Medium", "Rice-Boro-Hybrid -Fine", "Garlic (local) - Big Size", "Garlic (Imported)", "Onion (local)"]

CONFIG = {
    "url": "https://moa-services.com/market-directory/product-wise-market-price-report",
    "headless": False,
    "field_ids": {
        "division": "division_id",
        "district": "district_id",
        "upazila": "upazila_id",
        "market": "market_id",
        "commodity_group": "com_grp_id",
        "commodity_sub_group": "com_subgrp_id",
        "commodity_name": "commodity_id",
    },
    "output_csv": os.path.join(os.path.dirname(__file__), "moa_prices_sabith(2025_jun_aug_26)_Shariatpur_bazar.csv"),
    "wait_seconds": 15,
    "delay_between_requests": 2,
    "skip_weekdays": [4, 5], 
}

def last_valid_day(skip_weekdays):
    d = date.today() - timedelta(days=1)
    while d.weekday() in skip_weekdays:
        d -= timedelta(days=1)
    return d

# CONFIG["end_date"] = last_valid_day(CONFIG["skip_weekdays"])
# CONFIG["start_date"] = CONFIG["end_date"] - timedelta(days=7)

CONFIG["start_date"] = date(2025,6,15
                            ) 
CONFIG["end_date"] = date(2026,8,21)

CSV_COLUMNS = ["date", "division", "district", "upazila", "market_name",
               "commodity_group", "commodity_sub_group", "commodity_name",
               "year", "retail_unit", "retail_price_min", "retail_price_max",
               "average_price", "wholesale_unit", "wholesale_price_min",
               "wholesale_price_max"]

TABLE_HEADER_ORDER = ["division", "district", "upazila", "market_name",
                       "commodity_name", "year", "retail_unit",
                       "retail_price", "average_price", "wholesale_unit",
                       "wholesale_price"]


# ============================================================
# Browser setup
# ============================================================

def build_driver(headless):
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--window-size=1500,1100")
    options.add_argument("--disable-gpu")
    options.add_argument("--lang=en-US")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def force_english_ui(driver, wait):
    toggle_buttons = driver.find_elements(By.CSS_SELECTOR, "button.btn_lang")
    if len(toggle_buttons) == 0:
        return
    button_text = toggle_buttons[0].text.strip().lower()
    if "english" in button_text:
        toggle_buttons[0].click()
        time.sleep(2)


# ============================================================
# Custom checkbox-dropdown helpers
# ============================================================

def safe_click(driver, element, retry_count=4):
    for _ in range(retry_count):
        try:
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
            time.sleep(0.05)
            driver.execute_script("arguments[0].click();", element)
            return True
        except StaleElementReferenceException:
            time.sleep(0.2)
    raise StaleElementReferenceException("Element became stale while clicking.")


def select_dropdown_option(driver, field_id, option_text, wait):
    for attempt in range(5):
        try:
            container = wait.until(EC.presence_of_element_located((By.ID, field_id)))
            button = container.find_element(By.CSS_SELECTOR, "button.btn-select")
            safe_click(driver, button)
            time.sleep(0.5)

            try:
                WebDriverWait(driver, 5).until(
                    lambda d: len(d.find_elements(By.CSS_SELECTOR, f"#{field_id} li.selectItem")) > 0
                )
            except TimeoutException:
                pass

            items = container.find_elements(By.CSS_SELECTOR, "li.selectItem")
            target = option_text.strip().lower()
            found = False

            for item in list(items):
                try:
                    span = item.find_elements(By.TAG_NAME, "span")
                    if not span:
                        continue
                    text = span[0].text.strip().lower()
                    cb = item.find_elements(By.CSS_SELECTOR, "input[type='checkbox']")
                    is_checked = cb[0].is_selected() if cb else False

                    if text == target:
                        found = True
                        if not is_checked:
                            safe_click(driver, item)
                            time.sleep(0.1)
                    else:
                        if is_checked:
                            safe_click(driver, item)
                            time.sleep(0.1)
                except (StaleElementReferenceException, NoSuchElementException):
                    continue

            layer = container.find_elements(By.CSS_SELECTOR, ".checkboxLayer.open")
            if len(layer) > 0:
                safe_click(driver, button)
                time.sleep(0.2)

            if not found:
                raise ValueError(f"Option '{option_text}' not found in dropdown '{field_id}'")

            time.sleep(1.5)
            return

        except (StaleElementReferenceException, NoSuchElementException):
            time.sleep(0.5)
            continue

    raise ValueError(f"Could not reliably select option '{option_text}' in dropdown '{field_id}' after retries")


def select_specific_options(driver, field_id, wait, option_texts, level_name="", min_wait_seconds=8):
    for attempt in range(5):
        try:
            container = wait.until(EC.presence_of_element_located((By.ID, field_id)))
            button = container.find_element(By.CSS_SELECTOR, "button.btn-select")
            safe_click(driver, button)
            time.sleep(0.5)

            try:
                WebDriverWait(driver, min_wait_seconds).until(
                    lambda d: len(d.find_elements(By.CSS_SELECTOR, f"#{field_id} li.selectItem")) > 0
                )
            except TimeoutException:
                pass

            target_texts = [text.strip().lower() for text in option_texts]
            items = container.find_elements(By.CSS_SELECTOR, "li.selectItem")
            selected_count = 0
            found_targets = set()

            for item in list(items):
                try:
                    span = item.find_elements(By.TAG_NAME, "span")
                    if not span:
                        continue
                    text = span[0].text.strip().lower()
                    cb = item.find_elements(By.CSS_SELECTOR, "input[type='checkbox']")
                    is_checked = cb[0].is_selected() if cb else False

                    if text in target_texts:
                        found_targets.add(text)
                        if not is_checked:
                            safe_click(driver, item)
                            time.sleep(0.05)
                        selected_count += 1
                    else:
                        if is_checked:
                            safe_click(driver, item)
                            time.sleep(0.05)
                except (StaleElementReferenceException, NoSuchElementException):
                    continue

            missing = set(target_texts) - found_targets
            if missing:
                print(f"  WARNING: could not find these {level_name} options to select: {missing}")

            layer = container.find_elements(By.CSS_SELECTOR, ".checkboxLayer.open")
            if len(layer) > 0:
                safe_click(driver, button)
                time.sleep(0.2)

            time.sleep(1.5)

            if selected_count == 0:
                print(f"  WARNING: none of the requested {level_name} options were found/selected.")
                return False
            return True

        except (StaleElementReferenceException, NoSuchElementException):
            time.sleep(0.5)
            continue

    print(f"  WARNING: could not reliably select {level_name} after retries.")
    return False


def select_location(driver, ids, location, wait):
    steps = [
        ("division", location["division"]),
        ("district", location["district"]),
        ("upazila", location["upazila"]),
        ("market", location["market"]),
    ]
    for field_key, value in steps:
        try:
            select_dropdown_option(driver, ids[field_key], value, wait)
        except Exception as error:
            print(f"  WARNING: could not select {field_key} = '{value}' "
                  f"({error}) -- skipping this location.")
            return False
    return True


def select_specific_commodities(driver, ids, wait):
    group_ok = select_specific_options(driver, ids["commodity_group"], wait, COMMODITY_GROUPS, "commodity_group")
    if not group_ok: return False
    sub_group_ok = select_specific_options(driver, ids["commodity_sub_group"], wait, COMMODITY_SUB_GROUPS, "commodity_sub_group")
    if not sub_group_ok: return False
    name_ok = select_specific_options(driver, ids["commodity_name"], wait, COMMODITY_NAMES, "commodity_name")
    if not name_ok: return False
    return True


# ============================================================
# Date fields and Interactions
# ============================================================

def set_date_field(driver, css_selector, target_date):
    date_str = target_date.strftime("%Y-%m-%d")
    script = """
        var el = document.querySelector(arguments[0]);
        if (el) {
            if (el._flatpickr) {
                el._flatpickr.setDate(arguments[1], true);
                return true;
            } else {
                el.value = arguments[1];
                el.dispatchEvent(new Event('input', { bubbles: true }));
                el.dispatchEvent(new Event('change', { bubbles: true }));
                return true;
            }
        }
        return false;
    """
    success = driver.execute_script(script, css_selector, date_str)
    if not success:
        raise RuntimeError(f"Could not find or set date field: {css_selector}")


def select_daily(driver, wait):
    radio = wait.until(EC.presence_of_element_located(
        (By.CSS_SELECTOR, "input[type=radio][value=daily]")))
    driver.execute_script("arguments[0].click();", radio)


def click_search(driver, wait):
    for _ in range(5):
        try:
            button = wait.until(EC.element_to_be_clickable(
                (By.XPATH, "//button[@type='submit' and contains(@class,'btn-primary')]")))
            safe_click(driver, button)
            return
        except (StaleElementReferenceException, NoSuchElementException):
            time.sleep(0.3)
    raise TimeoutException("Could not click the search button after retrying.")


def split_range(text):
    text = text.strip()
    if text == "":
        return "", ""
    match = re.match(r"([\d,]+)\s*-\s*([\d,]+)", text)
    if match:
        return match.group(1).replace(",", ""), match.group(2).replace(",", "")
    single = text.replace(",", "")
    return single, single


def read_results_table(driver, wait):
    if "No Data Found" in driver.page_source:
        return []

    for _ in range(5):
        try:
            try:
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody tr")))
            except TimeoutException:
                return []

            rows_out = []
            body_rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")

            for row in list(body_rows):
                try:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) == 0:
                        continue

                    cell_texts = [cell.text.strip() for cell in cells]
                    raw = {}
                    for i, field_name in enumerate(TABLE_HEADER_ORDER):
                        raw[field_name] = cell_texts[i] if i < len(cell_texts) else ""

                    retail_min, retail_max = split_range(raw.get("retail_price", ""))
                    wholesale_min, wholesale_max = split_range(raw.get("wholesale_price", ""))

                    record = {
                        "division": raw.get("division", ""),
                        "district": raw.get("district", ""),
                        "upazila": raw.get("upazila", ""),
                        "market_name": raw.get("market_name", ""),
                        "commodity_name": raw.get("commodity_name", ""),
                        "year": raw.get("year", ""),
                        "retail_unit": raw.get("retail_unit", ""),
                        "retail_price_min": retail_min,
                        "retail_price_max": retail_max,
                        "average_price": raw.get("average_price", ""),
                        "wholesale_unit": raw.get("wholesale_unit", ""),
                        "wholesale_price_min": wholesale_min,
                        "wholesale_price_max": wholesale_max,
                    }
                    rows_out.append(record)
                except (StaleElementReferenceException, NoSuchElementException):
                    continue

            return rows_out
        except StaleElementReferenceException:
            time.sleep(0.3)

    return []


# ============================================================
# Progress tracking (resumable) + date loop
# ============================================================

def load_already_scraped_keys(output_csv):
    already_scraped = set()
    if not os.path.exists(output_csv):
        return already_scraped
    with open(output_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                row.get("division", ""),
                row.get("district", ""),
                row.get("upazila", ""),
                row.get("market_name", ""),
                row.get("date", ""),
                row.get("commodity_name", "").strip().lower(),
            )
            already_scraped.add(key)
    return already_scraped


def append_rows_to_csv(output_csv, rows):
    file_exists = os.path.exists(output_csv)
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def daterange(start_date, end_date, skip_weekdays=None):
    if skip_weekdays is None:
        skip_weekdays = []
    current = start_date
    while current <= end_date:
        if current.weekday() not in skip_weekdays:
            yield current
        current += timedelta(days=1)


# ============================================================
# Main scrape loop
# ============================================================

def main():
    config = CONFIG
    ids = config["field_ids"]

    dates_to_scrape = list(daterange(config["start_date"], config["end_date"], config["skip_weekdays"]))
    if len(dates_to_scrape) == 0:
        print("WARNING: date range resolved to zero dates.")

    driver = build_driver(config["headless"])
    wait = WebDriverWait(driver, config["wait_seconds"])

    already_scraped = load_already_scraped_keys(config["output_csv"])
    total_rows_saved = 0
    total_searches = 0

    print(f"Output file: {config['output_csv']}")
    print(f"Dates to scrape ({len(dates_to_scrape)}): {[d.strftime('%Y-%m-%d') for d in dates_to_scrape]}")

    for location_index, location in enumerate(LOCATIONS, start=1):
        print(f"\n[Location {location_index}/{len(LOCATIONS)}] "
              f"{location['division']} / {location['district']} / {location['upazila']} / {location['market']}")

        # FIX: Force a page reload for every new location to completely clear the UI state, 
        # destroying floating datepickers or locked cascading fields.
        driver.get(config["url"])
        force_english_ui(driver, wait)

        location_ok = select_location(driver, ids, location, wait)
        if not location_ok:
            continue

        commodities_ok = select_specific_commodities(driver, ids, wait)
        if not commodities_ok:
            print("  WARNING: could not select requested commodities -- skipping.")
            continue

        select_daily(driver, wait)

        for date_index, target_date in enumerate(dates_to_scrape, start=1):
            date_str = target_date.strftime("%Y-%m-%d")
            print(f"  [{date_index}/{len(dates_to_scrape)}] Searching {date_str} ...", end=" ")

            try:
                set_date_field(driver, "input.datepicker", target_date)
                set_date_field(driver, "input.datepicker1", target_date)
                click_search(driver, wait)
                total_searches += 1
                time.sleep(2) 

                rows = read_results_table(driver, wait)
                new_rows = []

                for row in rows:
                    row["date"] = date_str
                    row["commodity_group"] = ""
                    row["commodity_sub_group"] = ""
                    
                    row_key = (
                        row["division"], row["district"], row["upazila"], 
                        row["market_name"], date_str, row["commodity_name"].strip().lower()
                    )
                    if row_key not in already_scraped:
                        new_rows.append(row)
                        already_scraped.add(row_key)

                if len(new_rows) > 0:
                    append_rows_to_csv(config["output_csv"], new_rows)
                    total_rows_saved += len(new_rows)
                    print(f"saved {len(new_rows)} new row(s).")
                else:
                    if len(rows) > 0:
                        print("data found but already exists in CSV -- skipping.")
                    else:
                        print("no data found -- skipping.")

            except Exception as error:
                print(f"error ({error}) -- skipping.")

            time.sleep(config["delay_between_requests"])

    print(f"\nDone. Total searches run: {total_searches}. Total new rows saved: {total_rows_saved}")
    driver.quit()

if __name__ == "__main__":
    main()