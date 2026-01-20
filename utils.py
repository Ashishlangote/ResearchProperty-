from pymongo import MongoClient
import os

def price_to_number(price_str: str) -> float:
    """
    Converts '₹1.25 Cr' or '₹72 L' to numeric value in Lakhs
    """
    if not price_str:
        return 0

    price_str = price_str.replace("₹", "").strip()

    if "Cr" in price_str:
        return float(price_str.replace("Cr", "").strip()) * 100
    if "L" in price_str:
        return float(price_str.replace("L", "").strip())

    return 0


def get_mongo_collection():
    client = MongoClient(os.getenv("MONGODB_URI"))
    db = client["my_db"]
    return db["demo_col"]
