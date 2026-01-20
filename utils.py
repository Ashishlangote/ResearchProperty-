from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

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

client = MongoClient(os.getenv("MONGODB_URI"))
db = client["my_db"]
collection = db["demo_col"]

def flatten_dict(data, parent_key="", sep=": "):
    items = []
    for key, value in data.items():
        key_name = key.replace("_", " ").title()
        new_key = f"{parent_key}{key_name}"

        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key + " - ").items())

        elif isinstance(value, list):
            if all(isinstance(i, dict) for i in value):
                for idx, item in enumerate(value, 1):
                    items.extend(
                        flatten_dict(item, f"{new_key} {idx} - ").items()
                    )
            else:
                items.append((new_key, ", ".join(map(str, value))))

        else:
            items.append((new_key, str(value)))

    return dict(items)


def mongo_documents_to_string():
    result = []
    cursor = collection.find({})
    for doc in cursor:
        flattened = flatten_dict(doc)
        doc_string = "\n".join(f"{k}: {v}" for k, v in flattened.items())
        result.append(doc_string)

    return "\n\n" + ("=" * 80) + "\n\n".join(result)
