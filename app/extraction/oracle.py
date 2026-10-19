"""Ground-truth 'extractor' for evaluation only. Feeding the true field values to the later stages
isolates how good duplicate/anomaly/risk logic is from how good extraction is."""
from app.extraction.regex_extractor import Field, extract_fields


def truth_fields(truth: dict) -> dict:
    keys = ["vendor_name", "invoice_number", "invoice_date", "due_date", "currency", "total_amount", "po_number"]
    return {k: (Field(truth[k], 0.95, "oracle") if truth.get(k) is not None else None) for k in keys}


def oracle_extractor(email: dict) -> dict:
    """For ExtractStage(extractor=oracle_extractor); the replay puts the row's truth in email['_truth']."""
    return truth_fields(email["_truth"])


def fields_for(row: dict, oracle: bool) -> dict:
    return truth_fields(row["truth"]) if oracle else extract_fields(row["subject"], row["body"])
