#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    Suggested imports:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_deepseek import ChatDeepSeek

    Use the vision-capable DeepSeek Flash model named
    ``deepseek-v4-flash-vision-exp``. The API key is loaded from .env.
    """
    ### YOUR CODE HERE
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_deepseek import ChatDeepSeek

    # Thinking is on by default and can consume the whole completion before any
    # JSON is written. Receipt reading is extraction, so keep thinking off.
    model = ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        temperature=0,
        max_tokens=4096,
        timeout=180,
        max_retries=3,
        extra_body={"thinking": {"type": "disabled"}},
    )
    system = (
        "You read Hong Kong supermarket receipts. "
        "Reply with one JSON object and no other text."
    )
    rules = (
        "Read this receipt photo and return JSON with exactly these keys: "
        "item_amounts, discount_amounts, subtotal, rounding, amount_paid.\n"
        "item_amounts: every positive merchandise or bag-charge line total in "
        "the right-hand price column, above the subtotal. Use the line total, "
        "not the quantity and not an SP unit price.\n"
        "discount_amounts: absolute values of every discount, promotion, coupon, "
        "member-price, app-upgrade, or packaging-damage line above the subtotal. "
        "A wording like Buy 2 Save $5 and the -$5.00 beside it are ONE discount. "
        "Percentages and thresholds such as 5% OFF, OVER$40, or MB $200 are not "
        "amounts. Omit $0.00 coupon or VCODE lines.\n"
        "subtotal: the amount printed beside 小計 or SUBTOTAL. It is after "
        "discounts and before rounding.\n"
        "rounding: the signed ROUNDING amount, usually a few cents. Use 0 if "
        "that line is absent.\n"
        "amount_paid: the tender amount printed immediately after ROUNDING on "
        "OCTOPUS, VISA, CASH, EPS, or a similar payment line. Do not use Amount "
        "Deducted, 扣除金額, the card balance, 餘額, CHANGE, or 找續.\n"
        "These identities must hold: sum of item_amounts minus sum of "
        "discount_amounts equals subtotal, and subtotal plus rounding equals "
        "amount_paid."
    )
    image_block = {
        "type": "image_url",
        "image_url": {"url": "{image_url}", "detail": "original"},
    }
    extract_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", [{"type": "text", "text": rules}, image_block]),
        ]
    )
    repair_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                [
                    {
                        "type": "text",
                        "text": (
                            rules
                            + "\nThe previous reading failed the checks below. "
                            "Re-read the image and return a corrected JSON object.\n"
                            "{feedback}"
                        ),
                    },
                    image_block,
                ],
            ),
        ]
    )
    return {"extract": extract_prompt | model, "repair": repair_prompt | model}


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string.

    ``images`` contains every receipt in the selected folder. A valid return
    value looks like:

        {QUERY_1: "HK$123.40", QUERY_2: "HK$150.00"}

    Use the provided ``image_data_url(path)`` helper to put local images in
    multimodal human messages. LangChain's ``batch`` method is one simple way
    to process independent receipt-extraction prompts in parallel.
    """
    ### YOUR CODE HERE
    import json
    from decimal import Decimal

    def as_decimal(value: Any) -> Decimal:
        if isinstance(value, Decimal):
            return value
        text = str(value).strip().replace(",", "")
        for token in ("HK$", "HKD", "$"):
            text = text.replace(token, "")
        text = text.replace("−", "-").replace("–", "-").strip()
        return Decimal(text)

    def parse_payload(message: Any) -> dict[str, Any]:
        text = response_text(message)
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("model did not return a JSON object")
        data = json.loads(
            text[start : end + 1],
            parse_float=Decimal,
            parse_int=Decimal,
        )
        if not isinstance(data, dict):
            raise ValueError("JSON payload was not an object")
        return data

    def parts(
        data: dict[str, Any],
    ) -> tuple[Decimal, Decimal, Decimal, Decimal, Decimal]:
        items = [as_decimal(value) for value in data.get("item_amounts") or []]
        discounts = [abs(as_decimal(value)) for value in data.get("discount_amounts") or []]
        subtotal = as_decimal(data["subtotal"])
        rounding = as_decimal(data.get("rounding") or 0)
        paid = as_decimal(data["amount_paid"])
        item_total = sum(items, Decimal("0"))
        discount_total = sum(discounts, Decimal("0"))
        return item_total, discount_total, subtotal, rounding, paid

    def issues(data: dict[str, Any]) -> list[str]:
        item_total, discount_total, subtotal, rounding, paid = parts(data)
        found = []
        if abs((item_total - discount_total) - subtotal) > Decimal("0.02"):
            found.append(
                "item total "
                f"{item_total} minus discounts {discount_total} is "
                f"{item_total - discount_total}, but subtotal is {subtotal}."
            )
        if abs((subtotal + rounding) - paid) > Decimal("0.02"):
            found.append(
                f"subtotal {subtotal} plus rounding {rounding} is "
                f"{subtotal + rounding}, but amount_paid is {paid}."
            )
        return found

    def read_one(image: Path, raw: Any) -> dict[str, Any]:
        data: dict[str, Any] | None = None
        problem = "no reading yet"
        try:
            if isinstance(raw, Exception):
                raise raw
            data = parse_payload(raw)
            problem_list = issues(data)
            problem = "; ".join(problem_list)
        except Exception as exc:
            problem_list = [str(exc)]
            problem = str(exc)
        for _ in range(2):
            if data is not None and not problem_list:
                break
            feedback = problem
            if data is not None:
                feedback += "\nPrevious JSON:\n" + json.dumps(data, default=str)
            try:
                repaired = chain["repair"].invoke(
                    {"image_url": image_data_url(image), "feedback": feedback}
                )
                data = parse_payload(repaired)
                problem_list = issues(data)
                problem = "; ".join(problem_list)
            except Exception as exc:
                problem_list = [str(exc)]
                problem = str(exc)
        if data is None:
            raise RuntimeError(f"could not read {image.name}: {problem}")
        return data

    payloads = [{"image_url": image_data_url(path)} for path in images]
    raw_readings = chain["extract"].batch(
        payloads,
        config={"max_concurrency": 4},
        return_exceptions=True,
    )
    paid_total = Decimal("0")
    without_total = Decimal("0")
    for image, raw in zip(images, raw_readings):
        data = read_one(image, raw)
        _item_total, discount_total, subtotal, _rounding, paid = parts(data)
        paid_total += paid
        without_total += subtotal + discount_total

    def hkd(amount: Decimal) -> str:
        return f"HK${amount.quantize(Decimal('0.01'))}"

    return {QUERY_1: hkd(paid_total), QUERY_2: hkd(without_total)}


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
