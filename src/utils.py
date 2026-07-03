CURRENCY_SYMBOLS = {
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "INR": "₹",
    "JPY": "¥",
    "CAD": "C$",
    "AUD": "A$",
    "CHF": "Fr",
    "CNY": "¥",
    "HKD": "HK$",
    "NZD": "NZ$",
    "SGD": "S$",
    "KRW": "₩",
    "MXN": "Mex$"
}


def get_currency_symbol(currency_code: str) -> str:
    """
    Return the currency symbol corresponding to a currency ISO code.
    Defaults to the code itself followed by a space if no symbol is found.
    """
    if not currency_code:
        return "$"
    code_upper = currency_code.upper().strip()
    return CURRENCY_SYMBOLS.get(code_upper, f"{code_upper} ")
