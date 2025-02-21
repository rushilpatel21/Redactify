# PII Anonymization API

This project provides a Flask-based API that automatically detects and anonymizes sensitive information (PII) from input text. It uses a combination of detection methods including Presidio Analyzer, a Hugging Face NER pipeline, and custom regular expressions to identify entities like names, organizations, locations, emails, phone numbers, dates, URLs, etc.

## Overview

The API supports two redaction modes:
- **Full Redaction:** Every detected PII is replaced with a hash-based pseudonym (e.g., `[PERSON-320b8e]`).
- **Partial Redaction:** For certain types (such as emails and URLs), a custom partial mask is applied (e.g., preserving the first 2 and last 2 or 3 characters) while other tokens are partially masked using a general rule.

The API leverages three different methods for PII detection:
- **Presidio Analyzer:** Detects a wide range of standard PII.
- **Hugging Face NER Pipeline:** Uses a fine-tuned BERT model (dbmdz/bert-large-cased-finetuned-conll03-english) to extract entities.
- **Regex Patterns:** Additional patterns are applied to capture formats that may be missed by the above methods (e.g., various phone number formats, dates, etc.).

## Features

- **Multi-Method Detection:** Combines machine learning (via Hugging Face) with rule-based (regex) and Presidio-based detection.
- **Flexible Redaction:** Choose between full redaction (using hash-based pseudonyms) and partial redaction (custom masking for emails and URLs, general masking for others).
- **Configurable Options:** The detection confidence threshold, placeholders, and enabled PII types can be configured via environment variables and/or the request payload.
- **Performance Optimizations:** Detection methods run concurrently using a thread pool to improve processing speed.

## Installation and Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/rushilpatel21/Redactify.git
   cd Redactify
   ```

2. **Create a virtual environment and install dependencies:**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Run the API server:**

   ```bash
   python server.py
   ```

   By default, the server runs on port `8000`. You can override this by setting the `PORT` environment variable.

## Usage

Send a POST request to the `/anonymize` endpoint with a JSON payload containing:
- `"text"`: The input text to be anonymized.
- `"options"` (optional): A dictionary to enable/disable anonymization for specific PII types.
- `"full_redaction"` (optional, boolean):  
  - `true` for full redaction (hash-based pseudonyms)  
  - `false` for partial redaction (custom masking)

### Sample Payload (Partial Redaction)

```json
{
  "text": "This agreement, effective as of 01/05/2024 or 01-05-2024, is made between Generic & Associates (contact email: john.doe@example.com, phone: 555-123-4567) and the Client, Mr. John Smith (SSN: 123-45-6789, email: j.smith@example.net or info.other@example.net or my Example University mail xx123456@example.edu, phone: 987-654-3210 and 12345-67890). My roll number is 22BCE308. Visit https://www.linkedin.com/in/johnsmith and https://github.com/johnsmith for more info.",
  "options": {
      "PERSON": true,
      "ORGANIZATION": true,
      "LOCATION": true,
      "EMAIL_ADDRESS": true,
      "PHONE_NUMBER": true,
      "CREDIT_CARD": true,
      "SSN": true,
      "IP_ADDRESS": true,
      "URL": true,
      "DATE_TIME": true,
      "PASSWORD": true
  },
  "full_redaction": false
}

```

### Sample Output (Partial Redaction)

```json
{
    "anonymized_text": "This agreement, effective as of 01*****024 or 01*****024, is made between Ge***************tes (contact email: jo****oe@*******.com, phone: 55*******567) and the Client, Mr. Jo*****ith (SSN: 12******789, email: j.***th@*******.net or in******er@*******.net or my Ex*************ity mail xx****56@*******.edu, phone: 98*******210 and 12******890). My roll number is 22***308. Visit https://***.li***din.***/in/jo****ith and https://gi*hub.***/jo****ith for more info."
}
```

### Sample Payload (Full Redaction)

```json
{
  "text": "This agreement, effective as of 01/05/2024 or 01-05-2024, is made between Generic & Associates (contact email: john.doe@example.com, phone: 555-123-4567) and the Client, Mr. John Smith (SSN: 123-45-6789, email: j.smith@example.net or info.other@example.net or my Example University mail xx123456@example.edu, phone: 987-654-3210 and 12345-67890). My roll number is 22BCE308. Visit https://www.linkedin.com/in/johnsmith and https://github.com/johnsmith for more info.",
  "options": {
      "PERSON": true,
      "ORGANIZATION": true,
      "LOCATION": true,
      "EMAIL_ADDRESS": true,
      "PHONE_NUMBER": true,
      "CREDIT_CARD": true,
      "SSN": true,
      "IP_ADDRESS": true,
      "URL": true,
      "DATE_TIME": true,
      "PASSWORD": true
  },
  "full_redaction": true
}
```

### Sample Output (Full Redaction)

```json
{
    "anonymized_text": "This agreement, effective as of [DATE_TIME-cad1e6] or [DATE_TIME-0c0a3a], is made between [ORGANIZATION-0458a5] (contact email: [EMAIL_ADDRESS-8eb1b5], phone: [PHONE_NUMBER-ca71de]) and the Client, Mr. [PERSON-611732] (SSN: [SSN-1e8748], email: [EMAIL_ADDRESS-75fb49] or [EMAIL_ADDRESS-8cb50b] or my [ORGANIZATION-a75ee3] mail [EMAIL_ADDRESS-2c1b67], phone: [UK_NHS-607d40] and [PHONE_NUMBER-d32fe4]). My roll number is [ROLL_NUMBER-9c5d7c]. Visit [URL-b1cc0b] and [URL-b01233] for more info."
}
```


## Configuration

- **Confidence Threshold:**  
  Set via the `CONFIDENCE_THRESHOLD` environment variable (default is `0.6`).

- **Port:**  
  The server runs on port `8000` by default, which can be changed by setting the `PORT` environment variable.

- **PII Options:**  
  The API allows toggling anonymization for each PII type through the `"options"` field in the payload.

## Future Improvements

- **Custom Model Training:** Fine-tune a custom NER model to improve detection of domain-specific entities.
- **Advanced Contextual Recognition:** Incorporate context-aware recognizers to capture ambiguous identifiers.
- **Enhanced Error Handling:** Add more robust error handling and logging for production readiness.
- **Production Deployment:** Transition to a production-grade WSGI server (e.g., Gunicorn) and add security measures such as rate limiting.

---
