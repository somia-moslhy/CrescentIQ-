# Security Policy

## Supported Versions

Currently, only the latest release of CrescentIQ is actively supported with security updates.

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

Please **do not** report security vulnerabilities through public GitHub issues. 

If you discover a security vulnerability within CrescentIQ, please report it privately by sending an email to **[Your Email Address]**. 
You should receive a response within 48 hours. If the vulnerability is accepted, we will work on a patch and release it as quickly as possible.

## Scope of Security

As CrescentIQ is a Machine Learning and Demand Forecasting pipeline, our security considerations primarily focus on:

1. **Dependency Vulnerabilities:** Flaws in third-party data science libraries (e.g., `scikit-learn`, `xgboost`, `lightgbm`, `pandas`).
2. **Data Handling & Parsing:** Safe processing of raw CSV input files to prevent injection attacks or memory exhaustion.
3. **Model Serialization:** Protection against arbitrary code execution when loading serialized models.

## Best Practices & Warnings for Users

### ⚠️ Untrusted Model Files (Pickle/Joblib)
CrescentIQ exports and loads trained machine learning pipelines using `.pkl` formats (via `joblib` or `pickle`). 
**Never load a `.pkl` file from an untrusted source.** Loading a maliciously crafted pickle file can lead to Arbitrary Code Execution (ACE) on your machine. Only load models that you have trained locally or obtained from a completely trusted environment.

### Dependency Management
Users are strongly advised to keep their local environments updated. Run standard security audits on your environment:
```bash
pip-audit
# or
safety check
