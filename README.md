# E-Presso BI Analyst (starter)

> Replace this README as you build. Your final version should cover every section below.

## What this application does
_One paragraph for a manager, not a programmer._

## Live app
_URL here. Note that the app may take a moment to wake up._

## How to use it
_Upload files, use filters, ask the assistant. Include the file names and sheets it expects._

## Data model
### Source tables and grain

| Table | Source file / sheet | One row = | Key column(s) | Row count | Contains revenue? |
|---|---|---|---|---|---|
| Sales | Sales.csv, exported from Support_EX_B_Sales.accdb | Revenue for one store and one product in one month (Date = last day of the month) | Sales ID; also unique on Store ID + Product ID + month | 25,200 | Yes (Revenue) |
| Categories | Categories.csv, exported from Support_EX_B_Sales.accdb | One product category | Category ID | 5 | No |
| Products | Products.csv, exported from Support_EX_B_Sales.accdb | One product | Product ID | 20 | No |
| Stores | Support_EX_B_Stores.csv (tab separated) | One store or sales channel; SH00 is the Website | ID | 21 | No |
| Survey Results | Support_EX_B_Survey.xlsx, sheet "Survey Results" | One store in one year | Survey ID (matches StoreYear in Sales) | 105 | No |

Row counts are for the original E-Presso data. The app recalculates them from whatever files are uploaded.

### How the Access tables were exported

The Sales, Categories, and Products tables were exported from the Access database on a Mac with mdbtools:

    mdb-export Support_EX_B_Sales.accdb Sales > Sales.csv
    mdb-export Support_EX_B_Sales.accdb Categories > Categories.csv
    mdb-export Support_EX_B_Sales.accdb Products > Products.csv

### Data quality findings (original data)

- Sales has one row for every store, product, and month: 21 stores × 20 products × 60 months = 25,200. These are monthly totals, not individual transactions.
- Feb 2028 rows are dated the 28th, although 2028 is a leap year. Monthly totals are unaffected.
- Products P11, P15, and P19 are all named "Base Model." The app groups by Product ID, not name.
- Products has no Category ID column. Category is assigned through Sales.Category ID; each product maps to exactly one category.
- SH00 (Website) has no city, state, or postal code, and no survey scores. This is structural: there is no store visit to survey.
- SH01 and SH08 are both in Orlando, FL but are labeled South and East.
- SH07 (Flushing, NY) has Location = "Southwest," which appears to be a mislabel.
- SH11's phone number is malformed.
## Metric definitions
_What "revenue" means here, and anything it does not mean._

## Assistant
_State exactly what you built (live language-model agent, or rules-based assistant). List each tool, what it calculates, and what the assistant is not allowed to do._

## Validation
_Link to or summarize your Power BI comparison, independent check, and $1,000 test._

## Limitations
_Revenue is not profit. What else can this app not tell a manager?_

## Files
| File | Purpose |
|---|---|
| `streamlit_app.py` | Screen layout and interaction |
| `analytics.py` | Metrics and analytical functions (no Streamlit code) |
| `agent.py` | Tool registry, dispatcher, and routers |
| `tests/` | Tests with hand-checkable answers |
| `requirements.txt` | Packages the hosted app needs |

## Run it
Hosted: deploy from this repository on share.streamlit.io with main file `streamlit_app.py`.

Locally (optional): `pip install -r requirements-dev.txt`, then `streamlit run streamlit_app.py`. Run tests with `pytest`.

## Secrets
Never commit API keys. Use Streamlit Secrets (see `.streamlit/secrets.toml.example`).
