# E-Presso BI Analyst

An interactive business intelligence app for E-Presso, a fictional coffee equipment retailer. It loads the E-Presso sales, store, product, category, and survey files, checks and joins them, shows a filterable revenue dashboard, and includes an analytical assistant that answers business questions by running approved calculations on the loaded data.

## Live app

**URL:** https://espreapp-bi-analyst-g63qz98wfr9fayqb2zwsec.streamlit.app/

The app opens without a login. If it shows "This app has gone to sleep," click the button to wake it (about a minute).

## How to use it

1. **Upload the data.** In the sidebar, upload the five E-Presso files in any order: `Sales.csv`, `Categories.csv`, `Products.csv`, `Support_EX_B_Stores.csv`, and `Support_EX_B_Survey.xlsx`. Files are recognised by their column names, so renamed files still work. Uploading a changed file recalculates everything with no code change or restart.
2. **Dashboard tab.** Use the sidebar filters (year, sales channel, store, product category) to explore revenue. The scope line at the top states exactly what is included. Leave a filter blank to include everything.
3. **Assistant tab.** Ask a business question in the chat box. Each answer starts with its scope and is organised as Findings, Interpretation, Caveats, and Next steps. Expand the tool call log under an answer to see the function that ran, its arguments, and the raw result.
4. **Data, joins & quality tab.** See what was loaded, the grain of each table, the merge checks, and data quality findings.

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

### Table relationships and merge checks

| Merge | Join | Rows before | Rows after | Revenue sum before | Revenue sum after | Explanation |
|---|---|---|---|---|---|---|
| Sales + Stores | left, many-to-one on Store ID | 25,200 | 25,200 | $42,174,440 | $42,174,440 | Unchanged; every Store ID matched a store |
| Sales + Categories | left, many-to-one on Category ID | 25,200 | 25,200 | $42,174,440 | $42,174,440 | Unchanged; every Category ID matched |
| Sales + Products | left, many-to-one on Product ID | 25,200 | 25,200 | $42,174,440 | $42,174,440 | Unchanged; every Product ID matched |
| Store-year revenue + Survey | left, one-to-one on Store ID + Year | 105 | 105 | $42,174,440 | $42,174,440 | Survey joined after revenue was summed by store and year; 5 Website store-years have no score |

Numbers are for the original E-Presso data. The app recalculates this table from whatever files are uploaded. 

When I tested joining the survey to Sales on Store ID alone, the row count jumped from 25,200 to 126,000 and revenue from $42,174,440 to $210,872,200, five times the actual total. This happened because each store has one survey row per year for five years, so every sales row was matched five times. To fix this, the app sums revenue by store and year first, then joins the survey one to one.

## Metric definitions

Every number is calculated from the currently loaded data. Nothing is hardcoded.

| Metric | Definition |
|---|---|
| Total revenue | Sum of `Revenue` for all Sales rows in the current scope. One row is one store's revenue for one product in one month. |
| Website / brick and mortar revenue | Total revenue grouped by sales channel, taken from `StoreType` in the Stores table (joined on Store ID). |
| Website share | Website revenue divided by total revenue in the same scope. |
| Latest year growth | (Latest year revenue minus prior year revenue) divided by prior year revenue, using the same channel, store, and category filters. Shown as "no prior year" when the prior year is missing or zero. |
| Monthly share of year | A month's revenue divided by **that same year's** revenue under the same filters, so years of different size can be compared. |
| Category share | Category revenue divided by total revenue in the scope. Category comes from `Sales.Category ID` joined to Categories. |
| Store change | A store's revenue in the last year in scope minus its revenue in the first year, in dollars and as a percent of the first year. Left blank when the store had no revenue in the first year. |
| Top products | Revenue by product, grouped by Product ID because three different products are named "Base Model" (P11, P15, P19). |
| Satisfaction vs. revenue | Pearson correlation between each physical store's annual revenue and its survey Satisfaction score (0 to 5), one point per store and year. The Website has no survey and is excluded. This measures association only. |

## Analytical assistant

**What I built: Option A**, a live language model agent using function calling, with a rules based assistant (Option B) as an automatic fallback.

How a question is answered:

1. **Gemini** (`gemini-3.5-flash-lite`, free tier) reads the question and chooses one or more of the approved functions below, with parameters. Allowed stores, channels, and categories are limited to values in the loaded data.
2. **The app** runs the chosen function on the currently loaded data through a dispatcher that logs every call. The model never runs code and never sees the raw data.
3. The function results go back to Gemini, which writes the answer.
4. **Number check:** before an answer is shown, every dollar amount and percentage in it is compared with the function results. If any number is not found there, the app shows the rules based answer instead and says why.

**Approved functions** (all in `analytics.py`; each returns the tool contract: tool, value, definition, scope, rows used, caveats):

| Function | Answers questions like |
|---|---|
| `get_total_revenue` | How much did the website make in 2028? |
| `get_revenue_by_channel` | Website vs. stores in 2027? |
| `get_revenue_by_year` | How has revenue grown each year? |
| `get_store_trend` | Which store declined the most between two years? |
| `get_revenue_by_store` | Which store has the highest revenue? |
| `get_category_performance` | Which category sells best? |
| `get_top_products` | What are the top 5 products in 2029? |
| `get_monthly_revenue_share` | What share of 2026 revenue came in December? |
| `get_satisfaction_revenue_relationship` | Does satisfaction relate to revenue? |
| `get_data_quality_report` | Are there problems in the data? |

**Controls**

* Only the functions above can run. The dispatcher refuses any other name and drops arguments a function does not accept. There is no `eval`, `exec`, model written code or SQL, or DataFrame agent.
* Functions check every parameter against the loaded data. An unknown or ambiguous value (for example "Orlando", which matches two stores) returns an error result, not a guess.
* Gemini is instructed that revenue is not profit, that correlation does not establish cause in either direction, and that it must not recommend closing or opening stores from revenue alone.
* Questions the data cannot answer (profit, managers, prices, units, customer counts, individual transactions, forecasts) are identified, and the assistant says what it can answer instead.
* Each session is limited to 15 questions.
* **Fallback:** if there is no API key, the key is invalid, the quota is used up, Gemini is slow (30 second limit per request, 60 seconds per question), or the number check fails, the rules based router answers instead. The line under each answer says which router answered and which Sales file was used.
* A **Gemini connection check** panel in the Assistant tab tests each model and reports the time taken and any error.
* Sidebar filters with a single value can be used as defaults for questions; the scope line marks any value taken from the sidebar.

## Validation

Full records are in the submission. Summary:

| Metric | Power BI | App | Match? |
|---|---|---|---|
| Total revenue (five years) | $42,174,440 | $42,174,440 | Match |
| Website revenue | $27,218,880 | $27,218,880 | Match |
| Brick and mortar revenue | $14,955,560 | $14,955,560 | Match |
| Revenue by year | 2025 $7,181,750; 2026 $8,016,950; 2027 $8,824,720; 2028 $8,969,470; 2029 $9,181,550 | Same | Match, all five years |
| Monthly revenue pattern | 60 monthly values | Same | Match, 0 of 60 months differ |
| Revenue by category | Espresso Machines $27,093,220; Beans $5,672,720; Frothers $4,329,350; Syrups $2,712,040; Accessories $2,367,110 | Same | Match |

**Independent check:** I built a pivot table in Excel from the exported Sales.csv and calculated yearly revenue by summing the Revenue for each year of the Date column. My results matched my app and Power BI exactly for all five years and the $42,174,440 total. 

**$1,000 modified data test:** Sales ID S1120270312 (SH11 Chicago, March 2027, Espresso Machines) was changed from $2,420 to $3,420. Total revenue rose from $42,174,440 to exactly $42,175,440, every filter combination that includes the row rose by exactly $1,000, combinations that exclude it were unchanged, and no code change was needed. The original data was restored.

**Tests:** 77 automated tests (pytest) check the loader, joins, metrics, tool functions, router, and Gemini fallback, using small tables whose answers can be calculated by hand.

## Limitations

* **Revenue only.** There are no costs, so profit and margin cannot be calculated.
* **Monthly grain.** Each Sales row is one store, product, and month, so questions about individual transactions, order size, or units cannot be answered.
* **Survey coverage.** Satisfaction is one score per store per year, and the Website has none. The satisfaction analysis covers 100 physical store years and shows association only.
* **Source data quirks** (reported in the app, not corrected): February 2028 rows are dated the 28th although 2028 is a leap year; three products share the name "Base Model"; two Orlando stores have different regions; SH07 (Flushing, NY) is labelled "Southwest"; SH11's phone number is malformed.
* **Store change** compares only the first and last years in scope.
* **Assistant.** Each question is answered on its own, without memory of earlier questions. Gemini's free tier speed and availability vary (during testing the newest model timed out, so the app uses the fastest working model first). The rules based fallback understands only the wording it was built for.
* **Upload persistence.** Uploaded files last for the browser session; reload the page and upload them again.

## Files

| File | Purpose |
|---|---|
| `streamlit_app.py` | Screen layout only: upload, filters, dashboard, assistant chat, data tab |
| `data_prep.py` | Loading, file recognition, data types, grain statements, data quality checks, joins with merge checks |
| `analytics.py` | Business metrics and the assistant's tool functions (no Streamlit code) |
| `agent.py` | Tool registry, dispatcher, rules based router, Gemini router, number check, fallback |
| `tests/` | pytest tests with hand checkable answers |
| `requirements.txt` | Packages the hosted app needs |

## Running it locally

```bash
pip install -r requirements-dev.txt
streamlit run streamlit_app.py
pytest -q
```

## Secrets

The Gemini API key is stored in Streamlit Community Cloud under **Manage app → Settings → Secrets** as `GEMINI_API_KEY`. It is never stored in the code or the repository, and `.streamlit/secrets.toml` is listed in `.gitignore`. An optional `GEMINI_MODEL` secret sets the preferred model. Without a key, the app uses the rules based assistant.
