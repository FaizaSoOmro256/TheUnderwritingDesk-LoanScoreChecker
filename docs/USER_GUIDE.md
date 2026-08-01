# User Guide — How to Use the Loan Score Checker

This guide explains every field, button, and result on the app screen in plain language. No technical background needed.

## Table of Contents

- [What This Tool Does](#what-this-tool-does)
- [The Two Tabs](#the-two-tabs)
- [Loan Details — Every Field Explained](#loan-details--every-field-explained)
- [The "What Matters More?" Buttons](#the-what-matters-more-buttons)
- [Reading the Result](#reading-the-result)
- [Upload Data Tab](#upload-data-tab)
- [The Model Stats Tab](#the-model-stats-tab)
- [Suggested Way to Use It](#suggested-way-to-use-it)

## What This Tool Does

You enter details about a loan applicant — their income, age, loan amount, and so on — and the tool instantly calculates:

1. A **credit score** from 300 (highest risk) to 850 (lowest risk), the same scale used by real credit scores.
2. A **recommendation**: Approve, Review, or Decline.
3. A **breakdown** of which details pushed that specific applicant's score up or down.

Every number changes live as you move a slider or change a dropdown — there is no "submit" button, because there is nothing to submit; it just recalculates as you go.

## The Two Tabs

At the top of the screen are two buttons:

| Button | What it opens |
|---|---|
| **Check a Loan** | The calculator — enter an applicant's details and see their score. This is the main screen. |
| **Upload Data** | Score a whole spreadsheet of applicants at once instead of one at a time. |
| **Model Stats** | A results page showing how well the underlying model performs overall, and which details matter most across all applicants (not just the one you're currently looking at). |

## Loan Details — Every Field Explained

These are the inputs on the left side of the "Check a Loan" screen. Sliders can be dragged left/right; dropdowns are clicked to pick an option; the toggle switch flips on/off.

| Field | What it means | Why it matters |
|---|---|---|
| **Yearly income** | The applicant's total income per year, before tax. | Higher income generally means more ability to repay a loan. |
| **Age** | The applicant's age in years. | Used by the model as one factor among many — on its own it has a small effect. |
| **Years at current job** | How long they've held their current job. | Longer job tenure usually signals more stable income. |
| **Loan grade** | A risk grade (A through G) already assigned to the loan by the lender. **A is the safest, G is the riskiest** — like a school grade, but in reverse. | This is one of the strongest signals the model uses. |
| **Loan amount** | How much money the applicant wants to borrow. | Larger loans relative to income carry more risk. |
| **Interest rate** | The yearly interest rate charged on the loan. | Riskier loans are usually priced with a higher interest rate. |
| **Years of credit history** | How long the applicant has had any form of credit (credit card, prior loan, etc.). | A longer history gives more evidence of how reliably they repay debt. |
| **Housing** | Whether the applicant rents, owns their home, has a mortgage, or another situation. | Owning outright is generally seen as more financially stable than renting. |
| **Reason for loan** | What the money is being used for (paying off debt, school, home repairs, medical bills, personal use, or business). | Some purposes are statistically riskier than others — for example, business loans tend to carry more risk than debt consolidation. |
| **Missed a loan payment before?** | A yes/no switch — has this applicant ever defaulted on a loan before? | A past default is one of the strongest warning signs the model looks at. |

## The "What Matters More?" Buttons

Above the score dial, there are three buttons that change how strict the tool is when deciding Approve vs. Decline:

| Button | What it does | When to use it |
|---|---|---|
| **Standard** | Uses a plain 50/50 cutoff point with no adjustment. | A neutral baseline, mainly useful for comparison. |
| **Catch more risk** | Shifts the cutoff so the tool flags more risky loans — catching about 85 out of 100 people who would actually default, instead of the usual 79. The tradeoff: it will also wrongly flag more good customers as risky. | Use this if missing a defaulter is much more costly than turning away a good customer. |
| **Save more money** | Shifts the cutoff to whichever point saves the most money overall, calculated from real loan amounts and interest rates rather than just counting right/wrong guesses. | This is the recommended setting for realistic, real-world decisions. |

Switching between these three buttons can change the Approve/Review/Decline result for the same applicant, because it changes how cautious the tool is being — this is expected and is the whole point of having the option.

## Reading the Result

The right-hand panel shows the outcome for whatever applicant details are currently entered:

- **The dial** — a visual gauge from 300 to 850. The needle position shows where this applicant's score falls on that scale.
- **The big number** — the exact credit score.
- **The stamped verdict** — the tool's recommendation:
  - **APPROVE** (green) — low risk, safe to lend to.
  - **REVIEW** (amber) — borderline case, worth a second look by a person.
  - **DECLINE** (red) — high risk of default.
- **The risk line below it** — states the risk level in words (Low / Moderate / High risk) and the estimated percent chance this applicant does not repay the loan.
- **"Why this score"** — a list of the specific details that mattered most for this particular applicant, ranked from biggest effect to smallest:
  - A **red upward arrow** means that detail pushed the score toward higher risk.
  - A **green downward arrow** means that detail pushed the score toward lower risk.
  - The number next to each arrow shows how large that detail's effect was — a bigger number means it mattered more.

## Upload Data Tab

Instead of checking one applicant at a time, this tab scores an entire file at once.

1. Click the upload box and choose a `.csv` file. It needs these columns: `person_age`, `person_income`, `person_home_ownership`, `person_emp_length`, `loan_intent`, `loan_grade`, `loan_amnt`, `loan_int_rate`, `cb_person_default_on_file`, `cb_person_cred_hist_length` — the same layout as the original training dataset.
2. The tool scores every row using whichever "What Matters More?" setting is currently selected on the Check a Loan tab.
3. You'll see:
   - Summary tiles: how many were approved, sent to review, declined, and the average score.
   - If your file also has a `loan_status` column (0 = repaid, 1 = defaulted), an extra box shows how accurate the predictions were against those real, known outcomes.
   - A scrollable table of every row's score, risk level, and verdict.
4. Click **Download Results CSV** to get your original file back with the predictions added as new columns.
5. Click **Upload Another File** to start over.

A small tag tells you whether the file was scored by the real server-side model or, if the server isn't running, by the same browser-based approximation used on the Check a Loan tab.

A sample file with 25 real applicants is included in the project at `sample_data/sample_applicants.csv` if you want to try this tab immediately.

## The Model Stats Tab

This tab doesn't use the applicant details you've entered — it shows fixed results from testing the model on thousands of real, historical loans.

- **"Which Model Works Best"** (bar chart) — compares five different versions of the model that were tried during development. Each bar shows a score from 0 to 1, where closer to 1 means the model is better at telling risky and safe loans apart. The gold bar is the best-performing version — the one actually powering this tool.
- **"What Affects the Score Most"** (bar chart) — ranks which applicant details matter most *on average, across all applicants*. This is different from the "Why this score" list on the Check a Loan tab, which is personalized to one specific applicant.

## Suggested Way to Use It

1. Start on **Check a Loan** with the **Save more money** button selected — this gives the most realistic real-world recommendation.
2. Adjust the sliders to test different scenarios (for example, see how the score changes if income goes up or the loan amount goes down).
3. Read the **"Why this score"** list to understand what's driving the result for that applicant.
4. Visit **Model Stats** to see how trustworthy the underlying model is overall, and which details tend to matter most in general.
