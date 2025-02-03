# CreditRisk-GenAI
A DEMO for modeling Credit Risk with GenAI Enabled

Key Features to Consider for Credit Risk Modeling:
Here's a list of important features in the LendingClub data that are typically useful for credit risk modeling. Many of these features relate directly to a borrower’s ability to repay a loan or their historical payment behavior, which are key indicators of credit risk.

1. Credit Score (fico_range_high, fico_range_low)
Why Important: Credit scores (e.g., FICO score) are one of the strongest indicators of creditworthiness. A higher credit score indicates lower risk of default.
Recommendation: Use the highest FICO score (fico_range_high) as a feature. It's a strong predictor of whether a borrower is likely to repay the loan.
2. Loan Amount (loan_amnt)
Why Important: The amount of money borrowed can be indicative of the borrower's financial situation. Larger loans may pose a higher risk, especially if the borrower's income or creditworthiness is lower.
Recommendation: Larger loan amounts typically correlate with higher risk, especially if they are disproportionate to the borrower's income.
3. Annual Income (annual_inc)
Why Important: The borrower's income is a crucial determinant of their ability to repay the loan. Lower income is often associated with a higher likelihood of default.
Recommendation: This can be used to assess the borrower’s debt-to-income ratio (along with loan amount), which is often a good indicator of credit risk.
4. Debt-to-Income Ratio (DTI) (dti)
Why Important: The debt-to-income (DTI) ratio shows how much of the borrower’s income is already tied up in debt repayments. A higher DTI indicates higher financial stress and a greater likelihood of default.
Recommendation: This is a critical feature as it captures the balance between debt obligations and income.
5. Loan Purpose (purpose)
Why Important: The reason for taking the loan can affect the likelihood of default. For example, loans for debt consolidation or refinancing might be less risky compared to loans for credit card refinancing or home improvements.
Recommendation: This feature can be categorical but useful for distinguishing the intent behind the loan and its potential risk.
6. Home Ownership (home_ownership)
Why Important: Whether the borrower owns their home (or rents) is important. Homeownership can be a sign of financial stability, and renters might have less financial stability.
Recommendation: Homeowners are typically less risky because they have more financial assets (e.g., home equity) compared to renters.
7. Employment Length (emp_length)
Why Important: Employment stability is an important indicator of the borrower’s ability to repay a loan. A longer employment history might correlate with a more stable income.
Recommendation: More years of employment typically indicate a stable financial situation and lower default risk.
8. Interest Rate (int_rate)
Why Important: The interest rate on the loan is a proxy for the perceived risk by the lender. Higher interest rates are often charged to higher-risk borrowers, and this can be predictive of loan default.
Recommendation: This can be used as a feature to measure the risk level assessed by the lender.
9. Credit History Length (credit_history)
Why Important: A longer credit history can indicate a more reliable borrowing pattern and lower credit risk. Borrowers with a longer history might have more reliable repayment behaviors.
Recommendation: The length of a borrower’s credit history is a strong predictor of their default probability.
10. Months since Last Delinquent (mths_since_last_delinq)
Why Important: A recent history of delinquency suggests that the borrower has financial trouble, making them more likely to default again.
Recommendation: Use this to gauge how recently the borrower experienced financial difficulties.
11. Total Credit Lines (tot_coll_amt, tot_cur_bal, etc.)
Why Important: The number of existing credit lines or the amount of current credit balance can help determine the borrower’s total debt load and their capacity to manage additional debt.
Recommendation: The more credit lines and higher current balances a borrower has, the greater the potential for them to default on additional loans.
12. Delinquency Flag (delinq_2yrs)
Why Important: Indicates whether the borrower has been delinquent on any credit obligations in the past 2 years. A high number of delinquencies is a strong indicator of a higher risk of future default.
Recommendation: This is a direct predictor of past repayment behavior and a key feature for credit risk.
13. Number of Credit Inquiries (inq_last_6mths)
Why Important: A large number of credit inquiries within the last 6 months can signal financial distress or increased risk-taking behavior, both of which are correlated with higher default risk.
Recommendation: This feature provides insight into how actively the borrower is seeking new credit, which can correlate with future default risk.
14. Collections Status (collection_recovery_fee, collections_12_mths_ex_med)
Why Important: These features capture whether the borrower has had collections, which significantly impacts credit risk.
Recommendation: Borrowers who have active collections on their records are typically at higher risk of defaulting.
15. State (addr_state)
Why Important: The state in which the borrower resides might correlate with economic conditions, local laws, and other factors that influence repayment behavior.
Recommendation: Although location might not directly impact an individual’s financial situation, it could serve as a proxy for socio-economic conditions that affect risk.
16. Public Record Information (pub_rec)
Why Important: This refers to any public records like bankruptcies, judgments, or liens. A higher number of public records typically indicates financial trouble.
Recommendation: Use this as a strong indicator of financial instability.
