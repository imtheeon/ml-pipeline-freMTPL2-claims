# Explain this project out loud

## 30-second pitch
"I predicted how often car-insurance policies will have a claim, using 678,000 real French policies. Insurers price on this. I compared a standard Poisson regression with gradient boosting and XGBoost. On policies the models never saw, XGBoost was about 6.5% better than charging everyone the average, and the riskiest 20% of policies claimed 2.3 times the average while the safest 20% claimed half. I also found where it fails: short policies."

## Numbers to know
- 677,991 policies; 3.7% have a claim; 0.074 claims per policy-year
- Random 60/20/20 split (the file has no dates); exam = 135,598 policies, 5,277 claims
- Exam: XGBoost 6.5% better than the average (5.9-7.0); regression 4.9%; riskiest 20% hold 38.9% of claims
- Riskiest 20% claim 2.3x average; safest 20% claim 0.5x

## Questions you will get

**Why Poisson, and what is deviance?**
Claims are counts, mostly zero, sometimes one or two. Poisson is the standard model for counts. Deviance is its error score; lower is better. I report it as "% better than charging everyone the average".

**What is exposure and why isn't it a feature?**
Exposure is the share of the year a policy was in force. A policy insured for 2 months has fewer chances to claim, so I multiply the rate by exposure. As a feature it would leak: many short policies are short because a claim ended them.

**Why not predict "claim: yes or no"?**
It throws away how long the policy ran and how many claims. Pricing needs a rate, not a label.

**Why a random split here, when you used a date split before?**
This file has no dates, so there's no "future" to hold out. I can't test a future year, and I say so.

**How did you handle the messy data?**
Impossible values (exposure above one year, 100-year-old cars, 16 claims on one policy) were capped, not deleted, so no policy is lost and odd rows can't steer the model. The caps were set before any modeling.

**Why XGBoost over gradient boosting?**
A rule I wrote before training: lowest validation deviance wins, simpler wins if within 0.0005. XGBoost won by 0.0007. On the exam the two are tied. I wouldn't claim one is better.

**What's the model's biggest weakness?**
Policy length. Claims on very short policies are twice the prediction, and near-full-year ones are over-predicted. Claim counts aren't proportional to time insured. First fix: an exposure-dependent adjustment.

**Is 6.5% better a big deal?**
For pricing, yes: small accuracy gains across a large book matter, and the risk spread (2.3x vs 0.5x) is what lets you charge fairly. But it's frequency only, not price.

**What would you do next?**
Model claim size too (frequency x severity), fix the exposure issue, use data with dates to test a future year.

**Who wrote the code?**
"I built it with Claude Code. I set the question, approved each stage, reviewed the results, and I can explain every decision."

## Weak spots to admit before they ask
- One portfolio, no dates
- XGBoost and gradient boosting are tied
- Short policies are mispredicted
- Frequency only
