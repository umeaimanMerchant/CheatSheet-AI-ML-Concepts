# ML System Design Cheat Sheet
 
A step-by-step workflow for answering "design an ML system for X". It assumes you already know the ML basics (classification, regression, ranking). It covers implementation decisions only.
 
**How to use it:** work through the steps in order. Each step has **Do** (actions), **Decide** (choices to state out loud), and **Pushed deeper** (where interviewers usually go next). Say the choice, the reason, and what would change your mind.
 
---
 
## Step 1: Clarify requirements
 
**Do**
- Ask what the system outputs and who or what uses the output.
- Ask for numbers, not adjectives.
**Decide**
- Latency: how fast must a response be (p95 / p99)?
- Freshness: how stale may predictions and features be?
- Availability: when do users ask, and what happens if the system is down?
- Scale: average and peak requests per second, number of items, number of users.
- Constraints: budget, team size, privacy, explainability, where it runs (cloud / device).
- Business goal and the current non-ML baseline.
**Pushed deeper**
- Do we need ML at all, or would rules do?
- Fail open or fail closed when the model is down?
---
 
## Step 2: Define the ML problem
 
**Do**
- Write one sentence: *given [inputs], predict [target], so that [action is taken]*.
- Pick the task type: classification, regression, ranking / recommendation, forecasting, anomaly detection, generation.
**Decide**
- Prediction granularity (per user, item, session, transaction).
- One model or multi-stage (a cheap stage narrows candidates, an expensive stage ranks them).
- What the label is, and whether it is available at training time.
- Which inputs are available at prediction time.
- Relative cost of each error type.
**Pushed deeper**
- Is the target a proxy for the real goal, and how far apart are they?
---
 
## Step 3: Gather data
 
**Do**
- List every source: where it lives, its schema, its refresh rate, its owner.
- Make sure the data used for training is collected and processed **the same way as in deployment**.
- Profile it: missing values, duplicates, outliers, class balance.
**Decide**
- Which data you need but do not have yet, and how to start logging it.
- Privacy and consent limits.
- Data validation checks (schema, ranges, null rates) as a gate before training.
**Pushed deeper**
- Training-serving skew: how do you guarantee the same processing in both?
- What if the data is small, noisy or biased?
---
 
## Step 4: Features
 
**Do**
- Group features: user, item, context, interaction.
- Define each feature once, and reuse that definition in training and serving.
**Decide**
- Batch features (precomputed) vs real-time features (computed per request).
- Encoding for categories, text and numbers.
- Handling of missing values and scaling.
| Feature type | Use when | Cost |
| --- | --- | --- |
| Batch | Slow-changing signals | Cheap, can be stale |
| Real-time | Signal changes within a session or minutes | Latency and infrastructure cost |
 
**Pushed deeper**
- Feature store: why share definitions between training and serving?
- Point-in-time correctness: no future data in a training row.
---
 
## Step 5: Build the training data
 
**Do**
- Define labels with an exact time window.
- Split **by time** whenever the system predicts the future.
- Check for leakage: any feature unavailable at prediction time.
**Decide**
- Label source: human, behavior, or delayed outcome.
- Imbalance handling: class weights, resampling, or threshold tuning.
- Sampling when data is too large.
| Label source | Pros | Cons |
| --- | --- | --- |
| Human annotation | Accurate | Slow, expensive |
| User behavior | Cheap, plentiful | Noisy, biased by what was shown |
| Delayed outcome | True signal | Arrives late, recent data under-labeled |
 
**Pushed deeper**
- Label delay and selection bias.
- Why a random split inflates scores on time-dependent data.
---
 
## Step 6: Model
 
**Do**
- Build a simple baseline first: a rule, average, last value or most popular.
- Train a simple learned model next, then a more complex one only if the gap justifies it.
- Review wrong predictions to find what to fix.
**Decide**
- Simplest model that clears the bar.
- Fit with the latency, memory and cost limits from Step 1.
- Explainability needs.
- Build, buy or fine-tune a pretrained model.
| Choice | Pick when |
| --- | --- |
| Simple / linear | Small data, need explainability, fast iteration |
| Tree ensembles | Tabular data, strong default |
| Deep models | Large data, text / image / sequence inputs |
| Pretrained + fine-tune | Little labeled data, a similar task already solved |
 
**Pushed deeper**
- Why this model over a simpler one?
- Training pipeline: automate ingest → validate → features → train → evaluate → register, and track code, data and parameters per run.
---
 
## Step 7: Offline evaluation
 
**Do**
- Pick one primary metric that matches the problem type, plus guardrail metrics.
- Evaluate on a time-based holdout, and on slices (segments, rare classes).
- Set a ship rule: improve X by Y without hurting Z.
| Problem type | Offline metrics | Online metrics |
| --- | --- | --- |
| Classification | Precision, recall, PR-AUC, F1 | Cost of errors, business loss |
| Regression / forecasting | MAE, RMSE, WAPE, bias | Business quantity the forecast drives |
| Ranking / recommendation | Recall@k, NDCG, MRR | Click-through, conversion, engagement |
| Anomaly detection | Precision at fixed recall | Cases caught, review load |
 
**Decide**
- Decision threshold from error costs, not a default 0.5.
- Guardrails: latency, fairness across segments, complaints.
**Pushed deeper**
- Offline gain but no online gain: proxy mismatch, leakage, skew, position bias.
- Why accuracy fails on imbalanced data.
---
 
## Step 8: Architecture
 
**Do**
- Pick the serving mode from Step 1 numbers.
- Draw the request path and add up the latency of each step.
| | Batch | Online | Stream |
| --- | --- | --- | --- |
| Use when | Nobody waits for the answer | A user or system waits | React to events as they arrive |
| Latency | Lookup, very fast | Tens to hundreds of ms | Milliseconds to seconds |
| Freshness | Hours to a day | Per request | Near real time |
| Cost / complexity | Lowest | Medium | Highest |
| Features from | Precomputed tables | Feature store + request | Streaming aggregates |
 
**Decide**
- Caching: what, and for how long.
- Hardware and autoscaling.
- Model optimization if needed: request batching, quantization, distillation.
- A fallback: default answer, last known good prediction, or a rule.
- Timeouts and retries.
**Pushed deeper**
- Multi-stage design: how many candidates each stage passes on.
- Back-of-envelope sizing: requests per second = daily requests ÷ 86,400; scale for peak; compute instances from per-machine capacity; storage = records × size × retention.
- Hybrid: precompute what you can, compute the rest online.
---
 
## Step 9: Deploy and roll out
 
**Do**
- Package the model and its dependencies so every environment is identical.
- Test: code, data checks, a quality gate against the current production model.
- Log inputs, outputs, model version and latency for every prediction.
| Strategy | What it does | Use for |
| --- | --- | --- |
| Shadow | Runs on live traffic, results hidden | Checking behavior with zero user risk |
| Canary | Small share of real traffic | Catching problems early |
| A/B test | Compares against the current system | Proving business impact |
| Full rollout | All traffic, old version kept for rollback | Final step |
 
**Decide**
- Rollback trigger: which metric, which threshold, who can pull it.
- Traffic split and test duration.
**Pushed deeper**
- How do you roll back in one step?
- How do you know the A/B test has enough traffic?
---
 
## Step 9b: Monitor
 
**Do**
- Watch four layers, and compare live data against training data.
| Layer | Watch |
| --- | --- |
| System | Latency, error rate, throughput, cost |
| Data | Schema breaks, null rates, feature distributions |
| Model | Prediction distribution, accuracy once labels arrive |
| Business | The Step 1 success metric |
 
**Decide**
- Alert thresholds and a named owner for each alert.
- A proxy signal when labels are delayed.
- A runbook: alert → check → action.
**Pushed deeper**
- Data drift (inputs change) vs concept drift (input-target relationship changes).
- Drift does not always mean worse performance, so check performance before acting.
---
 
## Step 10: Retrain
 
**Do**
- Choose a trigger, a training window, and a promotion rule.
- Compare every candidate against the **current production model**.
| Trigger | Pros | Cons |
| --- | --- | --- |
| Schedule | Simple, predictable | May retrain when unnecessary |
| New data volume | Tied to available signal | Ignores change in the world |
| Performance drop | Targeted | Needs fast, reliable labels |
| Drift alert | Early warning | Can be a false alarm |
 
**Decide**
- Window: all history, rolling, or recent data weighted higher.
- Warm start or train from scratch.
**Pushed deeper**
- Feedback loops: the model's outputs shape the data it later learns from.
- Exploration or holdout traffic to collect unbiased data.
---
 
## How to answer in the interview
 
**Order to speak in**
1. Clarify requirements and state numbers (latency, freshness, availability, scale).
2. Define the ML problem in one sentence.
3. Data and labels: sources, split by time, leakage.
4. Features: batch vs real-time, same code in training and serving.
5. Baseline, then model, with the reason for each.
6. Metrics: offline, online, guardrails, threshold.
7. Architecture: batch / online / stream, latency budget, fallback.
8. Rollout, monitoring, retraining.
**Habits**
- State assumptions out loud, then ask if they are right.
- For every choice, give: the choice, the reason, what would change it.
- Always reach monitoring and retraining; most candidates stop at the model.
- When pushed on a topic, answer in three sentences, then offer the next topic.
- If stuck, return to Step 1 and ask for a number.
**Last-hour checklist**
- [ ] I can state requirements as numbers for any system.
- [ ] I can name a baseline before a model.
- [ ] I can pick batch, online or stream and justify it.
- [ ] I can name the offline metric, online metric and guardrail for the problem type.
- [ ] I can explain training-serving skew, leakage and label delay.
- [ ] I can describe a rollout with a one-step rollback.
- [ ] I can name what I would monitor and what triggers a retrain.
- [ ] I can do a quick back-of-envelope size estimate.
 
