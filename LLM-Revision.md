# LLM Fundamentals — Revision Cheatsheet

> Quick revision notes for understanding, using, and building applications with Large Language Models (LLMs).

## 1. How an LLM Works

### What is an LLM?
- A **Large Language Model (LLM)** is a neural network trained on large amounts of text to predict the next token.
- Most modern LLMs use the **Transformer architecture**.
- Examples: GPT, Claude, Gemini, Llama, Qwen.

### Transformer Architecture
- **Encoder:** Converts input into contextual representations. Common in understanding-focused models (e.g., BERT).
- **Decoder:** Generates output token by token. Common in generative models (e.g., GPT).
- **Encoder–Decoder:** Encoder processes input; decoder generates output (e.g., T5).

**Key concept — Self-Attention:** Determines how much importance each token should give to other tokens when building its contextual representation.

### How LLMs Are Trained
1. **Pretraining:** Learns language patterns by predicting the next token across massive datasets.
2. **Supervised Fine-Tuning (SFT):** Learns to follow instructions using curated input-output examples.
3. **Alignment / Preference Tuning:** Improves responses based on human preferences, using methods such as RLHF or DPO.

### Hosted vs Open-Weight Models
- **Hosted models:** Provider manages model infrastructure; access through APIs.
- **Open-weight models:** Model weights are available to download and run or customize.
- Open-weight does not automatically mean open-source training data, code, or unrestricted licensing.

---

## 2. How an LLM Generates Text

### Tokens and Tokenization
- A **token** is a unit of text: a complete word, part of a word, punctuation, or another text segment.
- Models process token IDs, not raw words or characters.
- A tokenizer converts text into token IDs; decoding converts IDs back into text.
- Tokenization differs across models and languages.

**Why tokens matter:**
- API cost is calculated using tokens.
- Context windows have token limits.
- Output generation speed depends partly on the number of generated tokens.

```python
import tiktoken

encoding = tiktoken.get_encoding("o200k_base")

text = "How do Large Language Models work?"

tokens = encoding.encode(text)
print(tokens)
print(encoding.decode(tokens))
print("Token count:", len(tokens))
```

### Next-Token Prediction
- LLMs predict the probability of every possible next token based on the existing context.
- The model produces **logits** (raw scores).
- Softmax converts logits into probabilities summing to 1.
- A decoding strategy selects the next token.

**Remember:**

`Input → Logits → Softmax → Probabilities → Token Selection`

### Text Generation
Generation is an autoregressive process:

1. Read input tokens.
2. Predict the next token.
3. Select a token.
4. Append it to the sequence.
5. Repeat until a stopping condition is reached.

**Autoregressive:** Each generated token becomes part of the context used to predict the next token.

### Hallucination
- An LLM can generate convincing but factually incorrect information.
- It predicts plausible continuations; it does not automatically verify facts.
- Common causes: missing information, ambiguous prompts, outdated knowledge, and unsupported assumptions.

**Ways to reduce hallucinations:**
- Provide reliable context.
- Use RAG to retrieve relevant information.
- Instruct the model to acknowledge uncertainty.
- Validate outputs against trusted sources.
- Evaluate responses against known test cases.

---

## 3. Controlling Model Output

### Temperature
Controls the randomness of token selection by scaling logits before applying softmax.

- **Low temperature:** More predictable, focused output.
- **High temperature:** More varied and creative output.
- **Temperature = 0:** Often behaves deterministically, but exact behavior depends on the provider and implementation.

**Use cases:**
- Classification, extraction → Low temperature.
- Brainstorming, creative writing → Higher temperature.

### Top-K Sampling
- Selects only the K most probable next tokens.
- Removes all candidates outside the top K before sampling.
- Smaller K → More restricted choices.
- Larger K → More variety.

### Top-P Sampling (Nucleus Sampling)
- Selects the smallest set of tokens whose cumulative probability reaches P.
- Samples from this dynamic set.
- Lower P → More focused output.
- Higher P → More diverse output.

**Top-K vs Top-P:**
- Top-K: Fixed number of candidate tokens.
- Top-P: Variable number of candidate tokens based on cumulative probability.

### Max Tokens
- Limits the maximum number of generated tokens.
- Prevents unnecessarily long responses and controls output cost.
- Reaching the limit may truncate an otherwise incomplete answer.

### Finish Reason
Indicates why generation stopped.

Common reasons:
- `stop`: A normal stopping condition was reached.
- `length`: Maximum token limit was reached.
- `tool_calls`: Model requested a tool invocation (where supported).

Always inspect the finish reason when processing API responses.

### Seeds
- A seed initializes the random number generator used in sampling.
- Same seed and settings can improve reproducibility.
- Identical results are not guaranteed across providers, model versions, hardware, or infrastructure.

---

## 4. Prompt Engineering

### Chat Messages and Roles

LLM APIs commonly accept a list of messages.

| Role | Purpose |
|---|---|
| System | Defines overall behavior and constraints |
| Developer | Provides application-level instructions |
| User | Contains the user's request |
| Assistant | Contains model responses |
| Tool | Contains tool execution results |

*Exact role support and precedence depend on the API.*

### Chat Templates
- Convert structured messages into the format expected by a model.
- Different model families may use different chat templates.
- Role markers and special tokens help distinguish instructions, user messages, and responses.

**Important:** Most API calls are stateless. If conversation history is needed, the application must send the relevant previous messages again or use a provider-supported state mechanism.

### System Prompt
Defines how the model should behave throughout a task.

A good system prompt specifies:

1. **Role:** Who should the model act as?
2. **Task:** What should it accomplish?
3. **Context:** What information should it use?
4. **Constraints:** What should it avoid?
5. **Output format:** How should the answer be returned?
6. **Fallback:** What should happen when information is missing?

### Few-Shot Prompting
- Provides examples of expected inputs and outputs inside the prompt.
- Helps the model recognize the desired pattern, structure, and classification behavior.
- Useful when instructions alone are insufficient.

**Zero-shot:** No examples.

**One-shot:** One example.

**Few-shot:** Multiple examples.

### Structured Output
- Forces or guides the model to return information in a predefined structure, commonly JSON.
- Useful for integrating LLMs with APIs and backend applications.
- Schema validation ensures the output meets application requirements.

Example:

```python
from pydantic import BaseModel
from typing import Literal

class Ticket(BaseModel):
    category: Literal["billing", "shipping", "other"]
    priority: Literal["low", "medium", "high"]
```

**Remember:** Valid JSON does not necessarily mean correct information. Use schema validation and semantic/business-rule validation.

### Prompt Evaluation
- Never assume a prompt works because one response looks correct.
- Create a representative test dataset with expected outputs.
- Run the same test cases against different prompt versions.
- Measure accuracy, consistency, format validity, and failure cases.

**Example:** For a ticket classifier, calculate:

`Accuracy = Correct Predictions / Total Predictions`

For production, also consider precision, recall, F1-score, and error costs.

---

## 5. Production Considerations

### Context Window
- Maximum number of tokens a model can process in a request.
- Includes input instructions, conversation history, retrieved documents, and generated output (subject to model/API-specific accounting).
- Exceeding the limit can cause request failures or require context reduction.

**Ways to manage context:**
- Remove irrelevant conversation history.
- Summarize older messages.
- Retrieve only relevant documents.
- Limit unnecessary prompt content.
- Reserve sufficient space for the output.

### LLM Cost
API costs generally depend on:

- Input tokens.
- Output tokens.
- Model-specific pricing.
- Additional charges for tools, caching, or other features where applicable.

**Cost estimation:**

`Total Cost = (Input Tokens / 1M × Input Price) + (Output Tokens / 1M × Output Price)`

**Cost optimization:**
- Use smaller models for simpler tasks.
- Reduce unnecessary prompt tokens.
- Limit output length.
- Cache reusable responses or context where appropriate.
- Use prompt caching when supported.
- Avoid sending unnecessary conversation history.

### LLM Latency
Latency measures how long an API request takes.

Important metrics:

- **Time to First Token (TTFT):** Time before the first generated token arrives.
- **Tokens Per Second (TPS):** Generation speed.
- **Total Response Time:** Time taken to complete the request.
- **End-to-End Latency:** Includes application overhead, network time, retrieval, tools, and model execution.

**Ways to improve latency:**
- Use smaller or faster models.
- Reduce input and output tokens.
- Stream responses.
- Optimize retrieval and external API calls.
- Use parallel execution where appropriate.

**Streaming vs Non-Streaming:**
- Non-streaming: Wait for the complete response.
- Streaming: Receive output chunks as they are generated.
- Streaming improves perceived responsiveness, not necessarily total generation time.

### Model Selection
Consider the following before selecting a model:

- Quality and task performance.
- Context window.
- Cost.
- Latency.
- Structured output and tool-calling support.
- Privacy and deployment requirements.
- Availability and licensing.

**Best model ≠ Largest model.** Choose based on actual task performance and production constraints.

---

## 6. Additional Concepts for Applied AI Engineers

These concepts extend the fundamentals into real-world LLM application development.

### Embeddings
- Convert text into numerical vectors representing semantic information.
- Similar meanings can have nearby vector representations.
- Used in semantic search, recommendation systems, and RAG.

**Important:** Embeddings are not the same as token IDs. Token IDs identify vocabulary entries; embeddings represent learned numerical features.

### RAG (Retrieval-Augmented Generation)
Combines information retrieval with LLM generation.

**Workflow:**

`Documents → Chunking → Embeddings → Vector Database`

`User Query → Query Embedding → Similarity Search → Retrieved Context → LLM → Answer`

- Reduces reliance on the model's internal knowledge.
- Provides access to external and updated information.
- Can improve factual grounding.
- Does not guarantee correctness; retrieval quality and answer grounding must be evaluated.

### Fine-Tuning vs RAG

| Fine-Tuning | RAG |
|---|---|
| Updates model behavior through training | Supplies external context at inference time |
| Useful for specialized behavior, style, or task patterns | Useful for private, changing, or domain-specific knowledge |
| Requires training data and compute | Requires retrieval infrastructure |
| Knowledge updates generally require another training process | Documents can be updated independently |

**Rule of thumb:** Use RAG to provide knowledge; use fine-tuning to adapt behavior.

### Function Calling / Tool Calling
- Allows an LLM to request execution of external functions.
- The application executes the requested tool and returns its result to the model.
- Useful for database queries, APIs, calculations, and external actions.

**Workflow:**

`User → LLM → Tool Call → Application Executes Tool → Tool Result → LLM → Response`

The model requests the action; the application controls execution and permissions.

### LLM Evaluation
Evaluate at multiple levels:

- **Model evaluation:** General capability and benchmark performance.
- **Prompt evaluation:** Whether instructions produce the expected behavior.
- **RAG evaluation:** Retrieval relevance and answer grounding.
- **Application evaluation:** End-to-end task success, cost, latency, and reliability.

Always maintain a representative evaluation dataset and compare changes against a baseline.

### LLM Security
Important risks when deploying LLM applications:

- **Prompt Injection:** Malicious instructions attempt to override intended behavior.
- **Sensitive Data Leakage:** Private information appears in model inputs or outputs.
- **Insecure Tool Execution:** Model-generated actions are executed without proper validation.
- **Untrusted Output:** Generated text is treated as safe code, SQL, HTML, or commands.

**Best practices:** Validate inputs and outputs, enforce authorization outside the model, restrict tool permissions, protect secrets, and treat retrieved documents as untrusted input.

---

## 7. Quick Revision — Key Differences

| Concept | Remember |
|---|---|
| Token vs Word | A token can be part of a word |
| Logits vs Probabilities | Raw scores vs normalized likelihoods |
| Temperature vs Top-K | Probability distribution randomness vs fixed candidate count |
| Top-K vs Top-P | Fixed number of candidates vs cumulative probability threshold |
| Context Window vs Max Tokens | Total request context capacity vs generation limit |
| Streaming vs Non-Streaming | Incremental output vs complete response |
| Zero-shot vs Few-shot | No examples vs multiple examples |
| System Prompt vs User Prompt | Behavior instructions vs task request |
| Embeddings vs Token IDs | Semantic vectors vs vocabulary identifiers |
| RAG vs Fine-Tuning | External knowledge retrieval vs model adaptation |
| Hosted vs Open-Weight | Provider-managed access vs downloadable model weights |


**Core takeaway:** An LLM predicts the next token. Prompt engineering guides its behavior, sampling controls its output, and production engineering makes the resulting application reliable, cost-effective, and useful.
