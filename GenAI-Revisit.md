# Generative AI & Agentic AI — Revision Cheatsheet

> Practical revision notes for building LLM applications, RAG pipelines, AI agents, and production-ready Generative AI systems.

---

# 1. Generative AI Foundations

## Generative AI
- AI systems that generate new content such as text, images, audio, video, and code.
- LLMs are generative models trained to predict and generate sequences of tokens.
- Examples: GPT, Claude, Gemini, Llama.

## Generative AI vs AI Agents vs Agentic AI

| Concept | Meaning |
|---|---|
| Generative AI | Generates content based on input |
| AI Agent | Uses an LLM, tools, and memory to perform tasks |
| Agentic AI | Systems that autonomously plan, execute, evaluate, and adapt across multiple steps |

**Example:**
- Generative AI: Write an email.
- AI Agent: Search emails and draft a response.
- Agentic AI: Understand a goal, research information, execute tasks, verify results, and continue until completion.

## How LLMs Are Built

Typical lifecycle:

1. Data collection and preprocessing.
2. Tokenization.
3. Pretraining using next-token prediction.
4. Supervised fine-tuning (SFT).
5. Alignment using human preferences or other methods.
6. Evaluation and deployment.

**Remember:** Pretraining builds general language capabilities; fine-tuning adapts the model to specific behaviors and tasks.

---

# 2. Building Applications with LLMs

## LangChain
A framework for building applications powered by LLMs.

Common capabilities:
- Model integrations.
- Prompt templates.
- Structured outputs.
- Tool integration.
- Retrieval pipelines.
- Conversation management.

**Why use it?** Provides reusable abstractions instead of manually managing every model interaction and integration.

## Basic LLM Application

`User Input → Prompt → LLM → Response`

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("provider:model-name")

response = model.invoke("Explain machine learning")
print(response.content)
```

## Messages and Prompts

Messages represent structured interactions with a model.

| Message | Purpose |
|---|---|
| System | Defines model behavior |
| User | Contains user instructions or questions |
| Assistant | Model-generated response |
| Tool | Contains tool execution results |

**Prompt Template:** A reusable prompt containing dynamic variables.

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI tutor."),
    ("human", "Explain {topic} simply.")
])

messages = prompt.invoke({"topic": "RAG"})
```

## Streaming vs Batch

**Streaming**
- Returns output incrementally as tokens are generated.
- Improves perceived responsiveness.
- Useful for chat interfaces and interactive applications.

**Batch**
- Processes multiple independent inputs together.
- Useful for large-scale document processing, classification, and evaluation.
- Can improve throughput and reduce orchestration overhead.

## Structured Output

Converts model responses into predictable, machine-readable formats.

Common formats:
- JSON.
- Pydantic models.
- Typed dictionaries.

```python
from pydantic import BaseModel
from typing import Literal

class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    summary: str

structured_model = model.with_structured_output(Review)

result = structured_model.invoke(
    "The product is excellent and easy to use."
)
```

**Why important?** Backend applications need predictable data structures rather than unstructured text.

## Chatbot Memory

LLMs do not automatically retain conversation history across independent API calls.

Two common types:

- **Short-term memory:** Maintains the current conversation context.
- **Long-term memory:** Stores information across sessions for future retrieval.

Common approaches:
- Message history.
- Conversation summaries.
- Database-backed memory.
- Vector-based memory retrieval.

**Important:** Memory management includes deciding what to store, retrieve, summarize, and discard. Sending the entire conversation indefinitely is not scalable.

---

# 3. Retrieval-Augmented Generation (RAG)

## What is RAG?

RAG combines information retrieval with LLM generation.

Instead of relying only on knowledge learned during training, the application retrieves relevant external information and supplies it to the model as context.

**Why RAG?**
- Access private company documents.
- Use updated information.
- Reduce unsupported answers.
- Provide source references.
- Avoid retraining a model whenever documents change.

## RAG Architecture

RAG consists of two pipelines.

### A. Indexing Pipeline

Runs before users ask questions.

`Documents → Loading → Parsing → Chunking → Embeddings → Vector Database`

1. **Document Loading:** Read PDFs, Word files, websites, databases, etc.
2. **Parsing:** Extract usable text and metadata.
3. **Chunking:** Split documents into smaller pieces.
4. **Embeddings:** Convert chunks into numerical vectors.
5. **Vector Store:** Store embeddings and associated document information.

### B. Retrieval Pipeline

Runs when a user asks a question.

`Query → Embedding → Similarity Search → Retrieved Chunks → Prompt → LLM → Answer`

1. Convert the query into an embedding.
2. Retrieve relevant chunks.
3. Add retrieved context to the prompt.
4. Generate an answer grounded in the context.

**Remember:**

- Indexing prepares the knowledge base.
- Retrieval finds relevant information at query time.
- RAG does not eliminate hallucinations. Poor retrieval can still produce incorrect answers.

## Document Chunking

Splits large documents into smaller sections before embedding.

**Why?**
- Fits information into the model's context window.
- Improves retrieval granularity.
- Reduces irrelevant context.

Common strategies:

| Strategy | Description |
|---|---|
| Fixed-size | Splits text using a fixed number of characters or tokens |
| Recursive | Splits using hierarchical separators such as paragraphs and sentences |
| Semantic | Splits based on changes in meaning |
| Structure-based | Uses document headings, sections, pages, or other boundaries |

**Chunk overlap:** Repeats some text between adjacent chunks to preserve context across boundaries.

Important parameters:
- Chunk size.
- Chunk overlap.
- Metadata.
- Document structure.

There is no universally optimal chunk size. Evaluate using your actual documents and queries.

## Embeddings

- Numerical vector representations of text.
- Capture semantic relationships.
- Used to find information based on meaning rather than exact wording.

**Example:**

"How do I reset my password?"

"Steps to recover account credentials."

These queries can have similar embeddings despite using different words.

## Vector Databases

Store and retrieve embeddings efficiently.

Common options:
- FAISS.
- Chroma.
- Pinecone.
- Weaviate.
- Qdrant.
- Milvus.

**Similarity Search:** Finds vectors closest to a query vector.

Common similarity measures:
- Cosine similarity.
- Dot product.
- Euclidean distance.

**Remember:** Similarity scores are not automatically confidence scores or probabilities of correctness.

## Improving Retrieval Quality

Basic vector similarity search is not always sufficient.

### 1. Semantic Search
- Retrieves documents based on meaning.
- Useful for natural-language queries and paraphrasing.

### 2. Keyword Search (BM25)
- Retrieves documents based on matching terms.
- Useful for exact names, identifiers, technical terms, and keywords.

### 3. Hybrid Search
- Combines keyword and vector search.
- Captures both lexical and semantic relevance.

**Reciprocal Rank Fusion (RRF):** Combines ranked results from multiple retrievers using their positions rather than directly adding incompatible retrieval scores.

### 4. Query Rewriting
- Reformulates the user's question into better search queries.
- Useful when the original query is ambiguous or poorly phrased.

### 5. Reranking
- Retrieves an initial set of candidate documents.
- A reranker scores their relevance against the original query.
- Returns the most relevant documents to the LLM.

**Retriever vs Reranker:**
- Retriever: Finds potentially relevant documents.
- Reranker: Improves the ordering of retrieved documents.

### 6. Metadata Filtering
Filters documents using attributes such as:
- Department.
- Document type.
- Date.
- Source.
- User permissions.

## Conversational RAG

Combines conversation history with document retrieval.

Problem: A follow-up question may depend on previous messages.

Example:
- User: What is the leave policy?
- User: How many days can be carried forward?

The second query needs to be interpreted using the first question's context.

**Workflow:**

`Conversation History → Query Rewriting → Retrieval → Context → LLM → Answer`

## Agentic RAG

Uses an agent to decide how to retrieve information.

Unlike basic RAG, where retrieval follows a predefined pipeline, agentic RAG can:
- Decide whether retrieval is necessary.
- Select different retrieval tools.
- Rewrite queries.
- Search multiple sources.
- Evaluate retrieved results.
- Retry retrieval when results are insufficient.

**Key difference:** Basic RAG follows a predefined retrieval workflow; Agentic RAG introduces model-driven decisions into retrieval.

## Vectorless RAG

Uses document retrieval without relying on conventional vector embeddings.

Possible approaches:
- Keyword-based retrieval.
- Full-text search.
- SQL queries.
- Metadata and structured-data filtering.
- Other lexical or symbolic retrieval methods.

**When useful?** Exact term matching, structured data, identifiers, and cases where embeddings are unnecessary or less effective.

## RAG Evaluation

Evaluate both retrieval quality and answer quality.

**Retrieval metrics:**
- Precision@K: How many retrieved documents are relevant?
- Recall@K: How many relevant documents were successfully retrieved?
- MRR: How highly is the first relevant result ranked?
- NDCG: How well are relevant results ranked, considering graded relevance?

**Generation metrics:**
- Faithfulness: Is the answer supported by retrieved context?
- Answer relevance: Does it address the question?
- Correctness: Does it match the expected answer?
- Context relevance: Is the retrieved information useful?

**Important:** A correct-looking answer does not prove retrieval worked correctly. Evaluate both stages independently.

---

# 4. AI Agents

## What is an AI Agent?

An AI agent is a system in which an LLM decides what action to take, uses tools when necessary, observes the results, and continues until a task is completed or a stopping condition is reached.

### Four Core Components

1. **Model:** Makes decisions and selects actions.
2. **Tools:** Perform external operations such as API calls, searches, and database queries.
3. **Memory:** Maintains relevant conversation and execution context.
4. **Loop:** Repeats the decision-action-observation process.

`Input → LLM → Tool Call → Tool Result → LLM → Response`

**Important:** The LLM requests a tool call. The application or agent runtime executes the actual function.

## Tool Calling

Allows an LLM to interact with external systems.

Examples:
- Search the web.
- Query a database.
- Execute calculations.
- Retrieve documents.
- Send emails.
- Call external APIs.

A tool typically includes:
- Name.
- Description.
- Input schema.
- Function implementation.

**Tool calling is not the same as executing code directly inside the LLM.** The application validates and executes the requested operation.

## ReAct Framework

ReAct = Reasoning + Acting.

The agent alternates between reasoning about the task and taking actions.

Typical cycle:

1. Thought: Decide what information is needed.
2. Action: Call an appropriate tool.
3. Observation: Receive the tool result.
4. Repeat: Decide the next step.
5. Final Answer: Return the result.

**Why useful?** Supports multi-step tasks where the next action depends on the previous result.

## Agent Memory

| Type | Purpose |
|---|---|
| Short-term | Maintains current task and conversation state |
| Long-term | Stores information across sessions |
| Episodic | Stores past interactions or experiences |
| Semantic | Stores reusable facts and knowledge |

Memory can be implemented using:
- Message history.
- Summaries.
- Databases.
- Vector stores.

## Human-in-the-Loop (HITL)

Introduces human approval or intervention into an agent workflow.

Useful for:
- Financial transactions.
- Sending external communications.
- Deleting or modifying records.
- Sensitive business decisions.
- Actions with significant consequences.

**Workflow:**

`Agent Decision → Approval Required → Human Review → Approve / Reject → Continue`

## LangGraph

A framework for building stateful, multi-step agent workflows.

Core concepts:

- **State:** Data shared across workflow steps.
- **Nodes:** Individual functions or operations.
- **Edges:** Connections between nodes.
- **Conditional Edges:** Determine the next node based on state or decisions.
- **Cycles:** Allow workflows to repeat.
- **Checkpoints:** Persist workflow state for recovery and continuation.

**Why LangGraph?**
- Explicit workflow control.
- Stateful execution.
- Conditional routing.
- Human approval.
- Retry and loop management.
- More control over complex agent behavior.

## Multi-Agent Systems

Multiple specialized agents collaborate to complete a task.

Example: Research system.

- Researcher Agent: Collects information.
- Analyst Agent: Analyzes findings.
- Writer Agent: Creates a report.
- Reviewer Agent: Checks quality.

Common coordination patterns:
- Sequential: Agents execute one after another.
- Parallel: Agents work simultaneously.
- Hierarchical: A supervisor delegates tasks.
- Collaborative: Agents exchange information and coordinate.

**Trade-off:** More agents do not automatically mean better results. Coordination overhead, latency, cost, and failure propagation increase.

## Model Context Protocol (MCP)

An open protocol that standardizes how AI applications connect to external tools and context sources.

Core components:

- **MCP Host:** Application that provides the AI experience.
- **MCP Client:** Manages the connection to an MCP server.
- **MCP Server:** Exposes capabilities to the client.

Common server capabilities:

| Capability | Purpose |
|---|---|
| Tools | Actions the model can request |
| Resources | Context or data exposed to applications |
| Prompts | Reusable prompt templates |

**Why MCP?**
- Standardizes integrations.
- Reduces custom integration work.
- Allows compatible clients to connect to different external systems.

**MCP vs Function Calling:**
- Function calling: A model interaction capability for requesting functions.
- MCP: A protocol for exposing and connecting tools and context between applications and servers.

MCP does not replace tool calling; it standardizes how external capabilities can be made available.

## Deep Agents

Agents designed to handle complex, longer-running tasks that require planning, delegation, and managing substantial context.

Common capabilities:
- Task decomposition.
- Planning.
- Sub-agent delegation.
- Persistent state.
- File and context management.
- Iterative execution.

**Remember:** A basic agent may execute a tool call; a deep agent may plan and coordinate an entire multi-step project.

---

# 5. Loop Engineering

## What is Loop Engineering?

Designing an iterative process where an agent generates an output, verifies it, receives feedback, and improves it until it meets defined requirements.

**Basic Agent:**

`Input → Agent → Output`

**Loop-Based Agent:**

`Input → Generate → Verify → Feedback → Regenerate → Verify → Output`

## Why Loops Matter

A model's first response may:
- Contain factual errors.
- Miss requirements.
- Produce invalid code.
- Violate formatting constraints.
- Fail business rules.

A verification loop catches these problems before the result is delivered.

## Verifiers

A verifier evaluates whether an output meets predefined requirements.

Types:

**1. Rule-Based Verifier**
- Uses deterministic checks.
- Examples: Word count, JSON validity, required fields, numerical constraints.

**2. Model-Based Verifier**
- Uses another LLM to evaluate quality.
- Examples: Relevance, tone, completeness, factual consistency.

**3. Execution-Based Verifier**
- Runs the generated output.
- Examples: Execute code, run tests, validate SQL.

## Feedback

When verification fails:
1. Identify the failure.
2. Generate actionable feedback.
3. Send feedback to the agent.
4. Generate an improved output.
5. Verify again.

## Stop Conditions

Every agent loop needs explicit termination criteria.

Common conditions:
- Task completed successfully.
- Maximum retry count reached.
- Time or token budget exceeded.
- Human approval required.
- No further improvement is possible.

**Important:** Never create an unlimited retry loop. It can increase cost and latency without improving results.

## Anatomy of a Good Loop

- Clear task objective.
- Defined output requirements.
- Independent verification.
- Actionable feedback.
- Bounded retries.
- Explicit success and failure conditions.
- Logging of every iteration.

---

# 6. Harness Engineering

## What is Harness Engineering?

Designing the control and infrastructure layer around an AI agent to make it reliable, secure, observable, and production-ready.

**Key distinction:**

- Agent/LLM: Decides what to do.
- Harness: Controls how the system operates.

## Five Components of a Harness

### 1. Routing
Determines which agent, model, tool, or workflow should handle a request.

Example:
- Billing question → Billing Agent.
- Technical question → Technical Agent.
- General question → Support Agent.

Routing can use deterministic rules, classifiers, or LLM decisions.

### 2. State Management
Maintains information required throughout execution.

Examples:
- User input.
- Conversation history.
- Tool results.
- Agent outputs.
- Workflow status.

### 3. Guardrails
Define what the system is allowed to accept, generate, and execute.

Examples:
- Block harmful requests.
- Prevent sensitive data leakage.
- Restrict tool access.
- Validate inputs and outputs.

### 4. Validation
Checks whether outputs satisfy the expected requirements.

Examples:
- Validate JSON schemas.
- Check business rules.
- Execute generated code.
- Verify retrieved evidence.

### 5. Tracing
Records what happened during execution.

Tracks:
- Model inputs and outputs.
- Tool calls.
- Execution sequence.
- Errors.
- Latency.
- Token usage and cost.

Useful tools: LangSmith, Langfuse, OpenTelemetry.

**Remember:** A harness makes an agent operationally reliable. A good model alone is not a production system.

## Deterministic vs LLM-Based Decisions

**Deterministic:**
- Fixed business rules.
- Permission checks.
- Input validation.
- Financial calculations.
- Routing based on known conditions.

**LLM-Based:**
- Understanding ambiguous requests.
- Summarization.
- Semantic classification.
- Flexible planning.

**Best practice:** Use deterministic logic wherever possible; use LLMs where language understanding or flexible reasoning adds value.

## Guardrails

Common guardrail layers:

- Input validation.
- Prompt injection protection.
- Output validation.
- PII detection and redaction.
- Tool authorization.
- Rate limiting.
- Human approval.
- Audit logging.

**Important:** Prompt instructions alone are not security controls. Enforce critical permissions and restrictions in application code.

---

# 7. Quantization and Fine-Tuning

## Quantization

Reduces the numerical precision used to represent model weights.

Example:

| Format | Bits | Memory per Parameter |
|---|---:|---:|
| FP32 | 32 | 4 bytes |
| FP16 | 16 | 2 bytes |
| BF16 | 16 | 2 bytes |
| INT8 | 8 | 1 byte |
| INT4 | 4 | 0.5 bytes |

For a 7-billion-parameter model, approximate weight storage:

- FP32: 28 GB.
- FP16/BF16: 14 GB.
- INT8: 7 GB.
- INT4: 3.5 GB.

*These estimates exclude activations, KV cache, metadata, and other runtime memory.*

### Why Quantization?
- Reduces model memory requirements.
- Allows larger models to fit on available hardware.
- Can improve inference speed.
- Reduces hardware costs.

**Trade-off:** Lower numerical precision can reduce model quality.

## Scale and Zero-Point

Used in common affine quantization schemes to map floating-point values to integers.

- **Scale:** Determines the size of the interval represented by each quantized step.
- **Zero-point:** Determines which integer value represents real zero.

Conceptually:

`Quantized Value ≈ Round(Original Value / Scale) + Zero Point`

Dequantization approximately reverses the mapping.

## 1-bit LLMs — BitNet b1.58

An approach to extremely low-bit model representations.

- Uses ternary weights: {-1, 0, +1}.
- The 1.58-bit label comes from the information required to represent three possible values: log₂(3) ≈ 1.585 bits.
- Aims to reduce memory and computational requirements.

**Remember:** It is a specialized model architecture and quantization approach, not simply ordinary INT1 quantization.

## Fine-Tuning

Adapts a pretrained model using additional training data.

Useful for:
- Specialized task behavior.
- Domain-specific language patterns.
- Consistent response formats.
- Instruction following.
- Task-specific performance.

**Fine-tuning vs RAG:**

| Fine-Tuning | RAG |
|---|---|
| Updates model weights | Supplies external context |
| Changes learned behavior | Changes information available at inference |
| Requires a training process | Requires retrieval infrastructure |
| Updates generally require retraining | Documents can be re-indexed |

## LoRA (Low-Rank Adaptation)

A parameter-efficient fine-tuning technique.

Instead of updating all pretrained model weights:
- Freeze the original weights.
- Introduce small trainable low-rank matrices.
- Train only these additional parameters.

**Why LoRA?**
- Fewer trainable parameters.
- Lower memory requirements.
- Faster fine-tuning.
- Smaller adapter files.

## QLoRA

Combines quantization with LoRA.

Typical approach:
1. Load the pretrained model in 4-bit precision.
2. Freeze the quantized base model.
3. Add trainable LoRA adapters.
4. Train the adapters.
5. Save and reuse the adapters with the base model.

**LoRA vs QLoRA:**
- LoRA: Fine-tunes adapters while keeping the base weights frozen.
- QLoRA: Uses a quantized frozen base model to reduce memory requirements during adapter training.

## Custom Dataset Fine-Tuning

A typical workflow:

`Data Collection → Cleaning → Formatting → Tokenization → Training → Evaluation → Model Deployment`

Important considerations:
- High-quality training examples.
- Consistent input-output formats.
- Training and validation separation.
- Appropriate hyperparameters.
- Avoiding data leakage.
- Evaluating against the original base model.

**Remember:** More training data does not automatically mean better fine-tuning. Data quality and relevance matter.

---

# 8. Production Generative AI

## Evaluating LLM Applications

Evaluation measures whether an LLM application performs its intended task correctly.

Typical workflow:

1. Create a representative evaluation dataset.
2. Define expected outputs or evaluation criteria.
3. Run the application on the dataset.
4. Score the results.
5. Compare models, prompts, and configurations.

## LLM-as-a-Judge

Uses an LLM to evaluate another model's response.

Example:
- Model A generates an answer.
- Judge Model evaluates it against a reference answer.
- Judge returns a score or structured verdict.

Useful for evaluating:
- Correctness.
- Relevance.
- Completeness.
- Tone.
- Faithfulness.

**Limitation:** LLM judges can be biased, inconsistent, or incorrect. Calibrate them against human evaluations and deterministic checks.

## LLM Gateways

A centralized layer between applications and model providers.

Common responsibilities:
- Model routing.
- API key management.
- Rate limiting.
- Logging and monitoring.
- Usage tracking.
- Cost management.
- Retries and fallback models.

**Why useful?** Applications can interact with multiple model providers through a centralized interface rather than implementing each integration independently.

## Production Architecture

A simplified production LLM application:

`User → API → Validation → Orchestration → LLM / Tools / Retrieval → Output Validation → Response`

Supporting infrastructure:

- Authentication and authorization.
- Caching.
- Database and state management.
- Observability.
- Error handling.
- Rate limiting.
- Evaluation.
- Cost and latency monitoring.

## Production Metrics

| Metric | What it measures |
|---|---|
| Accuracy | Correctness of task outputs |
| Faithfulness | Whether answers are grounded in context |
| Latency | Response time |
| TTFT | Time to first token |
| Token Usage | Input and output consumption |
| Cost per Request | Average inference cost |
| Error Rate | Failed requests |
| Task Success Rate | Percentage of successfully completed tasks |

---

# 9. Quick Revision — Key Differences

| Concept | Key Difference |
|---|---|
| Generative AI vs Agentic AI | Generates content vs performs goal-directed actions |
| AI Agent vs Agentic Workflow | Individual decision-making system vs broader autonomous workflow |
| RAG vs Fine-Tuning | External knowledge vs model adaptation |
| Vector Search vs BM25 | Semantic matching vs keyword matching |
| Hybrid Search | Combines vector and keyword retrieval |
| Retriever vs Reranker | Finds candidates vs reorders candidates |
| Basic RAG vs Agentic RAG | Fixed retrieval vs model-driven retrieval decisions |
| Streaming vs Batch | Incremental output vs processing multiple inputs |
| Tool Calling vs MCP | Model requests functions vs standardized tool/context connectivity |
| Agent vs Multi-Agent | Single agent vs collaborating specialized agents |
| LangChain vs LangGraph | LLM application abstractions vs stateful graph-based workflows |
| Agent Loop vs Harness | Iterative improvement vs complete operational control layer |
| Guardrails vs Validation | Enforces allowed behavior vs checks output correctness |
| LoRA vs QLoRA | Adapter fine-tuning vs adapter fine-tuning with a quantized base |
| Quantization vs Fine-Tuning | Reduces numerical precision vs adapts model parameters |
| LLM vs LLM Gateway | Model capability vs centralized infrastructure layer |

---

## Final Takeaway

**Generative AI** teaches us to build applications that generate useful content.

**RAG** gives LLMs access to external knowledge.

**AI Agents** enable models to use tools and perform multi-step tasks.

**Loop Engineering** makes agents verify and improve their work.

**Harness Engineering** makes the entire system controlled, reliable, and observable.

**Fine-Tuning and Quantization** help adapt and optimize models.

**Production Engineering** brings everything together into scalable, secure, and cost-effective AI applications.
