# AgenticSeek Architecture Documentation

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [System Overview](#system-overview)
3. [Core Architecture](#core-architecture)
4. [Agent System Design](#agent-system-design)
5. [Planning & Task Execution](#planning--task-execution)
6. [Comparison with SOTA AI Agents](#comparison-with-sota-ai-agents)
7. [Future Architecture Improvements](#future-architecture-improvements)
8. [Planning System Deep Dive](#planning-system-deep-dive)
9. [SOC Cloud Integration Guide](#soc-cloud-integration-guide)

---

## Executive Summary

AgenticSeek is a **100% local alternative to Manus AI**, designed as a voice-enabled autonomous AI assistant that can browse the web, write code, and plan complex tasks while keeping all data on your device. Unlike cloud-based AI agents, AgenticSeek prioritizes **privacy, local execution, and zero cloud dependency**.

### Key Differentiators
- **Fully Local Execution**: All LLM inference, speech processing, and tool execution run on local hardware
- **Multi-Agent Architecture**: Specialized agents for different task domains (coding, browsing, file management, planning)
- **Intelligent Routing**: Automated agent selection based on task complexity and type
- **Privacy-First Design**: No data leaves your machine

---

## System Overview

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                  │
│    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐               │
│    │   Web UI     │     │     CLI      │     │  Voice I/O   │               │
│    │  (Frontend)  │     │   (cli.py)   │     │  (STT/TTS)   │               │
│    └──────┬───────┘     └──────┬───────┘     └──────┬───────┘               │
└───────────┼────────────────────┼────────────────────┼───────────────────────┘
            │                    │                    │
            └────────────────────┼────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           API LAYER (api.py)                                 │
│    ┌──────────────────────────────────────────────────────────────────┐     │
│    │  FastAPI Server + Celery Task Queue + Redis Backend              │     │
│    └──────────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     INTERACTION LAYER (interaction.py)                       │
│    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐               │
│    │   Session    │     │    Query     │     │   Response   │               │
│    │  Management  │     │  Processing  │     │   Handler    │               │
│    └──────────────┘     └──────────────┘     └──────────────┘               │
└─────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      AGENT ROUTER (router.py)                                │
│    ┌─────────────────────────────────────────────────────────────────┐      │
│    │  • Zero-Shot Classification (BART)                              │      │
│    │  • LLM Router (AdaptiveClassifier)                              │      │
│    │  • Complexity Estimation (HIGH/LOW)                             │      │
│    │  • Language Detection & Translation                             │      │
│    └─────────────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
                                 │
            ┌────────────────────┼────────────────────┐
            ▼                    ▼                    ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                        AGENT POOL                                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │   Casual    │ │   Coder     │ │   Browser   │ │    File     │           │
│  │   Agent     │ │   Agent     │ │   Agent     │ │   Agent     │           │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘           │
│                        ┌─────────────┐                                      │
│                        │  Planner    │  ← Complex Task Handler              │
│                        │   Agent     │                                      │
│                        └─────────────┘                                      │
└────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                        CORE SERVICES                                        │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │   Memory    │ │    LLM      │ │   Browser   │ │    Tools    │           │
│  │   System    │ │  Provider   │ │ Automation  │ │   Library   │           │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘           │
└────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                     EXTERNAL INTEGRATIONS                                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │   Ollama    │ │  LM Studio  │ │   SearxNG   │ │  MCP Tools  │           │
│  │   Server    │ │   Server    │ │   Search    │ │  (Optional) │           │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘           │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Architecture

### Component Breakdown

#### 1. User Interface Layer
| Component | File | Purpose |
|-----------|------|---------|
| Web Frontend | `frontend/` | React-based web interface |
| CLI Interface | `cli.py` | Command-line interaction |
| Voice I/O | `speech_to_text.py`, `text_to_speech.py` | Voice input/output processing |

#### 2. API Layer
```python
# api.py - FastAPI server with async processing
api = FastAPI(title="AgenticSeek API", version="0.1.0")
celery_app = Celery("tasks", broker="redis://localhost:6379/0")
```

**Key Endpoints:**
- `POST /query` - Submit user query for processing
- `GET /latest_answer` - Poll for agent responses
- `GET /screenshot` - Get browser screenshots
- `GET /health` - System health check
- `GET /stop` - Stop current agent execution

#### 3. Agent Router System
The `AgentRouter` is the intelligence layer that determines which agent handles each request:

```python
class AgentRouter:
    """
    Two-stage classification:
    1. Task Type Classification (BART + AdaptiveClassifier voting)
    2. Complexity Estimation (HIGH → PlannerAgent, LOW → Specialized Agent)
    """
```

**Classification Flow:**
```
User Query → Language Detection → Translation (if needed)
     ↓
Complexity Check → HIGH? → Planner Agent
     ↓ (LOW)
Task Classification → code/web/files/talk/mcp
     ↓
Select Specialized Agent
```

#### 4. Memory System
```python
class Memory:
    """
    - Conversation history management
    - Automatic memory compression using LED summarization model
    - Session persistence (save/load)
    - Context window optimization based on model size
    """
```

**Memory Features:**
- **Compression**: Uses `pszemraj/led-base-book-summary` for summarization
- **Context Estimation**: Dynamically calculates ideal context size based on model
- **Session Recovery**: Persists conversations across sessions

---

## Agent System Design

### Base Agent Architecture
All agents inherit from the abstract `Agent` class:

```python
class Agent(ABC):
    """
    Core Properties:
    - agent_name: Identifier for the agent
    - role: Classification label for routing
    - type: Agent category (casual_agent, code_agent, etc.)
    - memory: Conversation memory instance
    - tools: Dictionary of executable tools
    - llm: LLM provider instance
    """
```

### Specialized Agents

#### CasualAgent
- **Purpose**: General conversation and simple queries
- **Role**: "talk"
- **Tools**: None (pure conversation)

#### CoderAgent
- **Purpose**: Code generation, debugging, execution
- **Role**: "code"
- **Tools**: Code execution (Python, Bash, etc.)
- **Capabilities**: 
  - Multi-language code generation
  - Code execution in sandbox
  - Error interpretation and retry

#### BrowserAgent
- **Purpose**: Web browsing and automation
- **Role**: "web"
- **Tools**: Browser automation, web scraping
- **Capabilities**:
  - Selenium/undetected-chromedriver for stealth browsing
  - Screenshot capture
  - Form filling and interaction
  - SearxNG integration for search

#### FileAgent
- **Purpose**: File system operations
- **Role**: "files"
- **Tools**: File read/write/search
- **Capabilities**:
  - File discovery
  - Content manipulation
  - Directory operations

#### MCPAgent
- **Purpose**: Model Context Protocol integration
- **Role**: "mcp"
- **Tools**: External MCP services
- **Capabilities**:
  - Calendar integration
  - Email services
  - Custom API calls

#### PlannerAgent (Orchestrator)
- **Purpose**: Complex multi-step task planning and execution
- **Role**: "planification"
- **Type**: "planner_agent"
- **Sub-Agents**: Controls all other agents

```python
class PlannerAgent(Agent):
    """
    Orchestrates complex tasks through:
    1. Task decomposition into JSON plan
    2. Agent assignment for each sub-task
    3. Sequential execution with information passing
    4. Dynamic plan updates based on execution results
    """
    agents = {
        "coder": CoderAgent,
        "file": FileAgent,
        "web": BrowserAgent,
        "casual": CasualAgent
    }
```

### Agent Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    AGENT EXECUTION LOOP                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Receive Query                                                │
│       ↓                                                          │
│  2. Push to Memory                                               │
│       ↓                                                          │
│  3. LLM Request (async)                                          │
│       ↓                                                          │
│  4. Extract Answer + Reasoning                                   │
│       ↓                                                          │
│  5. Parse Tool Blocks (```code```, ```bash```, etc.)             │
│       ↓                                                          │
│  6. Execute Tools                                                │
│       ↓                                                          │
│  7. Gather Feedback                                              │
│       ↓                                                          │
│  8. Success? ─────Yes───→ Return Answer                          │
│       │                                                          │
│       No                                                         │
│       ↓                                                          │
│  9. Push Feedback to Memory → Loop to Step 3                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Planning & Task Execution

### Current Planning Architecture

The PlannerAgent uses a **sequential task decomposition** approach:

```json
{
  "plan": [
    {
      "id": "1",
      "agent": "web",
      "task": "Search for API documentation",
      "need": []
    },
    {
      "id": "2", 
      "agent": "coder",
      "task": "Implement API integration",
      "need": ["1"]
    },
    {
      "id": "3",
      "agent": "file",
      "task": "Save results to output.txt",
      "need": ["2"]
    }
  ]
}
```

### Planning Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     PLANNER AGENT FLOW                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Receive Complex Task (HIGH complexity)                       │
│       ↓                                                          │
│  2. make_plan() → LLM generates JSON plan                        │
│       ↓                                                          │
│  3. parse_agent_tasks() → Extract tasks                          │
│       ↓                                                          │
│  4. show_plan() → Display plan to user                           │
│       ↓                                                          │
│  ┌───────────────────────────────────────────┐                   │
│  │  FOR EACH TASK:                           │                   │
│  │    a. Gather required info from prev      │                   │
│  │    b. Create task-specific prompt         │                   │
│  │    c. start_agent_process()               │                   │
│  │    d. Execute specialized agent           │                   │
│  │    e. Collect results                     │                   │
│  │    f. update_plan() if failure            │                   │
│  └───────────────────────────────────────────┘                   │
│       ↓                                                          │
│  5. Return final answer                                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Dynamic Plan Updates

When a task fails, the planner can update the remaining plan:

```python
async def update_plan(self, goal, agents_tasks, agents_work_result, id, success):
    """
    - Evaluates task success/failure
    - Decides if plan modification needed
    - Can add recovery tasks
    - Maintains original tasks before current point
    """
```

---

## Comparison with SOTA AI Agents

### Manus AI Architecture (Reference)

Based on leaked system prompts and technical analysis, Manus AI uses:

| Component | Manus AI | AgenticSeek |
|-----------|----------|-------------|
| **Foundation Models** | Claude 3.5/3.7 + Qwen (fine-tuned) | Ollama/LM-Studio (local) |
| **Execution Environment** | Cloud VM sandbox | Local machine |
| **Action Mechanism** | CodeAct (Python code as actions) | Tool-specific execution blocks |
| **Multi-Agent Design** | Parallel specialized agents | Sequential agent pipeline |
| **Memory System** | File-based + Vector store (RAG) | Summarization-based compression |
| **Planning** | Iterative todo.md updates | JSON plan with dynamic updates |
| **Context Engineering** | KV-cache optimization, file system as context | Memory compression |

### Key Manus Design Principles (From Their Blog)

1. **KV-Cache Optimization**: Keep prompts stable, append-only context
2. **File System as Context**: Unlimited persistent memory via files
3. **Attention Manipulation**: Use todo.md to maintain focus
4. **Error Retention**: Keep failures in context for learning
5. **Mask, Don't Remove**: Constrain actions without modifying definitions

### Other SOTA Agent Architectures

#### OpenAI Operator (CUA Model)
- **Approach**: Vision + GUI interaction via reinforcement learning
- **Strength**: Direct pixel/DOM manipulation like humans
- **Result**: 87% on WebVoyager, 58% on WebArena

#### Google Agent Design Patterns
1. **Single-Agent**: One model with tools in a loop
2. **Multi-Agent Sequential**: Pipeline processing
3. **Multi-Agent Parallel**: Concurrent task execution
4. **ReAct Pattern**: Reason → Act → Observe loop
5. **Iterative Refinement**: Loop until quality threshold

#### LangGraph/CrewAI Frameworks
- Graph-based agent orchestration
- Role-based agent teams
- Flexible workflow patterns

---

## Future Architecture Improvements

### 1. Enhanced Multi-Agent Orchestration

**Current Limitation**: Sequential task execution only

**Proposed Improvement**: Parallel task execution for independent sub-tasks

```
┌─────────────────────────────────────────────────────────────────┐
│              PROPOSED: PARALLEL ORCHESTRATION                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    Planner Agent                         │    │
│  │         (Dependency Graph Construction)                  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│           ┌───────────────┼───────────────┐                     │
│           ▼               ▼               ▼                     │
│     ┌──────────┐    ┌──────────┐    ┌──────────┐               │
│     │ Task 1   │    │ Task 2   │    │ Task 3   │  ← Parallel   │
│     │ (Web)    │    │ (Code)   │    │ (File)   │               │
│     └────┬─────┘    └────┬─────┘    └────┬─────┘               │
│          │               │               │                      │
│          └───────────────┼───────────────┘                     │
│                          ▼                                      │
│                    ┌──────────┐                                 │
│                    │ Task 4   │  ← Depends on 1,2,3            │
│                    │ (Merge)  │                                 │
│                    └──────────┘                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Implementation Path**:
```python
# Proposed dependency-aware execution
class ParallelPlannerAgent(PlannerAgent):
    async def execute_parallel_tasks(self, tasks_by_level):
        """Execute independent tasks concurrently"""
        for level_tasks in tasks_by_level:
            await asyncio.gather(*[
                self.start_agent_process(task) 
                for task in level_tasks
            ])
```

### 2. Hierarchical Planning (HiPlan-style)

**Current Limitation**: Flat task list without hierarchical structure

**Proposed Improvement**: Two-tier planning with milestones and step-wise hints

```
┌─────────────────────────────────────────────────────────────────┐
│                 HIERARCHICAL PLANNING                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  GLOBAL LEVEL (Milestones):                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ M1: Research    →  M2: Implement  →  M3: Test & Deploy  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  LOCAL LEVEL (Step-wise hints):                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ M1.1: Search docs                                        │    │
│  │ M1.2: Analyze API                                        │    │
│  │ M1.3: Summarize findings                                 │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Tree-of-Thoughts Integration

**Current Limitation**: Linear reasoning path

**Proposed Improvement**: Exploration of multiple reasoning branches

```python
class ToTPlanner:
    """
    Tree of Thoughts planning:
    1. Generate multiple candidate plans
    2. Evaluate each plan branch
    3. Prune low-quality branches
    4. Expand promising branches
    5. Select optimal execution path
    """
    
    async def generate_candidates(self, goal, k=3):
        """Generate k different plan approaches"""
        pass
    
    async def evaluate_branch(self, plan):
        """Score plan feasibility"""
        pass
```

### 4. ReAct Pattern Enhancement

**Current Limitation**: Fixed action-observation loops

**Proposed Improvement**: Dynamic ReAct with reflection

```
┌─────────────────────────────────────────────────────────────────┐
│                 ENHANCED ReAct LOOP                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐                 │
│  │  THINK   │ ──→ │   ACT    │ ──→ │ OBSERVE  │                 │
│  └────┬─────┘     └──────────┘     └────┬─────┘                 │
│       │                                  │                       │
│       │         ┌──────────┐            │                       │
│       └─────────│ REFLECT  │←───────────┘                       │
│                 │ (Self-   │                                     │
│                 │ Critique)│                                     │
│                 └────┬─────┘                                     │
│                      │                                           │
│                      ▼                                           │
│               Update Strategy                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5. Manus-Inspired Improvements

| Feature | Implementation |
|---------|---------------|
| **File-as-Memory** | Save intermediate results to files, reference in prompts |
| **Todo.md Pattern** | Maintain running task list to focus attention |
| **Error Retention** | Keep failed attempts in context for learning |
| **KV-Cache Optimization** | Stable prompt prefixes, append-only context |
| **Context Distillation** | Summarize large observations before storage |

### 6. Agent Communication Protocol

**Proposed**: Structured inter-agent messaging

```python
class AgentMessage:
    sender: str
    receiver: str
    message_type: Literal["task", "result", "query", "error"]
    content: dict
    context: dict  # Shared state
    priority: int
```

---

## Planning System Deep Dive

### Current Planning Mechanisms

#### 1. Task Decomposition
```python
def parse_agent_tasks(self, text: str) -> List[Tuple[str, str]]:
    """
    Parses JSON plan from LLM output:
    - Extracts task names from headings
    - Parses JSON block with agent assignments
    - Validates agent availability
    - Maps task dependencies via 'need' field
    """
```

#### 2. Information Passing
```python
def make_prompt(self, task: str, agent_infos_dict: dict) -> str:
    """
    Creates contextualized prompts:
    - Includes results from previous agents
    - References by agent ID
    - Maintains information chain
    """
```

#### 3. Plan Update Strategy
```python
async def update_plan(self, goal, agents_tasks, agents_work_result, id, success):
    """
    Dynamic replanning:
    - Triggered on task failure
    - Preserves completed tasks
    - Modifies remaining tasks
    - Can add recovery steps
    """
```

### Recommended Planning Improvements

#### A. Goal-Oriented Planning
```python
class GoalDecomposer:
    """
    Hierarchical goal decomposition:
    1. Main Goal → Sub-goals
    2. Sub-goals → Actionable tasks
    3. Tasks → Agent assignments
    """
    
    def decompose(self, goal: str) -> GoalTree:
        # Use LLM for semantic decomposition
        # Identify dependencies
        # Assign complexity scores
        pass
```

#### B. Plan Validation
```python
class PlanValidator:
    """
    Pre-execution plan validation:
    1. Check agent availability for each task
    2. Validate dependency graph (no cycles)
    3. Estimate resource requirements
    4. Identify potential failure points
    """
```

#### C. Experience-Based Planning
```python
class PlanningMemory:
    """
    Learn from past executions:
    1. Store successful plan patterns
    2. Retrieve similar past plans
    3. Adapt based on historical success rates
    4. Build milestone library (HiPlan-style)
    """
```

#### D. Iterative Refinement Loop
```python
class RefinementLoop:
    """
    Multi-iteration plan improvement:
    1. Generate initial plan
    2. Critic agent evaluates plan
    3. Refine based on feedback
    4. Repeat until quality threshold
    """
    
    async def refine(self, plan, max_iterations=3):
        for i in range(max_iterations):
            evaluation = await self.critic.evaluate(plan)
            if evaluation.score >= self.threshold:
                break
            plan = await self.improve(plan, evaluation.feedback)
        return plan
```

---

## SOC Cloud Integration Guide

### Overview

This section provides guidance for Security Operations Center (SOC) employees to customize and optimize AgenticSeek for SOC Cloud use cases.

### Typical SOC Use Cases

| Use Case | Agent | Configuration |
|----------|-------|---------------|
| Log Analysis | Coder + File | Parse and analyze security logs |
| Threat Intel Gathering | Browser | Search and summarize threat reports |
| Incident Documentation | File + Casual | Generate incident reports |
| Alert Triage | Planner | Multi-step investigation workflows |
| SIEM Query Building | Coder | Generate SPL/KQL queries |
| IOC Extraction | Browser + Coder | Extract indicators from reports |

### Configuration for SOC Operations

#### 1. Custom Agent Prompts

Create SOC-specific prompts in `prompts/soc/`:

```
prompts/
└── soc/
    ├── soc_analyst_agent.txt
    ├── threat_intel_agent.txt
    ├── incident_response_agent.txt
    └── log_analysis_agent.txt
```

**Example: soc_analyst_agent.txt**
```
You are a Senior SOC Analyst AI assistant. Your responsibilities include:

1. SECURITY CONTEXT
- You analyze security events, alerts, and logs
- You understand MITRE ATT&CK framework
- You can interpret IOCs (IP, hash, domain, URL)
- You follow incident response procedures

2. OUTPUT FORMAT
- Always categorize alerts by severity (Critical/High/Medium/Low)
- Include MITRE ATT&CK technique IDs when applicable
- Provide actionable recommendations
- Reference relevant KB articles or playbooks

3. SECURITY BEST PRACTICES
- Never expose sensitive data in outputs
- Sanitize IP addresses and hashes in examples
- Follow least privilege principles
- Log all actions for audit trail

4. TOOLS AVAILABLE
- SIEM query execution (Splunk, Elastic, etc.)
- Threat intelligence lookups
- Log file analysis
- Report generation
```

#### 2. SOC-Specific Tools

Add custom tools in `sources/tools/`:

```python
# sources/tools/siem_tools.py
class SIEMTool(Tools):
    """Tool for SIEM query execution"""
    
    def __init__(self, siem_type="splunk"):
        self.siem_type = siem_type
        self.tag = "siem"
        self.description = "Execute SIEM queries and analyze results"
    
    def execute(self, query: str):
        # Connect to SIEM API
        # Execute query
        # Return formatted results
        pass


# sources/tools/threat_intel_tools.py
class ThreatIntelTool(Tools):
    """Tool for threat intelligence lookups"""
    
    def lookup_ioc(self, ioc_type: str, value: str):
        """
        Query threat intel sources:
        - VirusTotal
        - AlienVault OTX
        - AbuseIPDB
        - Internal TIP
        """
        pass
```

#### 3. SOC Agent Implementation

```python
# sources/agents/soc_agent.py
class SOCAnalystAgent(Agent):
    """Specialized agent for SOC operations"""
    
    def __init__(self, name, prompt_path, provider, verbose=False):
        super().__init__(name, prompt_path, provider, verbose)
        self.role = "security_analysis"
        self.type = "soc_agent"
        
        # SOC-specific tools
        self.tools = {
            "siem": SIEMTool(),
            "threat_intel": ThreatIntelTool(),
            "log_parser": LogParserTool(),
            "report_gen": ReportGeneratorTool()
        }
    
    async def investigate_alert(self, alert_data: dict):
        """
        Automated alert investigation workflow:
        1. Parse alert details
        2. Enrich IOCs with threat intel
        3. Query SIEM for related events
        4. Generate investigation summary
        5. Recommend response actions
        """
        pass
```

#### 4. SOC Workflow Examples

##### A. Automated Alert Triage

```python
# Example query for planner agent
"""
Investigate security alert:
- Alert ID: SOC-2024-12345
- Type: Suspicious PowerShell Execution
- Host: WORKSTATION-001
- User: jdoe

Tasks:
1. Query SIEM for all events from this host in last 24h
2. Check if source IP/user has prior incidents
3. Look up any file hashes in VirusTotal
4. Determine MITRE ATT&CK techniques
5. Generate incident report with severity rating
"""
```

##### B. Threat Intelligence Gathering

```python
# Browser agent task
"""
Search for recent threat intelligence on:
- APT group: Lazarus
- Focus: Latest TTPs and IOCs
- Sources: Mandiant, CrowdStrike, CISA

Output:
- Summary of recent campaigns
- List of IOCs (IPs, domains, hashes)
- MITRE ATT&CK mapping
- Recommended detections
"""
```

##### C. Log Analysis Automation

```python
# Coder agent task
"""
Write a Python script to:
1. Parse Windows Security Event logs (EVTX format)
2. Filter for Event IDs: 4624, 4625, 4648, 4672
3. Identify unusual logon patterns:
   - Off-hours authentication
   - Multiple failed attempts
   - Service account anomalies
4. Generate CSV report with findings
"""
```

### Integration Architecture for SOC Cloud

```
┌─────────────────────────────────────────────────────────────────┐
│                     SOC CLOUD INTEGRATION                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   AgenticSeek Core                       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│        ┌──────────────────┼──────────────────┐                  │
│        ▼                  ▼                  ▼                  │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐              │
│  │   SIEM   │      │   TIP    │      │  SOAR    │              │
│  │ Connector│      │ Connector│      │ Connector│              │
│  └────┬─────┘      └────┬─────┘      └────┬─────┘              │
│       │                 │                 │                     │
│       ▼                 ▼                 ▼                     │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐              │
│  │ Splunk/  │      │ MISP/    │      │ Phantom/ │              │
│  │ Elastic  │      │ OpenCTI  │      │ XSOAR    │              │
│  └──────────┘      └──────────┘      └──────────┘              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Security Considerations

1. **Data Protection**
   - Configure `WORK_DIR` to isolated sandbox
   - Implement data masking for sensitive fields
   - Audit log all agent actions

2. **Access Control**
   - API authentication for SIEM/TIP connections
   - Role-based tool access
   - Session timeout policies

3. **Network Security**
   - Run in isolated network segment
   - Whitelist allowed external domains
   - Proxy all web requests through security gateway

4. **Compliance**
   - Log retention policies
   - Data sovereignty requirements
   - Audit trail for all investigations

### Configuration Template

Add to `config.ini`:

```ini
[SOC]
siem_type = splunk
siem_api_url = https://splunk.internal:8089
tip_enabled = True
tip_sources = virustotal,abuseipdb,otx
work_dir_security = /secure/soc_workspace
audit_logging = True
data_masking = True
```

Add to `.env`:

```
SPLUNK_TOKEN='[API_TOKEN]'
VIRUSTOTAL_API_KEY='[API_KEY]'
ABUSEIPDB_API_KEY='[API_KEY]'
OTX_API_KEY='[API_KEY]'
```

---

## Appendix: Quick Reference

### File Structure
```
agenticSeek/
├── api.py                 # FastAPI server
├── cli.py                 # CLI interface
├── config.ini             # Configuration
├── sources/
│   ├── agents/
│   │   ├── agent.py       # Base agent class
│   │   ├── planner_agent.py
│   │   ├── browser_agent.py
│   │   ├── code_agent.py
│   │   ├── casual_agent.py
│   │   ├── file_agent.py
│   │   └── mcp_agent.py
│   ├── tools/             # Tool implementations
│   ├── browser.py         # Browser automation
│   ├── interaction.py     # User interaction
│   ├── memory.py          # Memory management
│   ├── router.py          # Agent routing
│   ├── llm_provider.py    # LLM abstraction
│   └── utility.py         # Helpers
├── prompts/
│   ├── base/              # Default prompts
│   └── jarvis/            # Jarvis personality
├── frontend/              # Web UI
└── tests/                 # Test suite
```

### Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `Agent` | `agents/agent.py` | Abstract base for all agents |
| `PlannerAgent` | `agents/planner_agent.py` | Multi-agent orchestration |
| `AgentRouter` | `router.py` | Intelligent agent selection |
| `Memory` | `memory.py` | Conversation management |
| `Interaction` | `interaction.py` | Session handling |
| `Provider` | `llm_provider.py` | LLM abstraction layer |
| `Browser` | `browser.py` | Web automation |

### Command Reference

```bash
# Start services
docker-compose up -d

# Run CLI
python cli.py

# Run API server
python api.py

# Run with specific config
python cli.py --config custom_config.ini
```

---

## References

1. [Manus AI Context Engineering Blog](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)
2. [OpenAI Practical Guide to Building Agents](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf)
3. [Google Cloud Agent Design Patterns](https://docs.cloud.google.com/architecture/choose-design-pattern-agentic-ai-system)
4. [LangGraph Multi-Agent Workflows](https://blog.langchain.dev/langgraph-multi-agent-workflows/)
5. [ReAct: Synergizing Reasoning and Acting](https://research.google/blog/react-synergizing-reasoning-and-acting-in-language-models/)
6. [Tree of Thoughts Paper](https://arxiv.org/abs/2305.10601)
7. [HiPlan: Hierarchical Planning for LLM Agents](https://arxiv.org/abs/2508.19076)
8. [IBM AI Agent Planning](https://www.ibm.com/think/topics/ai-agent-planning)

---

*Last Updated: January 2026*
*Version: 1.0*
