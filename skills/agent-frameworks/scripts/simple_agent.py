#!/usr/bin/env python3
"""
Simple Agent Framework - Lightweight agent implementation.

Features:
- Tool registration and execution
- ReAct pattern (Reason + Act)
- Memory management
- Error handling

Usage:
    from simple_agent import Agent, Tool

    agent = Agent(llm_client)
    agent.register_tool(search_tool)
    result = agent.run("Find information about AI")
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any
from abc import ABC, abstractmethod
import json
import re


@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    output: str
    error: Optional[str] = None


@dataclass
class AgentStep:
    """Single step in agent execution."""
    thought: str
    action: Optional[str] = None
    action_input: Optional[str] = None
    observation: Optional[str] = None


class Tool(ABC):
    """Abstract base class for tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def run(self, input: str) -> ToolResult:
        pass


class SearchTool(Tool):
    """Web search tool."""

    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return "Search the web for information. Input: search query"

    def run(self, input: str) -> ToolResult:
        # Placeholder - would use actual search API
        return ToolResult(
            success=True,
            output=f"Search results for '{input}': [placeholder results]"
        )


class CalculatorTool(Tool):
    """Calculator tool."""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Perform mathematical calculations. Input: math expression"

    def run(self, input: str) -> ToolResult:
        try:
            # Safe eval for math expressions
            result = eval(input, {"__builtins__": {}}, {})
            return ToolResult(success=True, output=str(result))
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class Memory:
    """Simple conversation memory."""

    def __init__(self, max_tokens: int = 4000):
        self.history: List[Dict[str, str]] = []
        self.max_tokens = max_tokens

    def add(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        self._trim()

    def get_context(self) -> str:
        return "\n".join([
            f"{h['role']}: {h['content']}"
            for h in self.history
        ])

    def _trim(self) -> None:
        # Simple trimming - remove oldest messages
        while len(self.get_context()) > self.max_tokens and len(self.history) > 2:
            self.history.pop(0)

    def clear(self) -> None:
        self.history = []


class Agent:
    """ReAct-style agent with tool use."""

    REACT_PROMPT = """You are a helpful AI assistant with access to tools.

Available tools:
{tools}

Use this format:

Thought: [Think about what to do next]
Action: [tool name]
Action Input: [input to the tool]

Or if you have the final answer:
Thought: [Explain your conclusion]
Final Answer: [Your response to the user]

{memory}

User: {input}

Begin!
"""

    def __init__(self, llm_client, max_iterations: int = 10):
        self.llm = llm_client
        self.tools: Dict[str, Tool] = {}
        self.memory = Memory()
        self.max_iterations = max_iterations
        self.steps: List[AgentStep] = []

    def register_tool(self, tool: Tool) -> None:
        """Register a tool for the agent to use."""
        self.tools[tool.name] = tool

    def _format_tools(self) -> str:
        """Format tool descriptions for the prompt."""
        lines = []
        for name, tool in self.tools.items():
            lines.append(f"- {name}: {tool.description}")
        return "\n".join(lines)

    def _parse_response(self, response: str) -> AgentStep:
        """Parse LLM response into an agent step."""
        step = AgentStep(thought="")

        # Extract thought
        thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|Final Answer:|$)", response, re.DOTALL)
        if thought_match:
            step.thought = thought_match.group(1).strip()

        # Check for final answer
        final_match = re.search(r"Final Answer:\s*(.+?)$", response, re.DOTALL)
        if final_match:
            step.thought = step.thought or "Providing final answer"
            step.observation = final_match.group(1).strip()
            return step

        # Extract action
        action_match = re.search(r"Action:\s*(\w+)", response)
        if action_match:
            step.action = action_match.group(1).strip()

        # Extract action input
        input_match = re.search(r"Action Input:\s*(.+?)(?=Observation:|$)", response, re.DOTALL)
        if input_match:
            step.action_input = input_match.group(1).strip()

        return step

    def _execute_tool(self, action: str, action_input: str) -> str:
        """Execute a tool and return the observation."""
        if action not in self.tools:
            return f"Error: Tool '{action}' not found. Available: {list(self.tools.keys())}"

        tool = self.tools[action]
        result = tool.run(action_input)

        if result.success:
            return result.output
        else:
            return f"Error: {result.error}"

    def run(self, input: str) -> str:
        """Run the agent on an input."""
        self.steps = []

        # Build initial prompt
        prompt = self.REACT_PROMPT.format(
            tools=self._format_tools(),
            memory=self.memory.get_context(),
            input=input
        )

        for iteration in range(self.max_iterations):
            # Get LLM response
            response = self.llm.generate(prompt)

            # Parse response
            step = self._parse_response(response.text)
            self.steps.append(step)

            # Check for final answer
            if step.action is None and step.observation:
                self.memory.add("user", input)
                self.memory.add("assistant", step.observation)
                return step.observation

            # Execute tool
            if step.action:
                observation = self._execute_tool(step.action, step.action_input or "")
                step.observation = observation

                # Add to prompt for next iteration
                prompt += f"\n\nThought: {step.thought}"
                prompt += f"\nAction: {step.action}"
                prompt += f"\nAction Input: {step.action_input}"
                prompt += f"\nObservation: {observation}"

        # Max iterations reached
        return "I was unable to complete the task within the allowed steps."

    def get_execution_trace(self) -> List[Dict[str, Any]]:
        """Get the execution trace for debugging."""
        return [
            {
                "thought": step.thought,
                "action": step.action,
                "action_input": step.action_input,
                "observation": step.observation
            }
            for step in self.steps
        ]


class CrewStyleAgent:
    """Multi-agent orchestration (CrewAI-inspired)."""

    def __init__(self, llm_client):
        self.llm = llm_client
        self.agents: Dict[str, Dict[str, Any]] = {}

    def add_agent(self, name: str, role: str, goal: str, backstory: str = "") -> None:
        """Add an agent with a specific role."""
        self.agents[name] = {
            "role": role,
            "goal": goal,
            "backstory": backstory
        }

    def run_task(self, task: str, agent_name: str) -> str:
        """Run a task with a specific agent."""
        agent = self.agents.get(agent_name)
        if not agent:
            raise ValueError(f"Agent '{agent_name}' not found")

        prompt = f"""You are a {agent['role']}.
Goal: {agent['goal']}
Background: {agent['backstory']}

Task: {task}

Complete the task to the best of your ability:"""

        response = self.llm.generate(prompt)
        return response.text

    def run_crew(self, tasks: List[Dict[str, str]]) -> List[str]:
        """Run multiple tasks with different agents."""
        results = []

        for task in tasks:
            result = self.run_task(task["description"], task["agent"])
            results.append(result)

        return results


# Factory function
def create_agent(llm_client, tools: List[str] = None) -> Agent:
    """Create an agent with specified tools."""
    agent = Agent(llm_client)

    tool_map = {
        "search": SearchTool(),
        "calculator": CalculatorTool()
    }

    for tool_name in (tools or ["search", "calculator"]):
        if tool_name in tool_map:
            agent.register_tool(tool_map[tool_name])

    return agent


if __name__ == "__main__":
    # Demo with mock LLM
    class MockLLM:
        def generate(self, prompt):
            class Response:
                text = """Thought: I should search for information about AI.
Action: search
Action Input: artificial intelligence overview"""
            return Response()

    llm = MockLLM()
    agent = create_agent(llm)

    print("Agent created with tools:", list(agent.tools.keys()))
    print("\nSimulating agent run...")

    # In real usage:
    # result = agent.run("What is artificial intelligence?")
    # print(result)
