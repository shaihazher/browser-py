"""browser-py agent — LLM-powered browser automation.

The agent connects to any LLM provider (via LiteLLM) and uses browser-py
tools to accomplish tasks autonomously.

    from browser_py.agent import Agent

    agent = Agent()
    agent.chat("Go to github.com and find trending repos")
"""
