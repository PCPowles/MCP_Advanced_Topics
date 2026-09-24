import asyncio
import os
from anthropic import AsyncAnthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.session import RequestContext
from mcp.types import (
    CreateMessageRequestParams,
    CreateMessageResult,
    TextContent,
    SamplingMessage,
)

from dotenv import load_dotenv
load_dotenv()

anthrpc_ap_ky = os.getenv("ANTHROPIC_API_KEY")
anthrpc_wrkspc_id  = os.getenv("ANTHROPIC_WORKSPACE_ID")
# print(f"anthrpc_ap_ky: {anthrpc_ap_ky}")
# print(f"anthrpc_wrkspc_id: {anthrpc_wrkspc_id}")

anthropic_client = AsyncAnthropic(api_key=anthrpc_ap_ky, default_headers={"anthropic-workspace-id": anthrpc_wrkspc_id},)


model = "claude-sonnet-5"

server_params = StdioServerParameters(
    command="uv",
    args=["run", "server.py"],
)


async def chat(input_messages: list[SamplingMessage], max_tokens=4000):
    messages = []
    for msg in input_messages:
        if msg.role == "user" and msg.content.type == "text":
            content = (
                msg.content.text
                if hasattr(msg.content, "text")
                else str(msg.content)
            )
            messages.append({"role": "user", "content": content})
        elif msg.role == "assistant" and msg.content.type == "text":
            content = (
                msg.content.text
                if hasattr(msg.content, "text")
                else str(msg.content)
            )
            messages.append({"role": "assistant", "content": content})

    response = await anthropic_client.messages.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )

    text = "".join([p.text for p in response.content if p.type == "text"])
    return text


async def sampling_callback(
    context: RequestContext, params: CreateMessageRequestParams
):
    # Call Claude using the Anthropic SDK
    text = await chat(params.messages)

    return CreateMessageResult(
        role="assistant",
        model=model,
        content=TextContent(type="text", text=text),
    )


async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(
            read, write, sampling_callback=sampling_callback
        ) as session:
            await session.initialize()

            result = await session.call_tool(
                name="summarize",
                # arguments={"text_to_summarize": "lots of text"},
                # write a report about archaeology
                arguments={"text_to_summarize": "Write a report about archaeology"},
            )
            print(result.content)


if __name__ == "__main__":
    import asyncio

    asyncio.run(run())
