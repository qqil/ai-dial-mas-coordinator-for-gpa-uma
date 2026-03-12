import json
from typing import Optional

import httpx
from aidial_sdk.chat_completion import Role, Request, Message, Stage, Choice
from pydantic import StrictStr


_UMS_CONVERSATION_ID = "ums_conversation_id"


class UMSAgentGateway:

    def __init__(self, ums_agent_endpoint: str):
        self.ums_agent_endpoint = ums_agent_endpoint

    async def response(
            self,
            choice: Choice,
            stage: Stage,
            request: Request,
            additional_instructions: Optional[str]
    ) -> Message:
        #TODO:
        # ⚠️ Important point: we need to provide Agent with conversation history that is related to this particular
        #    Agent, otherwise it will confuse the Agent.
        # 1. Get UMS conversation id. UMS Agent is custom implementation that is storing all the conversation on its
        #    side and without created conversation we are unable to communicate with UMS agent.
        #    The `ums_conversation_id` with be persisted in some of assistant message state (if conversation was created),
        #    additionally we will have 1-to-1 relation (one our conversation will have one conversation on the UMS agent side)
        # 2. If no conversation id found then create new conversation and set it to choice state as dict {_UMS_CONVERSATION_ID: {id}}
        # 3. Get last message (the last always will be the user message) and make augmentation with additional instructions
        # 4. Call UMS Agent
        # 5. return assistant message
        conversation_id = self.__get_ums_conversation_id(request=request)
        if not conversation_id:
            conversation_id = await self.__create_ums_conversation()
            print(f"Created new conversation on UMS agent side with id: {conversation_id}")

        last_user_message = request.messages[-1].content
        if additional_instructions:
            last_user_message = f"{last_user_message}\n\n{additional_instructions}"

        agent_response = await self.__call_ums_agent(
            conversation_id=conversation_id,
            user_message=last_user_message, # type: ignore
            stage=stage
        )

        choice.set_state({
            _UMS_CONVERSATION_ID: conversation_id
        })

        return Message(
            role=Role.ASSISTANT,
            content=agent_response,
        )


    def __get_ums_conversation_id(self, request: Request) -> Optional[str]:
        """Extract UMS conversation ID from previous messages if it exists"""
        #TODO:
        # Iterate through message history, check if custom content with state is present and if it contains
        # _UMS_CONVERSATION_ID, if yes then return it, otherwise return None
        for message in request.messages:
            if (
                message.custom_content and
                message.custom_content.state and
                _UMS_CONVERSATION_ID in message.custom_content.state
            ):
                return message.custom_content.state[_UMS_CONVERSATION_ID]
        return None

    async def __create_ums_conversation(self) -> str:
        """Create a new conversation on UMS agent side"""
        #TODO:
        # 1. Create async context manager with httpx.AsyncClient()
        # 2. Make POST request to create conversation https://github.com/khshanovskyi/ai-dial-ums-ui-agent/blob/completed/agent/app.py#L159
        # 3. Get response json and return `id` from it
        async_context_manager = httpx.AsyncClient()
        async with async_context_manager as client:
            response = await client.post(
                f"{self.ums_agent_endpoint}/conversations", 
                json={"title": "UMS Agent Conversation"}, 
                timeout=30
            )

            response.raise_for_status()
            conversation_data = response.json()
            return conversation_data["id"]

    async def __call_ums_agent(
            self,
            conversation_id: str,
            user_message: str,
            stage: Stage
    ) -> str:
        """Call UMS agent and stream the response"""
        #TODO:
        # 1. Create async context manager with httpx.AsyncClient()
        # 2. Make POST request to chat https://github.com/khshanovskyi/ai-dial-ums-ui-agent/blob/completed/agent/app.py#L216
        #    it applies message as request body: {"message": { "role": "user","content": user_message},"stream": True}
        #    streaming must be enabled
        # 3. Now is the time to recall the first practice with console chat when we parsed raw streaming responses,
        #    don't worry, hopefully we made response in openai compatible (the same as in openai spec).
        #    Make async loop through `response.aiter_lines()` and:
        #       - Cut the `data: `. The streaming chunks will be returned in such format:
        #         data: {'choices': [{'delta': {'content': 'chunk 1'}}]}
        #         data: {'choices': [{'delta': {'content': 'chunk 2'}}]}
        #         data: {'choices': [{'delta': {'content': 'chunk ...'}}]}
        #         data: {'choices': [{'delta': {'content': 'chunk n'}}]}
        #         data: {'conversation_id': '{conversation_id}'}
        #         data: [DONE]
        #       - If in result you have [DONE] - that means that streaming is finished an you can break the loop
        #       - Make dict from json
        #       - Get content, accumulate it to return after and append content chunks to the stage
        async_context_manager = httpx.AsyncClient()
        async with async_context_manager as client:
            response = await client.post(
                f"{self.ums_agent_endpoint}/conversations/{conversation_id}/chat",
                json={"message": {"role": "user","content": user_message}, "stream": True},
                timeout=30
            )

            response.raise_for_status()

            content = ""
            async for line in response.aiter_lines():
                if not line.startswith('data:'):
                    print(f"Unexpected line format: {line}")
                    continue

                data = line[len('data: '):]

                if data == "[DONE]":
                    break

                if not line.startswith("data: "):
                    continue

                
                try:
                    chunk = json.loads(data)
                    if 'conversation_id' in chunk:
                        continue

                    if 'choices' in chunk and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0]["delta"]
                        
                        if 'content' in delta:
                            content += delta["content"]
                            stage.append_content(delta["content"])

                except json.JSONDecodeError:
                    print(f"Failed to parse JSON from line: {line}")
                    continue
            return content