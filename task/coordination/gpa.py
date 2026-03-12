from copy import deepcopy
from typing import Optional, Any

from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Role, Choice, Request, Message, CustomContent, Stage, Attachment
from pydantic import StrictStr

from task.stage_util import StageProcessor

_IS_GPA = "is_gpa"
_GPA_MESSAGES = "gpa_messages"


class GPAGateway:

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    async def response(
            self,
            choice: Choice,
            stage: Stage,
            request: Request,
            additional_instructions: Optional[str]
    ) -> Message:
        #TODO:
        # ℹ️ Cool thing about DIAL that all the apps that implement /chat/completions endpoint within DIAL infrastructure
        #    can be openai compatible and DIAL SDK supports us with creation of such applications. So, we can use any
        #    OpenAI compatible client (from openai, azureopenai, langchain, whatever...) and communicate with it like
        #    with openai LLM through /chat/completions endpoint.
        #    BZW this app as well implements /chat/completions endpoint.
        # ---
        # 1. Create AsyncDial (api_version='2025-01-01-preview')
        # 2. Make call, you will need to provide such parameters:
        #       - it should stream response
        #       - prepared messages with `additional_instructions`
        #       - `general-purpose-agent` is deployment name that we will call
        #          Full final URL will be `http://host.docker.internal:8052/openai/deployments/general-purpose-agent/chat/completions`
        #       - extra_headers={'x-conversation-id': request.headers.get('x-conversation-id')}
        #         Extra headers are required in this case since we have the logic in GPA for RAG that requires them
        # 3. Create such variables:
        #       - content, here we will collect content
        #       - result_custom_content: CustomContent with empty attachment list, here we will collect attachments
        #       - stages_map: dict[int, Stage] = {}, here we will the mirrored stages by their indexes
        # 4. Make async loop through chunks and:
        #       - get delta and print it to console it will later help you to see what is coming in response from GPA
        #       - if delta has content append it to the `stage`
        #       - if delta has custom_content it is time for magic✨
        #           - if custom_content has attachments then add them to the `result_custom_content` (we will deal with them later)
        #           - if custom_content has state, as well add it to `result_custom_content`. It is important for us
        #             since here persisted all the tool calls information that is required to GPA to restore the context
        #           - And now is magic, we will propagate stages from GPA to out MAS coordinator:
        #               - make dict with none excluded from the custom_content
        #               - if it has 'stages':
        #                   - iterate through them (iterated stage we will name as `stg`) and:
        #                       - get 'index' from it (each stage has it is index, it is required param)
        #                       - if `stages_map` contains Stage by such index then:
        #                           - if `stg` has 'content' then append it to the Stage by index
        #                           - if `stg` has 'attachments' then iterate through these attachments and add them to the Stage by index
        #                           - if stg` has 'status' and it is 'completed' then close the stage by index (StageProcessor.close_stage_safely)
        #                       - otherwise: we need to open the Stage on our side to propagate GPA Stage data, use
        #                         StageProcessor and don't forget to put it to the `stages_map` by index
        # 5. Propagate `result_custom_content` to choice.
        #    ⚠️ Here is the moment that aidial_client.AsyncDial and aidial_sdk has different pydentic models (classes)
        #       for Attachment, so, you can make such move to handle it `Attachment(**attachment.dict(exclude_none=True))`
        # 6. Now we need to to save information about conversation with GPA to the MASCoordinator choice state. Create
        #    dict {_IS_GPA: True, GPA_MESSAGES: result_custom_content.state} and set it to the choice state.
        # 7. Return assistant message with content
        dial_client = AsyncDial(
            api_version='2025-01-01-preview',
            base_url=self.endpoint,
            api_key=request.api_key
        )

        messages = self.__prepare_gpa_messages(request=request, additional_instructions=additional_instructions)
        extra_headers = {'x-conversation-id': request.headers.get('x-conversation-id', "")}

        response = await dial_client.chat.completions.create(
            deployment_name="general-purpose-agent",
            messages=messages, # type: ignore
            extra_headers=extra_headers,
            stream=True
        )

        content = ""
        result_custom_content = CustomContent(attachments=[])
        stages_map: dict[int, Stage] = {}
        
        if not result_custom_content.attachments:
            result_custom_content.attachments = []

        async for chunk in response:
            if not chunk.choices or len(chunk.choices) == 0:
                 continue
            
            delta = chunk.choices[0].delta

            if not delta:
                continue

            if delta.content:
                content += delta.content
                stage.append_content(delta.content)
            
            if delta.custom_content:
                custom_content_dict = delta.custom_content.dict(exclude_none=True)

                if delta.custom_content.attachments:
                    result_custom_content.attachments.extend(delta.custom_content.attachments)
                if delta.custom_content.state:
                    result_custom_content.state = delta.custom_content.state
                
                custom_content_dict = delta.custom_content.dict(exclude_none=True)
                if 'stages' in custom_content_dict:
                    for stg in custom_content_dict['stages']:
                        stage_index = stg['index']
                        existing_stage = stages_map.get(stage_index)

                        if existing_stage:
                            if stg.get('content'):
                                existing_stage.append_content(stg.get('content'))
                            if stg.get('attachments'):
                                for attachment in stg.get('attachments'):
                                    existing_stage.add_attachment(Attachment(**attachment))
                            if stg.get('status') and stg.get('status') == 'completed':
                                StageProcessor.close_stage_safely(existing_stage)
                        else:
                            stages_map[stage_index] = StageProcessor.open_stage(choice=choice, name=stg['name'])
        
        for stage in stages_map.values():
            StageProcessor.close_stage_safely(stage)
        
        for attachment in result_custom_content.attachments:
            choice.add_attachment(Attachment(**attachment.dict(exclude_none=True)))

        choice.set_state({
            _IS_GPA: True, 
            _GPA_MESSAGES: result_custom_content.state
        })

        return Message(
            role=Role.ASSISTANT,
            content=content
        )

    def __prepare_gpa_messages(self, request: Request, additional_instructions: Optional[str]) -> list[dict[str, Any]]:
        #TODO:
        # 1. Create `res_messages` empty array, here we will collect all the messages that are related to the GPA agent
        # 2. Make for i loop through range of len of request messages and:
        #       - if it is assistant message then:
        #           - Check if it has custom content and it has state:
        #               - if state dict has `_IS_GPA` and it is true then:
        #                   - 1. add to `res_messages` `request.messages[idx-1]` as dict with none excluded. Here we
        #                     add user message
        #                   - 2. make deepcopy of message, then set copied message with state from _GPA_MESSAGES add to
        #                     `res_messages`. What we do here is to restore appropriate format of assistant message for
        #                     GPA: {_IS_GPA: True, GPA_MESSAGES: {'tool_call_history': [{...}]}} -> {'tool_call_history': [{...}]}
        # 3. Add last message from `additional_instructions` (it will be user message) as dict with none excluded
        # 4. If `additional_instructions` are present we need to make augmentation for last message content in the `res_messages`
        # 5. Return `res_messages`
        res_messages = []
        for idx in range(len(request.messages)):
            message = request.messages[idx]
            if message.role == Role.ASSISTANT and message.custom_content and message.custom_content.state:
                state_dict = message.custom_content.state
                if state_dict.get(_IS_GPA):
                    res_messages.append(request.messages[idx-1].dict(exclude_none=True))
                    copied_message = deepcopy(message)
                    
                    if not copied_message.custom_content:
                        copied_message.custom_content = CustomContent()

                    copied_message.custom_content.state = state_dict.get(_GPA_MESSAGES)
                    res_messages.append(copied_message.dict(exclude_none=True))
        
        last_user_message = request.messages[-1]
        if additional_instructions:
            res_messages.append({
                "role": Role.USER,
                "content": f"{last_user_message.content}\n\n{additional_instructions}",
                "custom_content": last_user_message.custom_content.dict(exclude_none=True) if last_user_message.custom_content else None
            })
        else:  
            res_messages.append(last_user_message.dict(exclude_none=True))

        return res_messages
