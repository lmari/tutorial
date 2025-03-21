from IPython.display import Markdown, display
from pprint import pprint
from openai import OpenAI

conversation = ""
last_message = ""
sep = "\n\n<p style='text-align:center'>***</p>\n\n"

def stream_print(response, max_length=100):
    length = 0
    for chunk in response:
        text = chunk.choices[0].delta
        if hasattr(text, 'content') and text.content:
            print(text.content, end='', flush=True)
            length += len(text.content)
            if length > max_length:
                print()
                length = 0

def print_markdown(name, response, conversation):
    display(Markdown(conversation + "**" + name + "**: " + response + sep), clear=True)

def stream_print_markdown(name, response, conversation, printing: bool=True) -> str:
    buffer = "**" + name + "**: "
    local_conversation = conversation + buffer
    for chunk in response:
        text = chunk.choices[0].delta
        if hasattr(text, 'content') and text.content:
            buffer += text.content
            local_conversation += text.content
            if printing:
                display(Markdown(local_conversation), clear=True)
    buffer += sep
    return buffer

def set_context(target: str) -> None:
    global last_message, conversation
    last_message = target + sep
    conversation = last_message
    display(Markdown(conversation), clear=True)

class Agent:
    def __init__(self, name: str, base_url: str="http://localhost:1234/v1",
                 system_message: str="Rispondi sempre in italiano.", printing: bool=True, stream: bool=True,
                 max_tokens: int=-1, temperature: float=0.7, full_response: bool=False):
        self.client = OpenAI(base_url=base_url)
        self.name = name
        self.system_message = system_message
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.stream = stream
        self.full_response = full_response
        self.printing = printing
        self.last_response = ""

    def request(self, message: str="", last_message_only: bool=False) -> None:
        global last_message, conversation
        response = self.client.chat.completions.create(
            model="...",
            messages=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": message if message is not "" else last_message if last_message_only else conversation}
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=False if self.full_response else self.stream # non stream se full_response
        )
        if not self.stream or self.full_response:
            self.last_response = response.choices[0].message.content # type: ignore
            if self.printing:
                if self.full_response:
                    print("*** " + self.name + " ***")
                    pprint(dict(response)) # type: ignore
                    print()
                else:
                    print_markdown(self.name, self.last_response, conversation)
                    self.last_response = "**" + self.name + "**: " + self.last_response + sep # type: ignore
        else:
            self.last_response = stream_print_markdown(self.name, response, conversation, printing=self.printing)
        last_message = self.last_response
        conversation += self.last_response # type: ignore
