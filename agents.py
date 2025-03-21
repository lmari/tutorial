from IPython.display import Markdown, display
from pprint import pprint
from openai import OpenAI


class Context:
    def __init__(self):
        self.team = []
        self.last_message = ""
        self.conversation = ""
        self.buffer = ""
        self.sep = "\n\n<p style='text-align:center'>***</p>\n\n"

    def _append(self, text: str, add_to_conversation: bool=True, reinit_message: bool=False) -> None:
        self.last_message = text if reinit_message else self.last_message + text
        self.buffer += text
        if add_to_conversation:
            self.conversation += text

    def make_team(self, agents: list) -> None:
        for agent in agents:
            self.team.append(agent)

    def start(self, target: str, add_to_conversation: bool=True) -> None:
        self._append(target + self.sep, add_to_conversation)
        display(Markdown(self.conversation), clear=True)


class Agent:
    def __init__(self, name: str, context: Context,
                 base_url: str="http://localhost:1234/v1",
                 system_message: str="Rispondi sempre in italiano.",
                 max_tokens: int=-1, temperature: float=0.7, full_response: bool=False):
        self.client = OpenAI(base_url=base_url)
        self.name = name
        self.context = context
        self.system_message = system_message
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.full_response = full_response

    def _stream(self, response, add_to_conversation: bool=True):
        self.context._append("**" + self.name + "**: ", add_to_conversation, reinit_message=True)
        for chunk in response:
            text = chunk.choices[0].delta
            if hasattr(text, 'content') and text.content:
                self.context._append(text.content, add_to_conversation)
                display(Markdown(self.context.conversation), clear=True)
        self.context._append(self.context.sep, add_to_conversation)
        display(Markdown(self.context.conversation), clear=True)

    def request(self, message: str="", last_message_only: bool=False, add_to_conversation: bool=True) -> None:
        response = self.client.chat.completions.create(
            model="...",
            messages=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": message if message != "" else self.context.last_message if last_message_only else self.context.conversation}
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=not self.full_response # non stream se full_response
        )
        if self.full_response:
            print("*** " + self.name + " ***")
            pprint(dict(response)) # type: ignore
            print()
        else:
            self._stream(response, add_to_conversation)
