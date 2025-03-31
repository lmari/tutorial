from IPython.display import Markdown, display
from pprint import pprint
from openai import OpenAI
from duckduckgo_search import DDGS
import json

def _search_tool(query: str, max_results: int=3) -> list:
    return DDGS().text(query, max_results=max_results)

search_tool = {
    "type": "function",
    "function": {
        "name": "_search_tool",
        "description": "Cerca nel web pagine che abbiano come argomento la condizione di ricerca specificata",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "La condizione di ricerca"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Il numero massimo di risultati da restituire (3 se non specificato altrimenti)"
                }
            },
            "required": ["query"]
        }
    }
}
        
def _exec_tool(response, debug=False):
    if debug:
        print("L'intero oggetto JSON di risposta:")
        pprint(dict(response))
        print("\nIl messaggio generato:")
        pprint(response.choices[0].message)
    if response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        if debug:
            print("\nLa parte del messaggio riferita alla funzione da chiamare:")
            print(tool_call.function)
        function_name = tool_call.function.name
        function_arguments = json.loads(tool_call.function.arguments)
        if debug:
            pprint(f"\nLa funzione da chiamare è: {function_name}; i suoi argomenti sono: {function_arguments}")
        result = globals()[function_name](**function_arguments)
        if debug:
            pprint(f"\nIl risultato della funzione è: {result}")
        return result
    else:
        return None


class Context:
    def __init__(self, debug: bool=False):
        self.debug = debug
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
        if self.debug:
            print(target + "\n\n")
        else:
            display(Markdown(self.conversation), clear=True)


class Agent:
    def __init__(self, name: str, context: Context,
                 base_url: str="http://localhost:1234/v1",
                 role_and_skills: str="(da specificare)",
                 max_tokens: int=-1, temperature: float=0.7):
        self.client = OpenAI(base_url=base_url)
        self.name = name
        self.context = context
        self.role_and_skills = role_and_skills
        self.max_tokens = max_tokens
        self.temperature = temperature

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
                {"role": "system", "content": self.role_and_skills},
                {"role": "user", "content": message if message != "" else self.context.last_message if last_message_only else self.context.conversation}
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=not self.context.debug # non stream se in modo debug
        )
        if self.context.debug:
            print("*** " + self.name + " ***")
            pprint(dict(response)) # type: ignore
            print("\n\n")
        else:
            self._stream(response, add_to_conversation)


class SearchAgent(Agent):
    def __init__(self, name: str, context: Context,
                 base_url: str="http://localhost:1234/v1",
                 role_and_skills: str="(da specificare)",
                 max_tokens: int=-1, temperature: float=0.7):
        super().__init__(name, context, base_url, role_and_skills, max_tokens, temperature)


    def exec(self, message: str="", last_message_only: bool=False, add_to_conversation: bool=True) -> None:
        response = self.client.chat.completions.create(
            model="...",
            messages=[
                {"role": "system", "content": "Usa sempre lo strumento _search_tool.\n" + self.role_and_skills},
                {"role": "user", "content": message if message != "" else self.context.last_message if last_message_only else self.context.conversation}
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=False,
            tools=[search_tool] # type: ignore
        )
        if self.context.debug:
            print("*** " + self.name + " ***")
            pprint(dict(response)) # type: ignore
            print("\n\n")
            print(_exec_tool(response, debug=self.context.debug))
        else:
            try:
                final_response = _exec_tool(response, debug=self.context.debug)
                self.context._append("**" + self.name + "**: ", add_to_conversation, reinit_message=True)
                self.context._append(str(final_response), add_to_conversation)
            except Exception as e:
                self.context._append("Nessun risultato...", add_to_conversation)
            self.context._append(self.context.sep, add_to_conversation)
            display(Markdown(self.context.conversation), clear=True)
