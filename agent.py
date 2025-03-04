from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.output_parsers import StrOutputParser
# from langchain import hub
from tools.react_prompt_template import get_prompt_template
from tools.pdf_query_tools import Litigation_pdf_query, ICAI_pdf_query
import warnings


def agent(query: str):
    warnings.filterwarnings("ignore", category=FutureWarning)

    LLM = ChatGroq(model="llama3-8b-8192")  # for  more api limit

    # query = input("Enter your query: ")

    tools = [Litigation_pdf_query, ICAI_pdf_query]

    prompt_template = get_prompt_template()

    agent = create_react_agent(
        LLM,
        tools,
        prompt_template
    )

    # output_parser = StrOutputParser()

    # chain = LLM | output_parser
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)

    result = agent_executor.invoke({"input": query})
    # print(result)
    return result["output"]
