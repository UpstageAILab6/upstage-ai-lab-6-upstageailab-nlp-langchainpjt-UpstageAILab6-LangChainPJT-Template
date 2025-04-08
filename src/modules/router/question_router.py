import json
from pydantic import BaseModel, Field
from typing import Literal
from langchain.prompts import ChatPromptTemplate
from langchain_upstage import ChatUpstage

class RouteQuery(BaseModel):
    category: Literal["vacation", "timetable", "legal"] = Field(
        ...,
        description="Given a user question, choose which category among [vacation, timetable, legal]."
    )

def route_question(question: str) -> str:
    system_prompt = """\
You are an expert router that determines the question category among "vacation", "timetable", or "legal".
Return a JSON object in the format: 
{{
  "category": "<vacation or timetable or legal>"
}}
No extra keys, no additional explanations.
"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}")
        ]
    )

    chat = ChatUpstage(temperature=1)
    chain = prompt | chat
    raw_response = chain.invoke({"question": question})

    try:
        data = json.loads(raw_response.content.strip())
        route_result = RouteQuery(**data)
        return route_result.category
    except Exception:
        return "etc"