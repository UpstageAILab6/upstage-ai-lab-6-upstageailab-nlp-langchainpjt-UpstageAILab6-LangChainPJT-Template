import datetime
from typing import List, Tuple

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


def build_prompt(context: str, attached_files: str, query: str)-> List[Tuple[str, str]]:
    """
    주어진 context, 첨부 파일 정보와 질의를 바탕으로 Langchain ChatPromptTemplate을 생성합니다.
    """
    messages = [
        (
            "system",
            (
                "You are an academy QA bot. The questioner is \"AI+Lab_7기(Upstage_AI+6기)\". "
                "Your role is to answer questions about academic policies and procedures, including leave applications. "
                "Thoroughly analyze the provided context and extract all relevant details. When referring to additional resources, "
                "explicitly include:\n- The names of any attached files (e.g., files with a '.docx' extension)\n"
                "- Their corresponding source links as provided in the context.\n"
                "If the context does not contain complete information but relevant attached files are available, instruct the user to consult these documents for further details.\n"
                "Additionally, format your answer using bullet points to enhance readability and ensure that source links are always included."
            )
        ),
        (
            "system",
            f"Here is the context you must reference:\n{context}"
        ),
        (
            "system",
            f"The following attached files may contain additional details:\n{attached_files}"
        ),
        (
            "human",
            f"Using the above context and attached file information, answer the following question comprehensively:\n\n{query}"
        ),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    return prompt


def get_vacation_messages(context_text: str, attached_files: str, question: str) -> List[Tuple[str, str]]:
    """
    '휴가' 카테고리에 특화된 메시지 템플릿을 반환합니다.
    """
    messages = [
        (
            "system",
            (
                "당신은 학원 QA 봇입니다. 질문자는 \"AI+Lab_7기(Upstage_AI+6기)\"입니다. "
                "당신의 역할은 학사 정책 및 절차, 휴학 신청 등과 관련된 질문에 답변하는 것입니다. "
                "제공된 컨텍스트를 철저히 분석하여 모든 관련 세부사항을 추출하세요. 추가 자료를 참조할 때에는 "
                "명시적으로 다음을 포함해야 합니다:\n"
                "- 첨부파일은 해당 기수에 적합해야힙니다.\n"
                "- 첨부파일은 질문과 관련있어야합니다.\n"
                "- 컨텍스트에 제공된 해당 소스 링크들\n"
                "컨텍스트에 완전한 정보가 포함되어 있지 않더라도, 관련 첨부 파일들이 있는 경우 사용자가 추가 정보를 위해 해당 문서를 참조하도록 안내하세요.\n"
                "또한, 가독성을 높이기 위해 답변은 bullet point 형식으로 작성하고, 소스 링크가 항상 포함되도록 하세요.\n"
                "답변은 반드시 한글로 작성되어야 합니다."
            )
        ),
        (
            "system",
            f"참조해야 하는 컨텍스트는 다음과 같습니다:\n{context_text}"
        ),
        (
            "system",
            f"추가 세부 정보가 포함될 수 있는 첨부 파일들은 다음과 같습니다:\n{attached_files}"
        ),
        (
            "human",
            f"위의 컨텍스트와 첨부 파일 정보를 사용하여, 다음 질문에 대해 포괄적으로 답변하세요:\n\n{question}"
        ),
    ]
    return messages


def get_timetable_messages(context_text: str, attached_files: str, question: str) -> List[Tuple[str, str]]:
    """
    '시간표' 카테고리에 특화된 메시지 템플릿을 반환합니다.
    """
    weekdays_kr = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]

    today = datetime.date.today()
    today_str = today.strftime("%Y%m%d") + f"({weekdays_kr[today.weekday()]})"

    # system 메시지(지시사항) 템플릿
    system_instructions_template = PromptTemplate(
        input_variables=["today_str"],
        template=(
            "당신은 학원 QA 봇(시간표 전담)입니다. "
            "질문자는 'AI+Lab_7기(Upstage_AI+6기)'이며, "
            "제공된 컨텍스트와 첨부파일을 참고하여 강의·일정·시간표 관련 질문에 답하세요.\n\n"
            "오늘 날짜는 {today_str}입니다.\n\n"
            "아래 **답변 형식**을 반드시 지켜주세요:\n"
            "1) 요일은 절대 표기하지 않습니다.\n"
            "2) 일자별 안내를 불릿 포인트로 정리합니다.\n"
            "3) 불필요한 세부정보는 제외하고, 핵심 일정만 간략히 정리합니다.\n"
        )
    )

    # system 메시지(컨텍스트) 템플릿
    system_context_template = PromptTemplate(
        input_variables=["context_text", "attached_files"],
        template=(
            "참조해야 하는 컨텍스트:\n{context_text}\n\n"
            "첨부파일:\n{attached_files}\n"
        )
    )

    # human 메시지(질문) 템플릿
    human_question_template = PromptTemplate(
        input_variables=["question"],
        template=(
            "아래 질문에 대해 시간표 기준으로 답변해주세요:\n\n"
            "{question}\n"
        )
    )

    # PromptTemplate를 통해 실제 메시지를 생성
    system_instructions = system_instructions_template.format(today_str=today_str)
    system_context = system_context_template.format(
        context_text=context_text,
        attached_files=attached_files
    )
    human_question = human_question_template.format(question=question)

    # ChatGPT 등의 모델에 전달할 수 있는 형태로 반환
    messages = [
        ("system", system_instructions),
        ("system", system_context),
        ("human", human_question),
    ]

    return messages


def get_legal_messages(context_text: str, attached_files: str, question: str) -> List[Tuple[str, str]]:
    """
    '법령' 카테고리에 특화된 메시지 템플릿.
    """
    messages = [
        (
            "system",
            "당신은 학원 QA 봇입니다. (법령 관련 전담)"
            "법률, 규정, 제도 등에 대해 안내하는 역할을 합니다. "
            "질문자는 \"AI+Lab_7기(Upstage_AI+6기)\"입니다. "
            "컨텍스트, 첨부 파일을 참고하여 최대한 정확한 정보를 제공합니다. "
            "답변은 bullet point 형식 + 한글로 작성하세요."
        ),
        (
            "system",
            f"참조해야 하는 컨텍스트:\n{context_text}"
        ),
        (
            "human",
            f"아래 질문에 대해 법령의 관점에서 답변해주세요:\n\n{question}"
        ),
    ]
    return messages
