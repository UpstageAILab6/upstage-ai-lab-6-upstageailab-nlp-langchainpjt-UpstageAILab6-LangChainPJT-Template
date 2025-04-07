from langchain_core.prompts import PromptTemplate

class GovPolicyPrompt:
    def __init__(self):
        self.prompt = PromptTemplate.from_template(
            """
You are an assistant for answering questions about Korean government support policies.
Use the following retrieved context to answer the user's question.
If none of the relevant information is found in the context, say "잘 모르겠습니다." Otherwise, do not include this phrase.

When answering, include:
- 지원대상 (who can apply)
- 지역 또는 관할기관 (where this applies / which region or city is responsible)
- 상세한 정책 설명 (혜택, 신청방법 등)

Use the following format for each policy:

사업명: ...

내용: 문장에서 핵심 내용뿐만 아니라 대상, 조건, 신청 방식 등을 가능한 자세히 설명하세요.

신청 URL 또는 문의처: ...

- 신청 URL이 있으면 "신청 URL: ..." 형식으로,
- 신청 URL이 없으면 반드시 "문의처: ..." 형식으로 포함하세요.

Respond in Korean.

# Context:
{context}

# Question:
{question}

# Answer:
"""
        )

    def get_prompt(self):
        return self.prompt


class MarkdownFormatter:
    def format(self, text: str) -> str:
        lines = text.strip().split("\n")
        md_lines = []

        for line in lines:
            if line.startswith("사업명:"):
                md_lines.append(f"**{line}**")
            elif line.startswith("내용:"):
                md_lines.append(f"📍 {line}")
            elif line.startswith("신청 URL"):
                url = line.split(":", 1)[-1].strip()
                if url and url != "잘 모르겠습니다.":
                    md_lines.append(f"🔗 [신청 방법]({url})")
                else:
                    md_lines.append(f"📞 신청 URL이 없어 문의처를 확인해 주세요.")
            elif line.startswith("문의처:"):
                contacts = line.replace("문의처:", "").strip().split("||")
                for contact in contacts:
                    contact = contact.strip()
                    if contact:
                        md_lines.append(f"📞 {contact}")
            else:
                md_lines.append(line)

        return "\n\n".join(md_lines)