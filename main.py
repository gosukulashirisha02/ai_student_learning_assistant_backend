from fastapi import FastAPI, UploadFile,File,Query
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from tavily import TavilyClient

from dotenv import load_dotenv
load_dotenv()
import shutil
import os


app=FastAPI()

@app.get("/")
def home():
    return {"msg": "backend running sucessfully"}

client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


llm=ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

pdf_content=""


@tool
def pdf_reader_tool(file_path:str):
    """
    Read PDF Documents
    """
    loader=PyPDFLoader(file_path)
    docs=loader.load()
    text = "\n".join(
        [doc.page_content for doc in docs]
    )
    return text[:10000]
@tool
def web_search_tool(question:str):
    """
    Search the web and return relevant information.
    Use this tool whenever the user asks for
    current information, explanations, trends,
    technologies, tutorials, etc.
    """
    result=client.search(
        query=question,
        max_results=5
    )
    return result

@tool
def quiz_generator_tool(pdf_content: str):
    """
    Generate 10 MCQ questions from PDF content.
    """

    response = llm.invoke(
        f"""
        Using the PDF content below:

        {pdf_content}

        Generate exactly 10 MCQ questions.

        Rules:
        - Show all 10 questions first.
        - Each question should be displayed as:

        Question 1:
        <question>

        A) option
        B) option
        C) option
        D) option

        - Do NOT show the answer after each question.

        - After Question 10, create a separate section:

        ====================
        ANSWER KEY
        ====================

        1. B
        2. A
        3. D
        ...
        10. C

        Return only the quiz and answer key.
        """
    )

    return response.content

@tool
def pdf_summarizer_tool(pdf_content: str):
    """
    Summarize PDF content.
    Use this tool whenever the user wants
    notes, summary, key points, or highlights.
    """
    response = llm.invoke(
        f"""
        Summarize the following PDF content.

        PDF Content:
        {pdf_content}

        Provide:

        1. Summary
        2. Key Points
        3. Important Concepts
        """
    )

    return response.content
   
  

agent=create_agent(
    model=llm,
    tools=[
        pdf_reader_tool,web_search_tool,quiz_generator_tool,pdf_summarizer_tool
    ]
)

@app.post("/pdf_reader")
async def pdf_reader(
    question: str = Query(...),
    file: UploadFile = File(...)
):
    os.makedirs("uploads", exist_ok=True)

    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    pdf_text = "\n".join(
        [doc.page_content for doc in docs]
    )

    response = llm.invoke(
    f"""
    You are an expert teacher.

    Use ONLY the PDF content below to answer.

    PDF Content:
    {pdf_text[:8000]}

    User Question:
    {question}

    Instructions:
    - Give a detailed explanation.
    - Explain the concept in simple language.
    - Include definition.
    - Explain why it is important.
    - Give examples if available in the PDF.
    - Use bullet points when needed.
    - Answer in at least 8-15 lines.
    - Make the answer easy for students to understand.
    - Do not say "According to the PDF".
    """
)

    return {
        "result": response.content
    }

  
@app.post("/web_search")
def web_search(
    question:str=Query(...)
):
    result=agent.invoke({
        "messages":[
            {
                "role":"user",
                "content":f"""
                Use web_search_tool.
                Answer this question:
                {question}
                give clear and detailed answer
                """
            }
        ]
    })
    return result
@app.post("/quiz_generator")
async def quiz_generator(file: UploadFile = File(...)):

    os.makedirs("uploads", exist_ok=True)

    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    pdf_text = "\n".join(
        [doc.page_content for doc in docs]
    )

    response = llm.invoke(
    f"""
    Using the PDF content below:

    {pdf_text[:3000]}

    Create exactly 10 MCQ questions.

    STRICT FORMAT:

    Question 1:
    What is Python?

    A) Option A
    B) Option B
    C) Option C
    D) Option D

    Question 2:
    ...

    Continue until Question 10.

    IMPORTANT:
    - Show ALL QUESTIONS FIRST.
    - Each option must be on a NEW LINE.
    - Do NOT show answers after each question.

    After Question 10 display:

    ==================
    ANSWER KEY
    ==================

    1. B
    2. D
    3. A
    4. C
    5. B
    6. A
    7. D
    8. C
    9. B
    10. A

    Answers must appear ONLY in ANSWER KEY.
    """
)
    return {
            "result": response.content
        }


        
@app.post("/pdf_summarizer")
async def pdf_summarizer(
    file: UploadFile = File(...)
):
    os.makedirs("uploads", exist_ok=True)

    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    pdf_text = "\n".join(
        [doc.page_content for doc in docs]
    )

    response = llm.invoke(
    f"""
    Summarize this PDF.

    PDF Content:
    {pdf_text[:5000]}

    Give:
    1. Summary
    2. Key Points
    3. Important Concepts
    """
)

    return {
    "result": response.content
}

        



    
            