import asyncio
import os
import shutil
from pathlib import Path
import tempfile
import re
from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.prompt import Prompt
from mcp_agent.core.request_params import RequestParams

from lightrag_manager import LightRAGManager

VISION_MODEL = "generic.gemma3:27b"  # Use a larger model for vision tasks
UTILITY_MODEL = "generic.gemma3:27b"  # Use a smaller model for utility tasks
# UTILITY_MODEL = "generic.gemma3n:e2b"  # Use a smaller model for utility tasks
# GENERAL_MODEL = "generic.qwen2.5-coder:7b"  # Use a larger model for general tasks
SUMMARY_MODEL = "generic.gemma3:27b"  # Use a larger model for summarization tasks
# SUMMARY_MODEL = "generic.gemma3n:e4b"  # Use a larger model for summarization tasks
GENERAL_MODEL = "generic.gemma3:27b"  # Use a larger model for general tasks
# GENERAL_MODEL = "generic.llama3.2:latest"  # Use a larger model for general tasks
# GENERAL_MODEL = "generic.llama3.1:8b"  # Use a larger model for general tasks
# EVALUATION_MODEL = "generic.qwen2.5-coder:7b"  # Use a model suitable for evaluation tasks
# REASONING_MODEL = "generic.deepseek-r1:latest"  # Use a model suitable for reasoning tasks
REASONING_MODEL = "generic.gemma3:27b"  # Use a larger model for big tasks

REQUEST_PARAMS = RequestParams(
            maxTokens=32192,
        )


# Create the application
fast = FastAgent("Exam Tutoring Assistant")

@fast.agent(
        "OCR_transcriber",
        instruction=(
            "You are an OCR transcriber. Create a complete transcription of the provided image or "
            "document. Ensure that all text is accurately captured and "
            "formatted in markdown. If the document contains images, tables, "
            "or equations, describe them in detail. Respond with only the "
            "transcription and nothing else."
            "You are transcribing an exam question/answer/explanation that the user didn't understand. Include all relevant information."
            "Include the correct answer as well as which question the user selected."
            "The answer you see highlighted in blue is the answer the user selected. Designate this with the (User Selected) tag."
            "The answer with the check mark is the correct answer. Designate this with the (Correct) tag."
            "The answers with the X are incorrect. Designate these with the (Incorrect) tag."
            "Don't include the blue dot or the checks. just the appropriate tags. "
            "If the user selected the correct answer then label the answer with both tags."
            "Here is an example transcription: \n"
            "<example>\n"
            "Your customer notices that the exchange rate for the British pound in the spot market is listed at 148.47. "
            "What do you tell her when she asks you what this means?\n"
            " * A) $1 equals 14.847 pounds.  (Incorrect)\n"
            " * B) One pound equals $1.4847. (Correct)\n"
            " * C) $1 equals 14.487 pounds. (Incorrect)\n"
            " * D) One pound equals 14.847 cents. (Incorrect) - (User Selected)\n"
            "\n"
            "Explanation: The exchange rate refers to cents per British pound; 148.47 equals $1.4847.\n"
            "LO 9.h\n"
            "</example>"
            ),
        model=VISION_MODEL,  # Use a smaller model for transcription
        # model="generic.gemma3:4b",  # Use a smaller model for transcription
        # model="generic.llama3.2-vision:latest",  # Use a smaller model for transcription
        # model="generic.granite3.2-vision:latest",  # Use a smaller model for transcription
        # servers=["filesystem"]
        request_params=REQUEST_PARAMS
)

# @fast.agent(
#         "proofreader",
#         instruction="Ensure that this transcription is accurate and complete. If there are any errors or missing information, correct them.",
#         model="generic.gemma3:4b",  # Use a smaller model for transcription
#         # servers=["filesystem"]
#         request_params=RequestParams(
#             maxTokens=8192,
#         )
# )

# @fast.agent(
#         "curious",
#         instruction="Formulate a list of questions based on the exam question provided. Use the exam question to guide your search for relevant information. Reply with only the list of questions.",
#         # servers=["raganything"]
# )

# @fast.agent(
#         "knowledge_setup",
#         instruction="You are responsible for setting up the knowledge base. Use the RAG server tools to embed documents from the knowledge directory into the vectorstore.",
#         model="generic.llama3.2:3b",
#         servers=["rag"]
# )

@fast.agent(
        "question_generator",
        instruction="Generate 5 questions to search the knowledge graph and help understand this exam question/answer. Ask clear and concise self-contained questions. Return only the list of questions as a numbered list (1. Question one 2. Question two etc.). Do not respond with any other text. Ask why things are the way they are and what regulations and rules are put in place to ensure/prevent. What are the mechanics of all the variables?",
        model=UTILITY_MODEL,
        # servers=["rag"]
        # servers=["rag"]
        # request_params=REQUEST_PARAMS
)

@fast.agent(
        "follow_up_question_generator",
        instruction="Based on the initial research report, generate 5 additional follow-up questions to search for deeper understanding and identify any knowledge gaps. These questions should explore related concepts, edge cases, or practical applications. Ask clear and concise self-contained questions. Return only the list of questions. Do not respond with any other text. Ask why things are the way they are and what regulations and rules are put in place to ensure/prevent. What are the mechanics of all the variables?",
        model=UTILITY_MODEL,
        request_params=REQUEST_PARAMS
)



# @fast.agent(
#         "searcher",
#         instruction="First check the knowledge graph initialization status, then perform a SINGLE multi query search using the knowledge graph server for all querries provided to find relevant information. If the knowledge graph is not ready, wait and try again. If no knowledge base content is available, use your general knowledge to help explain the concepts. Only perform one tool call per turn for searching. ONLY PERFORM ONE TOOL CALL PER TURN.",
#         model=GENERAL_MODEL,
#         servers=["knowledge_graph"],
#         # servers=["rag"],
#         request_params=RequestParams(
#             maxTokens=15192,    
#         )
# )

@fast.agent(
        "summary_report",
        instruction="Create a report/dossier outlining all relevant information found. Summarize the results of the search queries and explain how they relate to the exam question/answers. Be concise but thorough, ensuring all relevant information is included. All math equations should be in simple plain text or python code blocks, no LaTeX. Think deeply and ensure all information is accurate.",
        # model=SUMMARY_MODEL,
        model=REASONING_MODEL,
        # servers=["rag"]
        # servers=["rag"],
        request_params=REQUEST_PARAMS
)

# @fast.agent(
#         "researcher",
#         instruction="Perform some queries requests and summarize the results to help explain the exam question/answers and use rag to search queries and generate a report. You can ask rag as many questions as you want through the batch tool. Ask complete self contained questions. You must try to use a tool to gain information. Worst case scenario use your own general knowledge. Create a report outlining all relevant information found. Don't respond until you have performed your search queries and generated a report.",
#         model=GENERAL_MODEL,
#         # servers=["rag"]
#         servers=["rag"]
# )

# @fast.chain(
#         "researcher",
#         instruction="Perform some queries requests and summarize the results to help explain the exam question/answers using the rag batch knowledge search tool and generate a report. You can ask rag as many questions as you want through the batch tool. Ask complete self contained questions. You must try to use a tool to gain information. Worst case scenario use your own general knowledge. Create a report outlining all relevant information found. Don't respond until you have performed your search queries and generated a report.",
#         # servers=["rag"]
#         sequence=["question_generator", "searcher", "summary_report"],
# )

# @fast.chain(
#     name="exam_tutor",
#     instruction="You are a helpful AI Agent that tutors students on exam questions. You will be given an exam question, answers, and explanation. Do some research and then provide a detailed explanation or solution to the question. If no knowledge base content is available, use perplexica. Worst case scenario, use your general knowledge to help explain the concepts. Teach the user everything they need to know to understand the question and how to solve it. Be brief but thorough, and ensure the user understands the concepts involved.",
#     sequence=["researcher", "tutoring_assistant"],
# )

@fast.agent(
        "professor",
        instruction="You are teaching a class of financial advisor students and you are teaching a series 7 prep course. Take in an exam question and any available knowledge base content, then provide a comprehensive explanation or solution in the form of a lesson. Your goal is to fill the users knowledge gaps. If no knowledge base content is available, use your general knowledge to help explain the concepts. Teach the user everything they need to know to understand the question and how to solve it. Be brief but thorough, and ensure the user understands the concepts involved. Include a glossary of terms and concepts used in the lesson with a brief definition. The entire lesson should ONLY be in english. All math formulas should be in python code blocks (```python ```), no LaTeX. Do not use any special characters or formatting. Explain why things are the way they are and what regulations and rules are put in place to ensure/prevent. What are the mechanics of all the variables? Respond ONLY in english.",
        # model=SUMMARY_MODEL, #GENERAL_MODEL,
        model=REASONING_MODEL,
        request_params=REQUEST_PARAMS
)

@fast.agent(
        "lecture_scriptwriter",
        # instruction="You are a script writer for an advanced text-to-speech system that supports XML emotion tags. Take a lesson and rewrite it as an engaging lecture script for an AI TTS engine. Most of the content should be delivered in a natural, untagged voice. Use emotion tags SPARINGLY and only for key moments where they genuinely enhance understanding or engagement. Structure it with clear sections that flow naturally when spoken. Make complex concepts easier to understand when heard rather than read.\n\nIMPORTANT: When using emotion tags, use PROPER XML FORMAT. Do NOT use markdown or annotations. \n\nCORRECT FORMAT: <realization>This is a moment of discovery that will change everything.</realization>\nINCORRECT FORMAT: **(A moment of revelation – <realization>)** or **<realization>** or any markdown formatting\n\nAVAILABLE EMOTION TAGS (use sparingly):\nBASIC EMOTIONS: <happy>text</happy>, <sad>text</sad>, <angry>text</angry>, <fearful>text</fearful>, <confused>text</confused>\nEXPRESSIVE STYLES: <narrator>text</narrator>, <whisper>text</whisper>, <sarcastic>text</sarcastic>, <calming>text</calming>, <enunciated>text</enunciated>, <fast>text</fast>, <projected>text</projected>\nCOMPLEX EMOTIONS: <adoration>text</adoration>, <amazement>text</amazement>, <amusement>text</amusement>, <contentment>text</contentment>, <cute>text</cute>, <desire>text</desire>, <disappointed>text</disappointed>, <disgust>text</disgust>, <distress>text</distress>, <embarrassment>text</embarrassment>, <ecstasy>text</ecstasy>, <fear>text</fear>, <guilt>text</guilt>, <interest>text</interest>, <neutral>text</neutral>, <pain>text</pain>, <pride>text</pride>, <realization>text</realization>, <relief>text</relief>, <serenity>text</serenity>\n\nCRITICAL RULES:\n1. USE SPARINGLY: Most content should be untagged. Only use emotion tags for truly impactful moments - perhaps 5-15% of the content at most.\n2. PROPER XML FORMAT: Use <tag>content</tag> format ONLY. Never use markdown, asterisks, or annotations.\n3. COMPLETE SENTENCES ONLY: Only wrap entire complete sentences or full paragraphs in emotion tags. Never tag individual words or partial sentences.\n4. NO MARKDOWN: Do not use any markdown formatting like **, __, or any other special characters.\n5. NATURAL FLOW: When you do use tags, ensure emotion changes feel natural and logical.\n6. STRATEGIC PLACEMENT: Use tags for key concepts, surprising revelations, important warnings, or moments that need emphasis.\n\nWrite primarily in natural, conversational tone with strategic emotional enhancement only where it truly adds value. Use proper XML emotion tags when needed, but keep most content untagged.",
        instruction="Take a lesson and rewrite it as an unformated plaintext lecture script for an AI TTS engine. DO NOT use ANY formatting at all, the TTS engine will read it and it will reduce quality. Write only in plain text and do not use any markdown. Structure it with clear sections that flow naturally when spoken. Make complex concepts easier to understand when heard rather than read. Keep all the educational content but make it more engaging and natural for oral presentation. The user is listening to these lectures independantly so make sure they are self contained. Dont add any prompts, headings or cues. Only write the script as its meant to be read. Don't add any markdown formatting or special characters. Dont include anything that isnt meant to be read out loud by the text to speech engine. Start by introducing the topic of the lecture.",
        # model=SUMMARY_MODEL,
        model=REASONING_MODEL,
        request_params=REQUEST_PARAMS
)


# @fast.chain(
#     name="exam_tutor",
#     instruction="You are a helpful AI Agent that tutors students on exam questions. You will be given an exam question, answers, and explanation. Do some research and then provide a detailed explanation or solution to the question. If no knowledge base content is available, use perplexica. Worst case scenario, use your general knowledge to help explain the concepts. Teach the user everything they need to know to understand the question and how to solve it. Be brief but thorough, and ensure the user understands the concepts involved.",
#     sequence=["researcher", "tutoring_assistant"],
# )

# @fast.orchestrator(
#     name="professor",
#     instruction="You are a passionate tenured professor that tutors students on exam questions. You will be given an exam question, answers, and explanation. First have the researcher do some research and then provide a detailed explanation or solution to the question. Then have lesson_generator teach the user everything they need to know to understand the question and how to solve it. Be brief but thorough, and ensure the user understands the concepts involved.",
#     agents=["lesson_generator", "teachers_assistant"],
#     model=EVALUATION_MODEL,
#     plan_type="iterative",
    
# )

# @fast.agent(
#         "mentor",
#         instruction="You are a helpful AI Agent that ensures that the teaching assistant provides accurate and helpful explanations. Review the tutoring assistant's response and provide feedback or corrections if necessary. Ensure that the explanation is clear, concise, and educational.",
#         model=EVALUATION_MODEL,  # Use a model suitable for mentoring and feedback
# )

# @fast.evaluator_optimizer(
#   name="converter",
#   generator="transcriber",
#   evaluator="proofreader",
#   min_rating="GOOD",
#   max_refinements=3
# )

# @fast.evaluator_optimizer(
#   name="professor",
#   generator="exam_tutor",
#   evaluator="mentor",
#   min_rating="GOOD",
#   max_refinements=3
# )

# @fast.agent(
#         "research_evaluator",
#         instruction="You are a research quality evaluator that assesses the completeness, accuracy, and relevance of research findings. Evaluate the knowledge retrieval results and provide a rating based on the following criteria: 1) Completeness - Does the research cover all important aspects of the topic? 2) Accuracy - Is the information factually correct and reliable? 3) Relevance - Is the information directly related to the query? 4) Depth - Does the research provide sufficient detail for understanding? Rate the research as EXCELLENT (exceeds expectations), GOOD (meets expectations), FAIR (partially meets expectations), or POOR (does not meet expectations). Provide specific feedback on what's missing or could be improved.",
#         model=EVALUATION_MODEL,
# )

# @fast.evaluator_optimizer(
#   name="researcher",
#   instruction="You are a helpful AI Agent that retrieves relevant in    formation from a knowledge base using the RAG server. You are curious and take in content and come up with questions and use your tools to search for answers. You can ask rag as many questions as you want through the batch tool. Ask complete self contained questionsFirst, try to search for educational content related to the exam question using knowledge_search. If no knowledge base is found, inform the user that the knowledge base needs to be set up first. Perplexica is another agent that does research online and returns a report. Prefer using rag over perplexica. Perplexica should only be used to find information that is not in the knowledge base. You mst try to use either tool to gain information. Worst case scenario create a report using your own general knowledge.",
#   generator="fetch_knowledge", 
#   evaluator="research_evaluator",
#   min_rating="GOOD",
#   max_refinements=3
# )

@fast.agent(
        "filename_generator",
        instruction="Generate a filename for this text. Return only the filename without the file extension.",
        model=UTILITY_MODEL,
        request_params=REQUEST_PARAMS
)

def remove_think_tokens(text: str) -> str:
    """
    Remove <think></think> tokens and all text between them from a response.
    
    Args:
        text: The input text that may contain <think></think> tokens
        
    Returns:
        The text with all <think></think> blocks removed
    """
    if not text:
        return text
    
    # Use regex to remove <think>...</think> blocks (including nested ones)
    # The pattern matches <think> followed by any content (including newlines) until </think>
    # Using re.DOTALL flag to make . match newlines as well
    pattern = r'<think>.*?</think>'
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up any extra whitespace that might be left behind
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)  # Replace multiple newlines with double newlines
    cleaned_text = cleaned_text.strip()  # Remove leading/trailing whitespace
    
    return cleaned_text


def parse_questions_to_array(questions_text: str) -> list[str]:
    """
    Parse a numbered list of questions into an array of strings.
    
    Args:
        questions_text: Text containing numbered questions (e.g., "1. Question one\n2. Question two")
        
    Returns:
        List of question strings
    """
    if not questions_text:
        return []
    
    # Split by lines and extract questions
    lines = questions_text.strip().split('\n')
    questions = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Match numbered questions (1. 2. etc.) or bullet points (- * etc.)
        # Remove the numbering/bullet and keep just the question
        question_match = re.match(r'^(?:\d+\.|\*|-|\•)\s*(.+)$', line)
        if question_match:
            questions.append(question_match.group(1).strip())
        elif line and not re.match(r'^\d+\.$', line):  # Skip standalone numbers
            # If it doesn't match the pattern but has content, include it anyway
            questions.append(line.strip())
    
    return questions


async def main():
    print("Connecting to existing database...")
    manager = LightRAGManager()
    await manager.initialize()
    tries = 0
    while tries < 100:
        # use the --model command line switch or agent arguments to change model
        # async with fast.run() as agent:
            # Setup knowledge base first
            # print("Setting up knowledge base...")
            # try:
            #     # Try to get vectorstore info first
            #     vectorstore_info = await agent.knowledge_setup(
            #         "Use get_vectorstore_info to check if knowledge base is loaded"
            #     )
            #     print(f"Vectorstore info: {vectorstore_info}")
                
            #     # If no knowledge base or empty, populate it
            #     if "No vectorstore loaded" in str(vectorstore_info) or "status" not in str(vectorstore_info):
            #         print("Knowledge base not found. Setting up from documents...")
            #         if os.path.exists(KNOWLEDGE_BASE_DIR):
            #             setup_result = await agent.knowledge_setup(
            #                 f"Use embed_documents_from_directory to embed all documents from {KNOWLEDGE_BASE_DIR}"
            #             )
            #             print(f"Knowledge base setup result: {setup_result}")
            #         else:
            #             print(f"Knowledge base directory {KNOWLEDGE_BASE_DIR} not found. Creating it...")
            #             os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)
            #             print("Please add documents to the knowledge base directory and run again.")
            #             return
            # except Exception as e:
            #     print(f"Error setting up knowledge base: {e}")
            
            # Create lessons directory if it doesn't exist
        lessons_dir = "./lessons"
        os.makedirs(lessons_dir, exist_ok=True)
        
        # Create lectures directory if it doesn't exist
        lectures_dir = "./lessons/lectures"
        os.makedirs(lectures_dir, exist_ok=True)
        
        # Create archive directory if it doesn't exist
        archive_dir = "./explain_exam_question_archive"
        os.makedirs(archive_dir, exist_ok=True)
        
        # Cycle through every file in ./explain_exam_question
        explain_dir = "./explain_exam_question"
        if os.path.exists(explain_dir):
            if len(os.listdir(explain_dir)) == 0:
                print("No files found in the explain_exam_question directory. Please add files to transcribe and explain.")
                tries += 1
                await asyncio.sleep(60 * 5)  # Wait before retrying
                continue

            tries = 0
            # Sort files by modification time (youngest to oldest)
            files = os.listdir(explain_dir)
            files.sort(key=lambda x: os.path.getmtime(os.path.join(explain_dir, x)), reverse=True)
            
            for filename in files:
                filepath = os.path.join(explain_dir, filename)
                if os.path.isfile(filepath):
                    async with fast.run() as agent:
                        try:
                            print(f"Processing file: {filename}")
                            transcription = await agent.OCR_transcriber(
                                Prompt.user(
                                    f"Transcribe this entire image of an exam question/answer/explanation I didn't understand: '{filename}'",
                                    Path(filepath)
                                )
                            )
                            
                            # Clean up transcription by removing <think></think> tokens
                            transcription = remove_think_tokens(transcription)

                            # transcription = await transcribe(filepath)
                            if transcription:
                                print("Transcription successful!")
                                # print(transcription)
                            else:
                                print("Transcription failed")
                                continue  # Skip to next file if transcription failed
                            
                            questions_response = await agent.question_generator(
                                f"Generate 5 questions to search the knowledge graph and help understand this exam question/answer: '{transcription}'"
                            )
                            
                            # Clean up questions by removing <think></think> tokens and ensure string type
                            questions_text = remove_think_tokens(str(questions_response))
                            assert isinstance(questions_text, str), "Expected string from remove_think_tokens"
                            
                            # Parse questions into an array
                            questions_array = parse_questions_to_array(questions_text)
                            print(f"Generated {len(questions_array)} questions: {questions_array}")
                            
                            # Query each question individually and collect results
                            individual_reports = []
                            for i, question in enumerate(questions_array, 1):
                                print(f"Querying question {i}: {question}")
                                try:
                                    query = (
                                        "<instructions>"
                                        "You are writing a lesson plan for financial advisor students and you are teaching a series 7 prep course."
                                        "Provide a comprehensive explanation and report to this exam question/answer using the research findings below. Explain why things are the way they are and what regulations and rules are put in place to ensure/prevent."
                                        "Your goal is to fill the user's knowledge gaps and explain all relevant knowledge and related concepts. Be comprehensive, thorough, and accurate."
                                        "Include a glossary of terms and concepts used in the lesson with a brief definition. The entire lesson should ONLY be in english. All math formulas should be in python code blocks (```python ```), no LaTeX. Do not use any special characters or formatting. Explain why things are the way they are and what regulations and rules are put in place to ensure/prevent. What are the mechanics of all the variables? Respond ONLY in english."
                                        "</instructions>\n"
                                    ) + question
                                    report = await manager.query(query, mode="hybrid", stream=False)
                                    individual_reports.append(f"**Question {i}:** {question}\n**Answer:**\n{report}\n")
                                    print(f"Completed query {i}/{len(questions_array)}")
                                except Exception as e:
                                    print(f"Error querying question {i}: {e}")
                                    individual_reports.append(f"**Question {i}:** {question}\n**Answer:**\nNo relevant information found.\n")
                            
                            # Combine all individual reports
                            combined_knowledge = "\n".join(individual_reports)
                            
                            # Create comprehensive report combining all knowledge
                            query = (
                                "You are writing a lesson plan for financial advisor students and you are teaching a series 7 prep course."
                                "Provide a comprehensive explanation and report to this exam question/answer using the research findings below. Explain why things are the way they are and what regulations and rules are put in place to ensure/prevent."
                                "Your goal is to fill the user's knowledge gaps and explain all relevant knowledge and related concepts. Be comprehensive, thorough, and accurate."
                                "If no knowledge base content is available, use your general knowledge to "
                                "help explain the concepts. Teach the user everything they need to know "
                                "to understand the question and how to solve it. "
                                "Be brief but thorough, and ensure the user understands the concepts "
                                "involved. Include a glossary of terms and concepts used in the lesson "
                                "with a brief definition. The entire lesson should ONLY be in english. "
                                "All math equations should be in simple plain text or python code blocks, "
                                "no LaTeX. Do not use any special characters or formatting."
                                "Here is the related context:\n"
                                f"<exam-question>\n{transcription}\n</exam-question>\n"
                                f"<research-findings>\n{combined_knowledge}\n</research-findings>"
                            )
                            report = await manager.query(query, mode="hybrid", stream=False) #, only_need_prompt=True)
                            # report_prompt = query + "\n" + report_prompt
                            # print(f"Generated comprehensive report: \n{report_prompt}")
                            # explanation_raw = await manager.query(query, mode="hybrid", stream=False)
                            # print(f"Context Retrieved: {context}")
                            # report1 = await agent.summary_report(
                            #     f"Create a comprehensive detailed report that combines and synthesizes the information from your knowledge base to create a detailed and comprehensive report on . "
                            #     f"Ensure all relevant information is included and well-organized for understanding the exam question. Don't mention either report. Create a report containing all relevant information. Focus on being informative, clear, and accurate.\n"
                            #     "Provide a comprehensive explanation and report to this exam question/answer and these other related questions. Explain why things are the way they are and what reulations and rules are put in place to ensure/prevent."
                            #     "Your goal is to fill the user's knowledge gaps and explain all relevant knowledge and related concepts. Be comprehensive, thorough, and accurate."
                            #     "If no knowledge base content is available, use your general knowledge to "
                            #     "help explain the concepts. Teach the user everything they need to know "
                            #     "to understand the question and how to solve it. "
                            #     "Be brief but thorough, and ensure the user understands the concepts "
                            #     "involved. Include a glossary of terms and concepts used in the lesson "
                            #     "with a brief definition. The entire lesson should ONLY be in english. "
                            #     "All math equations should be in simple plain text or python code blocks, "
                            #     "no LaTeX. Do not use any special characters or formatting."
                            #     "Here is the related context:\n"
                            #     f"<exam-question>\n{transcription}\n</exam-question>\n"
                            #     f"<search-queries>\n{questions}\n</search-queries>\n"
                            #     f"<knowledge-base>\n{context}\n</knowledge-base>\n"
                            # )
                            # Generate follow-up questions based on the first report
                            # follow_up_questions = await agent.follow_up_question_generator(
                            #     f"Based on this initial research report, generate 5 additional follow-up questions to explore deeper understanding:\n<initial-report>\n{report1}\n</initial-report>\n<exam-question>\n{transcription}\n</exam-question>"
                            # )
                            
                            # Clean up follow-up questions by removing <think></think> tokens
                            # follow_up_questions = remove_think_tokens(follow_up_questions)
                            # print(f"Generated follow-up questions: {follow_up_questions}")
                            
                            # Generate second report based on follow-up questions
                            # follow_up_query = (
                            #     "Is this initial report missing anything? Add any relevant information and analysis you have to add. "
                            #     "Is this report accurate? Fact check the information in the report and ensure it is correct. "
                            #     "Generate a comprehensive detailed report that combines and synthesizes the information from your knowledge base to create a detailed and comprehensive report on the exam question, and the general topics surrounding the exam question. "
                            #     "Focus on deeper concepts, edge cases, practical applications, and any knowledge gaps from the initial report. "
                            #     "Be thorough and accurate in your explanations."
                            #     "Here is the context:\n"
                            #     f"<exam-question>\n{transcription}\n</exam-question>\n"
                            #     f"<initial-report>\n{report1}\n</initial-report>\n"
                            #     f"<follow-up-questions>\n{follow_up_questions}\n</follow-up-questions>"
                            # )
                            # report2 = await manager.query(follow_up_query, mode="hybrid", stream=False)
                            # print(f"Generated second report: {report2}")
                            # report2 = await agent.summary_report(
                            #     f"Create a comprehensive detailed report that combines and synthesizes the information from your knowledge base to create a detailed and comprehensive report on . "
                            #     f"Ensure all relevant information is included and well-organized for understanding the exam question. Don't mention either report. Create a report containing all relevant information. Focus on being informative, clear, and accurate.\n"
                            #     "Provide a comprehensive explanation and report to this exam question/answer and these other related questions. Explain why things are the way they are and what reulations and rules are put in place to ensure/prevent."
                            #     "Your goal is to fill the user's knowledge gaps and explain all relevant knowledge and related concepts. Be comprehensive, thorough, and accurate."
                            #     "If no knowledge base content is available, use your general knowledge to "
                            #     "help explain the concepts. Teach the user everything they need to know "
                            #     "to understand the question and how to solve it. "
                            #     "Be brief but thorough, and ensure the user understands the concepts "
                            #     "involved. Include a glossary of terms and concepts used in the lesson "
                            #     "with a brief definition. The entire lesson should ONLY be in english. "
                            #     "All math equations should be in simple plain text or python code blocks, "
                            #     "no LaTeX. Do not use any special characters or formatting."
                            #     "Here is the related context:\n"
                            #     f"<exam-question>\n{transcription}\n</exam-question>\n"
                            #     f"<search-queries>\n{follow_up_questions}\n</search-queries>\n"
                            #     f"<knowledge-base>\n{context}\n</knowledge-base>\n"
                            # )
                            
                            # Summarize both reports using the summary_report agent
                            # combined_report = await agent.summary_report(
                            #     f"Create a comprehensive detailed report that combines and synthesizes the information from both research reports. "
                            #     f"Ensure all relevant information is included and well-organized for understanding the exam question. Don't mention either report. Just create one self-contained final report containing all relevant information. Focus on being informative, clear, and accurate.\n"
                            #     f"<exam-question>\n{transcription}\n</exam-question>\n"
                            #     f"<first-report>\n{report1}\n</first-report>\n"
                            #     f"<second-report>\n{report2}\n</second-report>"
                            # )
                            
                            # Clean up combined report by removing <think></think> tokens
                            # combined_report = remove_think_tokens(combined_report)
                            # print(f"Generated combined report: {combined_report}")
                            
                            # explanation = await manager.query(query, mode="hybrid", stream=False)
                            # explanation_raw = await agent.summary_report(
                            #     f"You are writing a lesson plan for financial advisor students and you are teaching a series 7 prep course."
                            #     f"Here is an exam question/answer. A assistant has collected some relevant information. Create a thorough and comprehensive explanation including all the provided information to the students so that they will be prepared for this type of question in the next exam."
                            #     f"Create a comprehensive detailed lesson that combines and synthesizes the information from both research reports. "
                            #     f"Ensure all relevant information is included and well-organized for understanding the exam question. Don't mention either report. Just create one self-contained final report containing all relevant information. Focus on being informative, clear, and accurate.\n"
                            #     f"Create a thorough and comprehensive explanation to the user so "
                            #     f"that they will be prepared for this type of question in the next exam:\n"
                            #     f"<research-findings>\n{combined_knowledge}\n\n{report1}\n</research-findings>"
                            #     f"<exam-question>\n{transcription}\n</exam-question>"
                            # )
                            # explanation_raw = await agent.professor(
                            #     report_prompt
                            # )
                            explanation = (
                                f"{report}\n\n\n## FAQs:\n\n{combined_knowledge}"
                            )
                            # Clean up explanation by removing <think></think> tokens
                            # explanation = remove_think_tokens(str(explanation_raw))
                            assert isinstance(explanation, str), "Expected string from remove_think_tokens"
                            
                            # Generate lecture script from the lesson
                            # lecture_script_raw = await agent.lecture_scriptwriter(
                            #     f"Convert this lesson into an engaging script that would be perfect for reading aloud to students. Be totally self contained and explain all of the relevant information. Here is the lesson:\n\n{explanation}"
                            # )
                            
                            # # Clean up lecture script by removing <think></think> tokens
                            # lecture_script = remove_think_tokens(str(lecture_script_raw))
                            # assert isinstance(lecture_script, str), "Expected string from remove_think_tokens"
                            
                            # Generate lesson filename with retry logic
                            lesson_filename = None
                            max_filename_retries = 3
                            for attempt in range(max_filename_retries):
                                try:
                                    lesson_filename_raw = await agent.filename_generator(
                                        f"Generate a suitable lesson filename for the explanation (dont include the file extensions) reply with only the filename and nothing else: '{explanation[:500]}...'"
                                    )
                                    
                                    # Clean up filename by removing <think></think> tokens
                                    lesson_filename = remove_think_tokens(str(lesson_filename_raw))
                                    assert isinstance(lesson_filename, str), "Expected string from remove_think_tokens"
                                    lesson_filename = lesson_filename.replace(" ", "_")
                                    lesson_filename = lesson_filename.strip()
                                    
                                    # Validate filename is not empty and doesn't contain invalid characters
                                    if lesson_filename and len(lesson_filename) > 0:
                                        # Remove invalid filename characters
                                        lesson_filename = re.sub(r'[<>:"/\\|?*]', '_', lesson_filename)
                                        lesson_filename = lesson_filename[:100]  # Limit length
                                        break
                                    else:
                                        raise ValueError("Generated filename is empty")
                                        
                                except Exception as e:
                                    print(f"Filename generation attempt {attempt + 1} failed: {e}")
                                    if attempt == max_filename_retries - 1:
                                        # Final fallback: use timestamp and original filename
                                        import datetime
                                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                        base_filename = os.path.splitext(filename)[0]
                                        lesson_filename = f"{base_filename}_{timestamp}"
                                        print(f"Using fallback filename: {lesson_filename}")
                            
                            # Ensure lesson_filename is not None
                            if lesson_filename is None:
                                import datetime
                                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                base_filename = os.path.splitext(filename)[0]
                                lesson_filename = f"{base_filename}_{timestamp}"
                                print(f"Using final fallback filename: {lesson_filename}")
                            
                            # Ensure it's a string for type checker
                            assert isinstance(lesson_filename, str), "lesson_filename must be a string"
                            
                            lesson_filename = lesson_filename + "_lesson.md"
                            lecture_filename = lesson_filename.replace("_lesson.md", "_lecture.md")
                            
                            # Save explanation to markdown file
                            lesson_filepath = os.path.join(lessons_dir, lesson_filename)
                            with open(lesson_filepath, 'w', encoding='utf-8') as lesson_file:
                                lesson_file.write(str(transcription) + "\n\n")
                                lesson_file.write(explanation)
                            
                            # Save lecture script to markdown file
                            # lecture_filepath = os.path.join(lectures_dir, lecture_filename)
                            # with open(lecture_filepath, 'w', encoding='utf-8') as lecture_file:
                            #     # lecture_file.write("# Lecture Script\n\n")
                            #     # lecture_file.write(f"**Original Question:**\n{transcription}\n\n")
                            #     # lecture_file.write("**Lecture Script:**\n\n")
                            #     lecture_file.write(lecture_script.replace("*", ""))
                            
                            print(f"Completed processing: {filename} -> {lesson_filename} and {lecture_filename}\n")
                            
                            # Archive the processed file with the same name as the lesson file
                            file_extension = os.path.splitext(filename)[1]
                            archived_filename = lesson_filename.replace("_lesson.md", file_extension)
                            archive_filepath = os.path.join(archive_dir, archived_filename)
                            shutil.move(filepath, archive_filepath)
                            print(f"Archived file: {filename} -> {archived_filename}")
                            
                        except Exception as e:
                            print(f"Error processing file {filename}: {e}")
        
        # Example usage of the researcher evaluator optimizer
        # You can call it like this to get evaluated research results:
        # research_result = await agent.researcher("Research question here")
        
        # await agent.interactive()






async def transcribe(image_path: str) -> str:
    """
    Convert an image file to Markdown format using MinerU
    
    Args:
        image_path: Path to the image file to process
        
    Returns:
        String containing the markdown content, or empty string if processing fails
    """
    try:
        # Import MinerU functions locally to avoid global import issues
        from mineru.cli.common import prepare_env, read_fn
        from mineru.data.data_reader_writer import FileBasedDataWriter
        from mineru.utils.enum_class import MakeMode
        from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
        from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
        from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
        
        # Set environment variables to force CPU usage
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        # Validate image file exists
        if not os.path.exists(image_path):
            print(f"Image file does not exist: {image_path}")
            return ""
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Prepare the output directory and file name
            image_file_name = Path(image_path).stem
            local_image_dir, local_md_dir = prepare_env(temp_dir, image_file_name, "auto")
            
            # Read image file
            image_bytes = read_fn(image_path)
            if not image_bytes:
                print(f"Failed to read image file: {image_path}")
                return ""
            
            # Process image using MinerU pipeline
            image_bytes_list = [image_bytes]
            p_lang_list = ["en"]  # Default to English, can be made configurable
            
            # Analyze document with CPU-only mode
            print(f"Processing image with CPU-only mode: {image_path}")
            infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(
                image_bytes_list, p_lang_list, parse_method="auto", formula_enable=True, table_enable=True
            )
            
            if not infer_results:
                print(f"Failed to analyze image: {image_path}")
                return ""
            
            # Process the first (and only) document
            model_list = infer_results[0]
            images_list = all_image_lists[0]
            pdf_doc = all_pdf_docs[0]
            _lang = lang_list[0]
            _ocr_enable = ocr_enabled_list[0]
            
            # Create data writers
            image_writer = FileBasedDataWriter(local_image_dir)
            
            # Convert to middle JSON format
            middle_json = pipeline_result_to_middle_json(
                model_list, images_list, pdf_doc, image_writer, _lang, _ocr_enable, True
            )
            
            pdf_info = middle_json["pdf_info"]
            
            # Generate markdown content
            image_dir = str(os.path.basename(local_image_dir))
            markdown_content = pipeline_union_make(pdf_info, MakeMode.MM_MD, image_dir)
            
            # Ensure we return a string
            if isinstance(markdown_content, str):
                return markdown_content
            elif isinstance(markdown_content, list):
                return "\n".join(str(item) for item in markdown_content)
            else:
                return str(markdown_content) if markdown_content else ""
            
    except Exception as e:
        print(f"Error converting image to Markdown using MinerU: {e}")
        import traceback
        traceback.print_exc()
        return ""


if __name__ == "__main__":
    # Uncomment the line below to run the research example instead of the main exam processing
    # asyncio.run(run_research_example())
    asyncio.run(main())