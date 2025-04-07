import openai
from dotenv import load_dotenv
import os
import streamlit as st
from openai import OpenAI, beta
import pandas as pd
import sys
import io
from pydantic import BaseModel, Field
import json
from IPython.display import Markdown as md
import streamlit
#%matplotlib inline

# structured response
class PandasCode(BaseModel):
    code: str = Field(description="Pandas code to be implemented")
    explanation: str = Field(description="String explaining the code")

class SynthesizeFinalAnswer(BaseModel):
    original_question: str = Field(description="The original question asked by the user")
    final_answer: str = Field(description="The final answer synthesized by the model")
    analysis: str = Field(description="Additional analysis or insights as a result of the final answer")
    method: str = Field(description="The method used to synthesize the final answer")

def generate_pandas_code(nl_instruction=None, df_preview=None, client=None,model=None):
    """
    Given a natural language instruction and a preview of the DataFrame,
    call OpenAI to generate Python code that operates on an existing DataFrame (df)
    and assigns the final output to a variable named 'result'.
    """

    # prompting
    system_prompt = """You are an expert Python programmer. "
                "Your response must contain only valid Python code that uses pandas. "
                "Assume a pandas DataFrame called 'df' already exists. "
                "IMPORTANT: Your code must end with an assignment to a variable named 'result'. "
                "Do not include any markdown formatting, explanations, or extra text."""
                
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": (
                f"Below is a preview of a pandas DataFrame:\n{df_preview}\n\n"
                f"Write Python code to accomplish the following task:\n{nl_instruction}"
            )
        }
    ]

    response = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=8000,
        response_format = PandasCode
    )
    response_string = response.choices[0].message.content.strip()
    response_json = json.loads(response_string)
    code = response_json['code']
    explanation = response_json['explanation']
    return code

def correct_code(code_str=None, nl_instruction=None, df_preview=None, client=None, model=None, error_message=""):
    """
    Send the error message along with the original code and instruction to OpenAI
    to generate a corrected version of the code.
    """
    system_prompt = f"""You are an expert Python programmer. 
                The following code produced an error during execution. 
                Your task is to correct the code so that it runs successfully. 
                The code should use pandas and assume a DataFrame named 'df' exists. 
                Ensure the final corrected code assigns its output to a variable named 'result'.
                Verify the code answers the following task or question: {nl_instruction}, modifying the code as needed.
                Output only the corrected Python code, with no explanations."""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": (
                f"Original code:\n{code_str}\n\n"
                f"Error encountered:\n{error_message}\n\n"
                f"DataFrame preview:\n{df_preview}\n\n"
                f"Task description:\n{nl_instruction}"
            )
        }
    ]

    response = client.beta.chat.completions.parse(
        model=MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=8000,
        response_format = PandasCode
    )
    response_string = response.choices[0].message.content.strip()
    response_json = json.loads(response_string)
    corrected_code = response_json['code']
    explanation = response_json['explanation']
    return corrected_code

def execute_generated_code(code_str=None, exec_env=None):
    """
    Execute the provided Python code in the provided environment (a dict) and
    return the value of the variable 'result', if defined.
    """
    try:
        corrected_code = code_str
        exec(corrected_code, exec_env)
    except Exception as e:
        print(f"Error during code execution: {e}")
        print(f"Code that failed: {corrected_code}")
        print(f"Execution environment: {exec_env}")
        print(f"Attempting to correct the code...")
        corrected_code = correct_code(corrected_code, exec_env.get("task_description", ""), exec_env.get("df_preview", ""), str(e))
        print(f"Corrected code: {corrected_code}")
        try:
            exec(corrected_code, exec_env)
        except Exception as e:
            print(f"Error during code execution: {e}")
            print(f"Code that failed: {corrected_code}")
            print(f"Execution environment: {exec_env}")
            print(f"Attempting to correct the code...")
            corrected_code = correct_code(corrected_code, exec_env.get("task_description", ""), exec_env.get("df_preview", ""), str(e))
            print(f"Corrected code: {corrected_code}")
            try:
                exec(corrected_code, exec_env)
            except Exception as e:
                print(f"Error during code execution: {e}")
                print(f"Code that failed: {corrected_code}")
                print(f"Execution environment: {exec_env}")
                print(f"Attempting to correct the code...")
                corrected_code = correct_code(code_str, exec_env.get("task_description", ""), exec_env.get("df_preview", ""), str(e))
                print(f"Corrected code: {corrected_code}")
                try:
                    exec(corrected_code, exec_env)
                except Exception as e:
                    print(f"Error during code execution: {e}")
                    return None, str(e)
    return exec_env.get("result", None), corrected_code

def synthesize_final_answer(original_question=None, execution_result=None, client = None, model = None):
    """
    Call OpenAI to synthesize a final answer based on the original question
    and the result produced by the executed code.
    """
    system_prompt = """You are an expert data analyst. "
                "Based on the following execution result from a pandas operation and the original question, "
                "provide a clear and concise final answer.
                "Expand upon the answer to draw out insights or implications as needed, but keep it relevant to the original question."""
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": (
                f"Original question: {original_question}\n\n"
                f"Execution result: {execution_result}"
            )
        }
    ]
    response = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=4000,
        response_format = SynthesizeFinalAnswer
    )
    
    response_string = response.choices[0].message.content.strip()
    response_json = json.loads(response_string)
    question = response_json['original_question']
    answer = response_json['final_answer']
    analysis = response_json['analysis']
    method = response_json['method']
    return question, answer, analysis ,method

import csv
from datetime import datetime

def log_to_csv(log_file, data):
    """
    Logs the provided data into a CSV file.

    Args:
        log_file (str): Path to the CSV log file.
        data (dict): Dictionary containing the data to log.
    """
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

# def process_user_instruction(instruction=None, csv_file_path=None, client=None, model=None):
#     """
#     Processes a natural language instruction to generate, correct, and execute pandas code,
#     and synthesizes the final answer.

#     Args:
#         instruction (str): The natural language instruction from the user.
#         csv_file_path (str): The file path to the CSV file.

#     Returns:
#         dict: A dictionary containing the question, answer, analysis, method, final code, and visualization (if any).
#     """
#     # Load the CSV file
#     df = pd.read_csv(csv_file_path)

#     # Preview the DataFrame
#     df_preview = df.head().to_string()

#     # Generate pandas code based on the instruction and preview
#     code = generate_pandas_code(nl_instruction=instruction, df_preview=df_preview, client=client, model=model)

#     # Correct the generated code
#     corrected_code = correct_code(code_str=code, nl_instruction=instruction, df_preview=df_preview, client=client, model=model, error_message="")

#     # Execute the corrected code
#     exec_env = {"df": df}
#     execution_result, final_code = execute_generated_code(code_str=corrected_code, exec_env=exec_env)

#     # Check for visualization (e.g., Matplotlib figure)
#     visualization = None
#     if "plt" in exec_env:  # Check if Matplotlib is used
#         import matplotlib.pyplot as plt
#         visualization = plt.gcf()  # Get the current figure

#     # Synthesize the final answer
#     question, answer, analysis, method = synthesize_final_answer(
#         original_question=instruction, execution_result=execution_result, client=client, model=model
#     )

#     # Return the results
#     return {
#         "Question": question,
#         "Answer": answer,
#         "Analysis": analysis,
#         "Method": method,
#         "Code": final_code,
#         "Execution_Result": execution_result,
#         "Visualization": visualization,
#     }

def process_user_instruction(instruction=None, csv_file_path=None, client=None, model=None):
    """
    Processes a natural language instruction to generate, correct, and execute pandas code,
    and synthesizes the final answer. Logs the process into a CSV file.

    Args:
        instruction (str): The natural language instruction from the user.
        csv_file_path (str): The file path to the CSV file.

    Returns:
        dict: A dictionary containing the question, answer, analysis, method, final code, and visualization (if any).
    """
    log_file = "llm_analysis_log.csv" 
    log_data = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Original_Question": instruction,
        "CSV_File_Path": csv_file_path,
        "Final_Answer": None,
        "Analysis": None,
        "Method": None,
        "Final_Code": None,
        "Execution_Result": None,
        "Error": None
    }
    visualization = None
    try:
        df = pd.read_csv(csv_file_path)

        df_preview = df.head().to_string()

        code = generate_pandas_code(nl_instruction=instruction, df_preview=df_preview, client=client, model=model)

        corrected_code = correct_code(code_str=code, nl_instruction=instruction, df_preview=df_preview, client=client, model=model, error_message="")

        exec_env = {"df": df}
        execution_result, final_code = execute_generated_code(code_str=corrected_code, exec_env=exec_env)
        
        if "plt" in exec_env:
            import matplotlib.pyplot as plt
            visualization = plt.gcf()

        question, answer, analysis, method = synthesize_final_answer(
            original_question=instruction, execution_result=execution_result, client=client, model=model
        )

        log_data.update({
            "Final_Answer": answer,
            "Analysis": analysis,
            "Method": method,
            "Final_Code": final_code,
            "Execution_Result": execution_result
        })

    except Exception as e:
        log_data["Error"] = str(e)

    log_to_csv(log_file, log_data)

    return {
        "Question": log_data["Original_Question"],
        "Answer": log_data["Final_Answer"],
        "Analysis": log_data["Analysis"],
        "Method": log_data["Method"],
        "Code": log_data["Final_Code"],
        "Execution_Result": log_data["Execution_Result"],
        "Visualization": visualization,
    }

def llm_analysis_app():
    st.title("LLM Augmented Analysis")

    instruction = st.text_area("Enter your instruction:")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None and instruction.strip():
        if st.button("Process Instruction"):
            result = process_user_instruction(
                instruction=instruction, 
                csv_file_path=uploaded_file, 
                client=CLIENT, 
                model=MODEL
            )

            if result:
                st.divider()
                st.header("Results")
                
                st.subheader("Question")
                st.write(result.get("Question", "No question generated."))

                st.subheader("Answer")
                st.write(result.get("Answer", "No answer generated."))

                st.subheader("Analysis")
                st.write(result.get("Analysis", "No analysis provided."))

                st.subheader("Method")
                st.write(result.get("Method", "No method provided."))

                st.subheader("Code")
                st.code(result.get("Code", "No code generated."), language="python")

                st.subheader("Execution Result")
                st.write(result.get("Execution_Result", "No execution result available."))

                if result.get("Visualization"):
                    st.subheader("Visualization")
                    st.pyplot(result["Visualization"])
            else:
                st.error("No results were generated. Please check your input.")
        else:
            st.info("Please provide an instruction and upload a CSV file.")

load_dotenv()
MODEL = os.getenv('MODEL_CHOICE', 'gpt-4o-mini')
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
CLIENT = OpenAI(api_key=OPENAI_API_KEY)
llm_analysis_app()