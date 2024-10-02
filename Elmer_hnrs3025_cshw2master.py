import argparse
import json
import os
import re
import time

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from the .env file
load_dotenv('.env')


def four_o_mini_answers():
    """
    Generate answers using the 4o-mini model via OpenAI's batch API.

    This function reads questions from 'dev-v2.0.json', formats them into batch tasks,
    submits the batch request to OpenAI, waits for completion, and saves the responses.
    """
    # Retrieve and print the OpenAI API key for debugging purposes
    print(os.getenv("OPENAI_API_KEY"))

    # Load the dataset containing questions
    with open('dev-v2.0.json') as f:
        data = json.load(f)

    # Extract up to 5000 questions from the dataset
    questions = []
    for article in data['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                question = qa['question']
                questions.append(question)
                if len(questions) >= 5000:
                    break
            if len(questions) >= 5000:
                break
        if len(questions) >= 5000:
            break

    # Define system and user prompts
    system_prompt = (
        "You are an intelligent AI Model. Answer the question, and make sure that your response "
        "is short and concise. Prioritize accuracy above all else."
    )
    user_prompt = (
        "Answer the question concisely and factually. Prioritize accuracy above all else: {question}"
    )

    tasks = []
    for idx, question in enumerate(questions, 1):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(question=question)}
        ]
        custom_id = f"{idx}. {question}"

        task = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "temperature": 0.2,
                "messages": messages
            }
        }
        tasks.append(task)

    # Write the batch tasks to a JSONL file
    with open("input_batch.jsonl", "w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")

    # Initialize the OpenAI client with the API key
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Upload the batch file to OpenAI
    batch_file = client.files.create(
        file=open("input_batch.jsonl", 'rb'),
        purpose='batch'
    )
    print(batch_file)

    # Create a batch job for processing
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    # Poll the batch job status until completion
    complete = False
    while not complete:
        check = client.batches.retrieve(batch_job.id)
        print(f'Status: {check.status}')
        if check.status == 'completed':
            complete = True
        time.sleep(1)

    print("Done processing batch.")
    print(batch_job)
    print("Writing data...")
    print(check)

    # Retrieve and save the batch job results
    result = client.files.content(check.output_file_id).content
    output_file_name = "40_mini_output_batch.jsonl"
    with open(output_file_name, 'wb') as file:
        file.write(result)

    # Load and print the results
    results = []
    with open(output_file_name, 'r') as file:
        for line in file:
            json_object = json.loads(line.strip())
            results.append(json_object)


def llama_answers():
    """
    Generate answers using the Llama model via Azure's ChatCompletionsClient.

    This function reads questions from 'dev-v2.0.json', submits each question to the Llama model,
    and appends the responses to 'llama8Boutput.json'.
    """
    # Initialize the Azure ChatCompletionsClient with endpoint and credentials
    client = ChatCompletionsClient(
        endpoint=os.environ["AZURE_MLSTUDIO_ENDPOINT"],
        credential=AzureKeyCredential(os.environ["AZURE_MLSTUDIO_KEY"]),
    )

    questions = []
    # Load questions from the dataset
    with open('dev-v2.0.json', 'r') as file:
        data = json.load(file)
        for article in data['data']:
            for paragraph in article['paragraphs']:
                for qa in paragraph['qas']:
                    if len(questions) >= 5000:
                        break
                    questions.append(qa['question'])

    # Open the output file in append mode
    with open('llama8Boutput.json', 'a') as output_file:
        for i, question in enumerate(questions, 1):
            # Submit the question to the Llama model
            response = client.complete(
                messages=[
                    SystemMessage(
                        content=(
                            "You are an intelligent AI Model. Answer the question, and make sure that "
                            "your response is short and concise. Prioritize accuracy above all else."
                        )
                    ),
                    UserMessage(content=question),
                ],
                timeout=900  # Timeout set to 15 minutes
            )
            # Structure the result
            result = {
                "question": question,
                "response": response.choices[0].message.content,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens
            }

            output_file.write(json.dumps(result) + '\n')


def four_o_mini_grading():
    """
    Grade the 4o-mini model's answers using OpenAI's batch API.

    This function reads the correct answers and 4o-mini's responses, formats grading tasks,
    submits them as a batch to OpenAI, and saves the grading results.
    """
    # Retrieve and print the OpenAI API key for debugging purposes
    print(os.getenv("OPENAI_API_KEY"))

    # Load the dataset containing correct answers
    with open('dev-v2.0.json') as f:
        correct_data = json.load(f)

    # Load 4o-mini's responses from the batch output file
    with open('40_mini_output_batch.jsonl') as f:
        gpt_data = [json.loads(line) for line in f]

    correct_answers = []
    gpt_answers = []

    # Extract correct answers from the dataset
    for article in correct_data['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                if qa['answers']:
                    answer = qa['answers'][0]['text'].lower()
                    correct_answers.append(answer)
                else:
                    correct_answers.append(None)
                if len(correct_answers) >= 5000:
                    break
            if len(correct_answers) >= 5000:
                break
        if len(correct_answers) >= 5000:
            break

    # Extract questions and 4o-mini's responses
    for entry in gpt_data:
        question = entry["custom_id"]
        response_content = entry["response"]["body"]["choices"][0]["message"]["content"]

        gpt_answers.append({
            "question": question,
            "response": response_content
        })

    # Define prompts for grading
    system_prompt = (
        "You are a teacher tasked with determining whether a student's answer to a question was "
        "correct, based on a set of possible correct answers. You must only use the provided possible "
        "correct answers to determine if the student's response was correct."
    )
    user_prompt = """Question: {question})
    Student's Response: {student_response}
    Possible Correct Answers: {correct_answers}
    Your response should be a valid JSON in the following format:
    {{
    "explanation": "A short explanation of why the student's answer was correct or incorrect.",
    "score": true or false (boolean)
    }}"""

    tasks = []

    # Create grading tasks for each question-response pair
    for idx, (correct_answer, gpt_answer) in enumerate(zip(correct_answers, gpt_answers), 1):
        if correct_answer is None:
            continue
        question = gpt_answer['question']
        student_response = gpt_answer['response']

        formatted_user_prompt = user_prompt.format(
            question=question,
            student_response=student_response,
            correct_answers=correct_answer
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_user_prompt}
        ]

        custom_id = f"{idx}. {question}"

        task = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "temperature": 0.2,
                "messages": messages
            }
        }

        tasks.append(task)

    # Write grading tasks to a JSONL file
    with open("input_batch.jsonl", "w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")

    # Initialize the OpenAI client with the API key
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Upload the batch file to OpenAI
    batch_file = client.files.create(
        file=open("input_batch.jsonl", 'rb'),
        purpose='batch'
    )
    print(batch_file)

    # Create a batch job for grading
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    # Poll the batch job status until completion
    complete = False
    while not complete:
        check = client.batches.retrieve(batch_job.id)
        print(f'Status: {check.status}')
        if check.status == 'completed':
            complete = True
        time.sleep(1)

    print("Done processing batch.")
    print(batch_job)
    print("Writing data...")
    print(check)

    # Retrieve and save the grading results
    result = client.files.content(check.output_file_id).content
    output_file_name = "4o_grading_results.jsonl"
    with open(output_file_name, 'wb') as file:
        file.write(result)

    # Load and print the grading results
    results = []
    with open(output_file_name, 'r') as file:
        for line in file:
            json_object = json.loads(line.strip())
            results.append(json_object)


def llama_grading():
    """
    Grade the Llama model's answers using OpenAI's batch API.

    This function reads the correct answers and Llama's responses, formats grading tasks,
    submits them as a batch to OpenAI, and saves the grading results.
    """
    # Retrieve the OpenAI API key from environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Load the dataset containing correct answers
    with open('dev-v2.0.json') as f:
        correct_data = json.load(f)

    # Load Llama's responses from the output file
    with open('llama8boutput.json') as f:
        llama_data = json.load(f)

    correct_answers = []
    gpt_answers = []

    # Extract correct answers from the dataset
    for article in correct_data['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                if qa['answers']:
                    answer = qa['answers'][0]['text'].lower()
                    correct_answers.append(answer)
                else:
                    correct_answers.append(None)
                if len(correct_answers) >= 5000:
                    break
            if len(correct_answers) >= 5000:
                break
        if len(correct_answers) >= 5000:
            break

    # Extract questions and Llama's responses
    for entry in llama_data:
        question = entry["question"]
        response_content = entry["response"]
        gpt_answers.append({
            "question": question,
            "response": response_content
        })

    # Define prompts for grading
    system_prompt = (
        "You are a teacher tasked with determining whether a student's answer to a question was "
        "correct, based on a set of possible correct answers."
    )
    user_prompt = """Question: {question})
    Student's Response: {student_response}
    Possible Correct Answers: {correct_answers}
    Your response should be a valid JSON in the following format:
    {{
    "explanation": "A short explanation of why the student's answer was correct or incorrect.",
    "score": true or false (boolean)
    }}"""

    tasks = []

    # Create grading tasks for each question-response pair
    for idx, (correct_answer, gpt_answer) in enumerate(zip(correct_answers, gpt_answers), 1):
        if correct_answer is None:
            continue
        question = gpt_answer['question']
        student_response = gpt_answer['response']

        formatted_user_prompt = user_prompt.format(
            question=question,
            student_response=student_response,
            correct_answers=correct_answer
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_user_prompt}
        ]

        custom_id = f"{idx}. {question}"

        # Define the grading task
        task = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "temperature": 0.2,
                "messages": messages
            }
        }

        tasks.append(task)

    # Write grading tasks to a JSONL file
    with open("input_batch.jsonl", "w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")

    # Initialize the OpenAI client with the API key
    client = OpenAI(api_key=openai_api_key)

    # Upload the batch file to OpenAI
    batch_file = client.files.create(
        file=open("input_batch.jsonl", 'rb'),
        purpose='batch'
    )

    # Create a batch job for grading
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    # Poll the batch job status until completion
    complete = False
    while not complete:
        check = client.batches.retrieve(batch_job.id)
        print(f'Status: {check.status}')
        if check.status == 'completed':
            complete = True
        time.sleep(1)

    print("Batch processing completed.")

    # Retrieve and save the grading results
    result = client.files.content(check.output_file_id).content
    output_file_name = "llama_grading_results.jsonl"
    with open(output_file_name, 'wb') as file:
        file.write(result)


def fix_invalid_escapes(json_string):
    """
    Fix invalid escape sequences in a JSON string.

    Args:
        json_string (str): The JSON string with potential invalid escape sequences.

    Returns:
        str: The corrected JSON string with valid escape sequences.
    """
    # Replace backslashes not followed by valid escape characters with double backslashes
    return re.sub(r'\\(?!["\\/bfnrt])', r'\\\\', json_string)


def calc_avgs(file_path):
    """
    Calculate the average score from a grading JSONL file.

    This function reads each line of the JSONL file, extracts the 'score' field,
    and computes the average percentage of correct scores.

    Args:
        file_path (str): Path to the grading JSONL file.

    Returns:
        float: The average score as a decimal (e.g., 0.85 for 85%).
    """
    total_score = 0
    total_responses = 0

    with open(file_path, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in the line: {e}")
                continue

            try:
                response_content = entry["response"]["body"]["choices"][0]["message"]["content"]
            except (KeyError, IndexError) as e:
                print(f"Error extracting response content: {e}")
                continue

            if not response_content.strip():
                print("Empty response content, skipping this entry.")
                continue

            # Strip any code block markers (```json and ```)
            if response_content.startswith("```json"):
                response_content = response_content.replace("```json", "").strip()
            if response_content.endswith("```"):
                response_content = response_content.replace("```", "").strip()
            response_content = fix_invalid_escapes(response_content)

            # Handle empty or invalid JSON responses
            try:
                response_data = json.loads(response_content)

                if response_data.get('score') is True:
                    total_score += 1

                total_responses += 1
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from response_content: {e}")
                continue
    # Calculate the average score (as a percentage)
    average_score = (total_score / total_responses)
    return average_score


def main():
    """
    Main function to parse command-line arguments and execute the selected activity.
    """
    parser = argparse.ArgumentParser(
        description=(
            'CSC 3025 HW2 - Making batch calls to 4o-mini and Llama using Azure and OpenAI'
        )
    )
    parser.add_argument(
        'activity',
        choices=[
            'four_o_mini_answers',
            'llama_answers',
            'four_o_mini_grading',
            'llama_grading',
            'mini_avg',
            'llama_avg'
        ],
        help='Select activity to perform.'
    )

    args = parser.parse_args()

    # Execute the selected activity
    if args.activity == 'four_o_mini_answers':
        four_o_mini_answers()
    elif args.activity == 'llama_answers':
        llama_answers()
    elif args.activity == 'four_o_mini_grading':
        four_o_mini_grading()
    elif args.activity == 'llama_grading':
        llama_grading()
    elif args.activity == 'llama_avg':
        average = calc_avgs('llama_grading_results.jsonl') * 100
        print(f"Llama Score: {average:.2f}%")
    elif args.activity == 'mini_avg':
        average = calc_avgs('4o_grading_results.jsonl') * 100
        print(f"4o Mini Score: {average:.2f}%")


if __name__ == '__main__':
    main()