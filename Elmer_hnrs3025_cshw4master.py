import os
import json
import chromadb
from chromadb.utils import embedding_functions
import openai
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv('../.env')

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def setup_chroma_db(squad_data):
    """
    Set up ChromaDB using PersistentClient and store context chunks from SQuAD2.0.
    If the collection already exists, use the existing one.
    """
    # Establish ChromaDB PersistentClient
    chroma_client = chromadb.PersistentClient(path="../data/chroma_db")

    # Define OpenAI embedding function
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai.api_key,
        model_name="text-embedding-ada-002"
    )

    # Check if collection exists, otherwise create a new one
    if "squad2_context" in [collection.name for collection in chroma_client.list_collections()]:
        collection = chroma_client.get_collection(name="squad2_context")
    else:
        collection = chroma_client.create_collection(
            name="squad2_context",
            embedding_function=openai_ef
        )

    # Check if the collection is already populated
    if len(collection.get()) == 0:  # If it's empty, populate it
        documents = []
        metadatas = []
        ids = []
        idx = 1  # To generate unique ids

        # Store context chunks in ChromaDB
        for article in squad_data['data']:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                documents.append(context)
                metadatas.append({'article_title': article['title']})
                ids.append(f"doc_{idx}")
                idx += 1

        # Add the documents and their metadata to the collection
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    return collection

def select_tool(query, chat_history):
    """
    Select the appropriate tool based on the user's query and chat history.
    """
    tool_selector_prompt = """You are an AI base in Baton Rouge, Lousiana, who is tasked with selecting the right tool 
to use, based on a user's query. Assume that the user is also based on Baton Rouge, LA.

Here are the available tools and their arguments:

\t 1. rag(query): Select the "rag" tool if the user's query is a question that is not related to the weather.
\t\t - query: String. A rephrased version of the user's question for optimized information retrieval. Only rephrase for grammar or mispelling.
\t 2. summarize: Select the "summarize" tool if the user instructs you to summarize something. This tool has no arguments
\t 3. yodaspeak(sentence): Select the "yodaspeak" tool if the user's query includes something along the lines of "translate into yoda speak" or "Say this like yoda."
\t\t -sentence: String. A sentence that the user wants translated to yoda speak

Here is the user's query: {question}

Your response should be only a valid JSON object, using the following format: 
{{
\t"tool_name": String. The name of the tool, must be either "rag", "summarize", or "yoda",
\t"args": {{
\t\t"the first argument name here, based on tool description above": "The argument value here",
\t\t"the second argument name here, based on tool description above": "The argument value here",
\t\tetc...,}}
}}

Your response: """

    # Prepare the prompt by formatting it with the user's query
    formatted_prompt = tool_selector_prompt.format(question=query)

    messages = [{"role": "system", "content": formatted_prompt}]

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini-2024-07-18",
        messages=messages,
        temperature=0
    )

    tool_response = response['choices'][0]['message']['content'].strip()

    # Parse the JSON response
    try:
        tool_data = json.loads(tool_response)
        tool_name_raw = tool_data.get('tool_name', '').lower()
        args = tool_data.get('args', {})
    except json.JSONDecodeError as e:
        print("Error parsing JSON response:", e)
        print("Response content:", tool_response)
        tool_name_raw = ''
        args = {}

    # Use regular expressions to match the tool name
    tool_name_clean = re.sub(r'[^a-zA-Z]', '', tool_name_raw).lower()

    if tool_name_clean == 'rag':
        tool_name = 'RAG'
    elif tool_name_clean == 'summarize':
        tool_name = 'Summarize'
    elif tool_name_clean == 'yodaspeak' or tool_name_clean == 'yoda':
        tool_name = 'YodaSpeak'
    else:
        tool_name = 'Unknown'

    return tool_name, args

def answer_question(query, chat_history, collection):
    """
    Use RAG to answer the user's question.
    """
    # Rephrase/contextualize the question based on chat history
    rephrase_prompt = """
You are an assistant that helps to rephrase the user's question based on the conversation history.

Conversation history:
{chat_history}

User's question: {query}

Please provide a rephrased question that incorporates the context from the conversation history.
"""
    # Prepare the rephrase prompt
    history_text = '\n'.join([f"{msg['role']}: {msg['content']}" for msg in chat_history if msg['role'] != 'system'])
    prompt = rephrase_prompt.format(chat_history=history_text, query=query)

    # Get the rephrased question
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    rephrased_question = response['choices'][0]['message']['content'].strip()

    # Get the embedding of the rephrased question
    embedding_model = 'text-embedding-ada-002'
    question_embedding = openai.Embedding.create(
        input=rephrased_question,
        model=embedding_model
    )['data'][0]['embedding']

    # Retrieve relevant documents from ChromaDB
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=5
    )

    # Combine retrieved context chunks into a single string
    context = "\n".join(results['documents'][0])

    system_prompt = (
        "You are an intelligent AI Model. Use the provided context to answer the question "
        "as concisely and factually as possible. Prioritize accuracy."
    )
    user_prompt = f"Context:\n{context}\nQuestion: {rephrased_question}"

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )
    answer = response['choices'][0]['message']['content'].strip()
    return answer

def summarize_content(query, chat_history):
    """
    Use Chain of Density to summarize the content requested.
    """

    extract_content_prompt = """
You are an assistant that extracts the content that the user wants to summarize from their query.

User's query: {query}

Please extract the content to summarize.
"""

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": extract_content_prompt.format(query=query)}
        ],
        temperature=0
    )
    content_to_summarize = response['choices'][0]['message']['content'].strip()

    # Apply Chain of Density summarization
    chain_of_density_prompt = """I will provide you with some content.

You will generate increasingly concise, entity-dense summaries of the provided content.

Repeat the following 2 steps 5 times.

Step 1. Identify 1-3 informative Entities (";" delimited) from the Content which are missing from the previously generated summary.
Step 2. Write a new, denser summary of identical length which covers every entity and detail from the previous summary plus the Missing Entities.

A Missing Entity is:
\t1. Relevant: to the main story.
\t2. Specific: descriptive yet concise (5 words or fewer).
\t3. Novel: not in the previous summary.
\t4. Faithful: present in the content piece.
\t5. Anywhere: located anywhere in the Article.

Guidelines:
\t - The first summary should be long (4-5 sentences, ~80 words) yet highly non-specific, containing little information beyond the entities marked as missing. Use overly verbose language and fillers (e.g., "this article discusses") to reach ~80 words.
\t - Make every word count: re-write the previous summary to improve flow and make space for additional entities.
\t - Make space with fusion, compression, and removal of uninformative phrases like "the article discusses".
\t - The summaries should become highly dense and concise yet self-contained, e.g., easily understood without the Article.
\t - Missing entities can appear anywhere in the new summary.
\t - Never drop entities from the previous summary. If space cannot be made, add fewer new entities.

Remember, use the exact same number of words for each summary.

<Content Start>
{content}
<Content End>

Answer only in valid JSON, using the following format:
{{
    "summaries": a list of dictionaries, with each item in the list being another dictionary whose keys are "Missing_Entities" and "Denser_Summary"
}}

Your Response: """

    # Format the prompt with the content to summarize
    formatted_prompt = chain_of_density_prompt.format(content=content_to_summarize)

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": formatted_prompt}
        ],
        temperature=0.5
    )
    summary_json = response['choices'][0]['message']['content'].strip()

    # Parse the JSON response and extract the final summary
    try:
        summary_data = json.loads(summary_json)
        summaries = summary_data.get('summaries', [])
        if summaries:
            final_summary = summaries[-1].get('Denser_Summary', '')
            return final_summary
        else:
            return "No summary could be generated."
    except json.JSONDecodeError as e:
        return f"Error parsing the summary: {e}"


def yoda_speak(sentence, chat_history):
    """
    Translate the user's sentence into Yoda-speak using the language model.
    """
    # Use the language model to generate the Yoda-speak translation
    yoda_prompt = f"""
You are Yoda from Star Wars. Translate the following sentence into Yoda-speak:

"{sentence}"
"""

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": "You are Yoda from Star Wars."},
            {"role": "user", "content": yoda_prompt}
        ],
        temperature=0.7
    )
    yoda_translation = response['choices'][0]['message']['content'].strip()
    return yoda_translation

def main():
    # Load SQuAD2.0 dataset
    with open('dev-v2.0.json') as f:
        squad_data = json.load(f)

    # Setup ChromaDB
    collection = setup_chroma_db(squad_data)

    # Initialize chat history
    chat_history = []

    print("Welcome to the AI assistant. Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        # Update chat history with user's input
        chat_history.append({"role": "user", "content": user_input})

        tool, args = select_tool(user_input, chat_history)

        print(f"Selected tool: {tool}")

        # Use the selected tool
        if tool == 'RAG':
            # Get the 'query' argument from args
            query_arg = args.get('query', user_input)
            answer = answer_question(query_arg, chat_history, collection)
            print(f"Assistant: {answer}")
            chat_history.append({"role": "assistant", "content": answer})
        elif tool == 'Summarize':
            summary = summarize_content(user_input, chat_history)
            print(f"Assistant: {summary}")
            chat_history.append({"role": "assistant", "content": summary})
        elif tool == 'YodaSpeak':
            # Get the 'sentence' argument from args
            sentence_arg = args.get('sentence', user_input)
            yoda_response = yoda_speak(sentence_arg, chat_history)
            print(f"Assistant: {yoda_response}")
            chat_history.append({"role": "assistant", "content": yoda_response})
        else:
            print("Assistant: Sorry, I couldn't understand your request.")
            chat_history.append({"role": "assistant", "content": "Sorry, I couldn't understand your request."})

if __name__ == '__main__':
    main()