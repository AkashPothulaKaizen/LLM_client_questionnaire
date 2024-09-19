import json
import boto3
import botocore
import pandas as pd
from botocore.exceptions import ClientError
import io
from io import StringIO,BytesIO
import numpy as np
from sklearn.neighbors import NearestNeighbors
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.prompts import PromptTemplate

def embed_content(prompt_data):
    body = json.dumps({"inputText": prompt_data})
    bedrock_runtime = boto3.client('bedrock-runtime')
    modelId = "amazon.titan-embed-text-v1"  # (Change this to try different embedding models)   
    try:
    
        response = bedrock_runtime.invoke_model(
            body=body, modelId=modelId
        )
        response_body = json.loads(response.get("body").read())
    
        embedding = response_body.get("embedding")
        # print(f"The embedding vector has {len(embedding)} values\n{embedding[0:3]+['...']+embedding[-3:]}")
        return embedding
    
    except botocore.exceptions.ClientError as error:
    
        if error.response['Error']['Code'] == 'AccessDeniedException':
               print(f"\x1b[41m{error.response['Error']['Message']}\
                    \nTo troubeshoot this issue please refer to the following resources.\
                     \nhttps://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\
                     \nhttps://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html\x1b[0m\n")
    
        else:
            raise error

def question_answer(prompt,model_id):
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    
    # Format the request payload using the model's native structure.
    native_request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
    }
    # Convert the native request to JSON.
    request = json.dumps(native_request)
    try:
        # Invoke the model with the request.
        response = client.invoke_model(modelId=model_id, body=request)
        # print(response)
        # print(response)
        # print(f"response is {response}")
        model_response = json.loads(response["body"].read())
        response_text = model_response["content"][0]["text"]
        return response_text
    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        exit(1)
        
SecondPrompt = PromptTemplate(
    input_variables=["question", "data"],
    template="""You are an AI assistant working for Kaizen Organization. Your task is to answer questions based on the provided data. Follow these steps to ensure accuracy and relevance:

    Question: {question}
    Data: {data}

    
    1. **Understand the Question**: Carefully read the question to grasp what information is being sought.
    2. **Analyze the Data**: Examine the provided data thoroughly to identify the most relevant piece of information that directly answers the question.
    3. **Extract the Answer**: Select the single most relevant piece of information from the data that answers the question. If the answer cannot be found in the data, clearly state "The information required to answer is missing from the data."
    4. make sure to extract correct Serial_number 

    
    Please ensure that your response is in JSON format with the following structure. Include only the most relevant details, formatted as a single JSON object. 
    
    Your response should be in this exact format:
    
    {{
        "user_question": "{question}",
        "Serial_number": 5,
        "Question": "question",
        "Yes/No": "Yes",
        "Answer": "The relevant answer based on the data.",
        "Owner": "Name of the person or department responsible",
        "Category": "Category of the information",
        "Subsidiaries": "Name of the subsidiary, if applicable",
        "Last Reviewed": "Date of the last review"
    }}

    If the information required to answer is missing from the data, use the following JSON format:
    
    {{
        "user_question": "{question}",
        "Serial_number": Not Available,
        "Question": "question",
        "Yes/No": "No",
        "Answer": "The information required to answer is missing from the data. Please provide more relevant data in the prompt",
        "Owner": "Not Available",
        "Category": "Not Available",
        "Subsidiaries": "Not Available",
        "Last Reviewed": "Not Available"
    }}
    

    Write only the JSON output and nothing more. Do not include additional text, explanations, or formatting. Ensure the output is a single JSON object that aligns with the structure provided. Here is the JSON output:
    """,
)


def lambda_handler(event, context):
    
    s3_client_query = boto3.client("s3")
    source_bucket = event['Records'][0]['s3']['bucket']['name']
    source_key = event['Records'][0]['s3']['object']['key']
    print(f"source bucket is {source_bucket}")
    
    response = s3_client_query.get_object(Bucket=source_bucket, Key=source_key)
    query_file_content = response['Body'].read()
    df_query = pd.read_excel(BytesIO(query_file_content))
    print(f"data frame is : {df_query}")
    


    # implement
    s3_client = boto3.client("s3")
    bucket = 'akashaudio'
    key = 'vector_store.csv'
    
    response = s3_client.get_object(Bucket=bucket,Key=key)
    file_content = response['Body'].read().decode('utf-8')
    df = pd.read_csv(io.StringIO(file_content))
    document_embeddings_list = df["embeddings"].apply(eval).to_list()
    document_embeddings = np.array(document_embeddings_list)
    
    # Assuming the query is in the first cell of the first sheet
    query = df_query.iloc[0, 0]
    print(f"query is {query}")
    
    
    # query= "Do you maintain a record or database of all the assets your organization possesses?"
    query_embeddings_list = embed_content(query)
    print("question is converted to embeddings sucessfully")
    
    query_embeddings = np.array(query_embeddings_list).reshape(1, -1)
    knn = NearestNeighbors(n_neighbors=10)
    knn.fit(document_embeddings)
    distances, indices = knn.kneighbors(query_embeddings)
    retrieved_data = []
    for i, idx in enumerate(indices[0]):
        retrieved_data.append(df["question_answer_data"].iloc[idx])
    print("data is retrieved")
    formatted_data = "\n\n".join(retrieved_data)
    second_formatted_prompt = SecondPrompt.format(question=query, data=formatted_data)
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0"
    Best_answer = question_answer(second_formatted_prompt,model_id)
    print("LLM gave the answer")
    json_format= json.loads(Best_answer)
    df = pd.DataFrame([json_format])
    
    # Convert DataFrame to CSV in memory
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    
    # Upload the CSV file to S3 bucket
    s3_client = boto3.client('s3')
    bucket_name = 'akashdemos3bucket'
    s3_client.put_object(Bucket=bucket_name, Key='output.csv', Body=csv_buffer.getvalue())
    
    # print(f"CSV file has been successfully uploaded to the S3 bucket '{bucket_name}'")
    
    return {
        'statusCode': 200,
        'body': json.dumps('CSV file has been successfully uploaded to the S3 bucket')
    }
