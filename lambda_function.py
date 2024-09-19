import json
import boto3
import botocore
import pandas as pd
import io
import numpy as np
from sklearn.neighbors import NearestNeighbors

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



def lambda_handler(event, context):
    # implement
    s3_client = boto3.client("s3")
    bucket = 'akashaudio'
    key = 'vector_store.csv'
    
    response = s3_client.get_object(Bucket=bucket,Key=key)
    file_content = response['Body'].read().decode('utf-8')
    df = pd.read_csv(io.StringIO(file_content))
    document_embeddings_list = df["embeddings"].apply(eval).to_list()
    document_embeddings = np.array(document_embeddings_list)
    
    query= "Do you maintain a record or database of all the assets your organization possesses?"
    query_embeddings_list = embed_content(query)
    print("question is converted to embeddings sucessfully")
    
    query_embeddings = np.array(query_embeddings_list).reshape(1, -1)
    knn = NearestNeighbors(n_neighbors=10)
    knn.fit(document_embeddings)
    distances, indices = knn.kneighbors(query_embeddings)
    retrieved_data = []
    for i, idx in enumerate(indices[0]):
        retrieved_data.append(df["question_answer_data"].iloc[idx])
        
    formatted_data = "\n\n".join(retrieved_data)
    second_formatted_prompt = SecondPrompt.format(question=query, data=formatted_data)
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0"
    Best_answer = question_answer(second_formatted_prompt,model_id)
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
