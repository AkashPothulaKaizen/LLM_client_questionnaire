import json
import boto3
from botocore.exceptions import ClientError
from typing import Optional


class LLMModelCaller:
    def __init__(self, config):
        """
        Initialize the LLMModelCaller with a specified model ID and AWS region.
        Parameters:
        model_id (str): The ID of the model to use for generating the answer.
        region_name (str): The AWS region where the Bedrock runtime is located.
        """
        self.model_id = config.model_id
        self.client = boto3.client("bedrock-runtime",
                                   region_name=config.region_name)

    def question_answer(self, prompt: str) -> Optional[str]:
        """
        Generate an answer to the given prompt using the specified model.
        Parameters:
        prompt (str): The input prompt for the model.
        Returns:
        str: The generated answer from the model.
        """
        # Format the request payload using the model's native structure
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
        # Convert the native request to JSON
        request = json.dumps(native_request)
        try:
            # Invoke the model with the request
            response = self.client.invoke_model(
                modelId=self.model_id, body=request)
            # print(response)
            response_body = json.loads(response["body"].read())
            response_text = response_body["content"][0]["text"]
            return response_text
        except (ClientError, Exception) as e:
            print(f"ERROR: Can't invoke '{self.model_id}'. Reason: {e}")
            return None