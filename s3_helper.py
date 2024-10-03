import boto3
import pandas as pd
from io import BytesIO, StringIO
import uuid


class S3Handler:
    def __init__(self, config):
        self.s3_client = boto3.client("s3")
        self.config = config

    def save_data_to_s3(self, data_frame: pd.DataFrame) -> None:
        """
        Save the given DataFrame to the specified S3 bucket.
        Parameters:
        data_frame (pd.DataFrame): The DataFrame to be saved.
        """
        if not isinstance(self.config.target_bucket, str):
            raise TypeError("target_bucket must be strings")
        csv_buffer = StringIO()
        data_frame.to_csv(csv_buffer, index=False)
        file_key = f"answers-file-{uuid.uuid4()}.csv"
        self.s3_client.put_object(Bucket=self.config.target_bucket,
                                  Key=file_key, Body=csv_buffer.getvalue())
        print(f"CSV file has been successfully uploaded to the S3 bucket "
              f"'{self.config.target_bucket}' with key '{file_key}'")
