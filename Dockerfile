FROM public.ecr.aws/lambda/python:3.12

# Copy requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install the specified packages
RUN pip install -r requirements.txt

# Copy function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

COPY retrieve_data_helper.py ${LAMBDA_TASK_ROOT}
COPY call_llm_model_helper.py ${LAMBDA_TASK_ROOT}
COPY s3_helper.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your lambda_handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "lambda_function.lambda_handler" ]