from src.api import app

def handler(event, context):
    
    return {
        "statusCode": 200,
        "body": "Hello from FastAPI on AWS Lambda!",
    }