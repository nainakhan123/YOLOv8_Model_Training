import boto3

from starlette.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile


app=FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

s3=boto3.client('s3')

bucket_name='fastapiimages'

def create_upload_file(file: UploadFile = File(...)):

    file_name = file.file
    s3.upload_file(file.file, bucket_name, file_name)
    print("File uploaded")
    return {"filename": file_name}