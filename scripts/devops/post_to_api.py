import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
import json
import requests

aws_region = 'us-east-1'
api_url = 'https://8np7tbux24.execute-api.us-east-1.amazonaws.com/v3/log'
payload = {'key': 'value'}

session = boto3.Session()
credentials = session.get_credentials().get_frozen_credentials()

headers = {'Content-Type': 'application/json'}
request = AWSRequest(
    method='POST',
    url=api_url,
    headers=headers,
    data=json.dumps(payload)
)

SigV4Auth(credentials, 'execute-api', aws_region).add_auth(request)

response = requests.post(
    api_url,
    headers=dict(request.headers.items()),
    data=request.body
)

print(response.status_code, response.text)
