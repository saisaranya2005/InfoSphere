import boto3
import json
import botocore.config
import base64

s3_bucket = "echoes-of-india-app"
region = "ap-south-1"
model_id = "meta.llama3-70b-instruct-v1:0"

comprehend = boto3.client('comprehend', region_name=region)
translate = boto3.client('translate', region_name=region)
polly = boto3.client('polly', region_name=region)

# List of restricted topics
RESTRICTED_TOPICS = [
    "sex", "condoms", "violence", "drugs", "hate", "racism", "abuse",
    "pornography", "self-harm", "suicide", "profanity", "adult content"
]

def analyze_content(cultural_topic):
    # Check for predefined inappropriate topics
    for keyword in RESTRICTED_TOPICS:
        if keyword in cultural_topic.lower():
            return True  # Content is inappropriate

    # Use AWS Comprehend for additional sentiment analysis
    response = comprehend.detect_sentiment(Text=cultural_topic, LanguageCode='en')
    sentiment = response['Sentiment']

    # Check for negative sentiment
    return sentiment in ['NEGATIVE', 'MIXED']

def generate_cultural_content(cultural_topic):
    prompt = f"<s>[INST]Human: Write a brief overview on the topic {cultural_topic}. Assistant:[/INST]"
    body = {
        "prompt": prompt,
        "max_gen_len": 512,
        "temperature": 0.5,
        "top_p": 0.9
    }

    try:
        bedrock = boto3.client("bedrock-runtime", region_name=region,
                               config=botocore.config.Config(read_timeout=300, retries={'max_attempts': 3}))
        
        response = bedrock.invoke_model(
            body=json.dumps(body),
            modelId=model_id,
            contentType="application/json",
            accept="application/json"
        )

        response_content = response['body'].read()
        response_data = json.loads(response_content)
        return response_data.get('generation', '')
    except Exception as e:
        print(f"Error generating cultural content: {e}")
        return ""

def save_content_to_s3(s3_key, content):
    s3 = boto3.client('s3')
    try:
        s3.put_object(Bucket=s3_bucket, Key=s3_key, Body=content, ContentType="application/json")
    except Exception as e:
        print(f"Error saving content to S3: {e}")

def fetch_content_from_s3(cultural_topic):
    s3 = boto3.client('s3')
    s3_key = f"cultural-content/{cultural_topic.replace(' ', '_')}.json"

    try:
        response = s3.get_object(Bucket=s3_bucket, Key=s3_key)
        return response['Body'].read().decode('utf-8')
    except s3.exceptions.NoSuchKey:
        return ""
    except Exception as e:
        print(f"Error fetching content from S3: {e}")
        return ""

def delete_content_from_s3(cultural_topic):
    s3 = boto3.client('s3')
    s3_key = f"cultural-content/{cultural_topic.replace(' ', '_')}.json"

    try:
        s3.delete_object(Bucket=s3_bucket, Key=s3_key)
        return True
    except Exception as e:
        print(f"Error deleting content from S3: {e}")
        return False

def list_all_generated_content():
    s3 = boto3.client('s3')
    prefix = "cultural-content/"

    try:
        response = s3.list_objects_v2(Bucket=s3_bucket, Prefix=prefix)
        return [content['Key'] for content in response.get('Contents', [])]
    except Exception as e:
        print(f"Error listing content in S3: {e}")
        return []

def translate_content(content, target_language):
    response = translate.translate_text(
        Text=content,
        SourceLanguageCode="en",
        TargetLanguageCode=target_language
    )
    return response.get('TranslatedText', '')

def synthesize_speech(text, language_code="en"):
    response = polly.synthesize_speech(
        Text=text,
        OutputFormat="mp3",
        VoiceId="Joanna" if language_code == "en" else "Aditi"  # Choose voice based on language
    )
    
    audio_stream = response['AudioStream'].read()
    return base64.b64encode(audio_stream).decode('utf-8')  # Encode to base64

def lambda_handler(event, context):
    print(f"Received event: {json.dumps(event)}")
    
    try:
        http_method = event.get('httpMethod', 'POST')  # Default to POST if not specified

        if http_method == 'GET':
            content_history = list_all_generated_content()
            return {
                'statusCode': 200,
                'headers': {'Access-Control-Allow-Origin': '*'},
                'body': json.dumps({
                    'status': 'success',
                    'message': 'Content history retrieved.',
                    'history': content_history
                })
            }

        elif http_method == 'POST':
            event_body = json.loads(event.get('body', '{}'))
            cultural_topic = event_body.get('cultural_topic')
            target_language = event_body.get('target_language', 'en')
            text_to_speech = event_body.get('text_to_speech', False)

            if not cultural_topic:
                return {
                    'statusCode': 400,
                    'headers': {'Access-Control-Allow-Origin': '*'},
                    'body': json.dumps({'error': 'Cultural topic is required.'})
                }

            # Analyze the content for harmful or inappropriate topics
            if analyze_content(cultural_topic):
                return {
                    'statusCode': 400,
                    'headers': {'Access-Control-Allow-Origin': '*'},
                    'body': json.dumps({'error': 'The requested topic contains inappropriate content. Please try again with a different topic.'})
                }

            existing_content = fetch_content_from_s3(cultural_topic)
            if existing_content:
                content_data = json.loads(existing_content)
                content_text = content_data.get("content", "No content found.")

                if target_language != "en":
                    content_text = translate_content(content_text, target_language)

                response_body = {'status': 'success', 'content': content_text}

                if text_to_speech:
                    audio_stream = synthesize_speech(content_text, language_code=target_language)
                    response_body['audio'] = audio_stream  # Base64 encoded audio

                return {
                    'statusCode': 200,
                    'headers': {'Access-Control-Allow-Origin': '*'},
                    'body': json.dumps(response_body)
                }

            generated_content = generate_cultural_content(cultural_topic)
            if generated_content:
                s3_key = f"cultural-content/{cultural_topic.replace(' ', '_')}.json"
                save_content_to_s3(s3_key, json.dumps({"topic": cultural_topic, "content": generated_content}))

                if target_language != "en":
                    generated_content = translate_content(generated_content, target_language)

                response_body = {'status': 'success', 'content': generated_content}

                if text_to_speech:
                    audio_stream = synthesize_speech(generated_content, language_code=target_language)
                    response_body['audio'] = audio_stream  # Base64 encoded audio

                return {
                    'statusCode': 200,
                    'headers': {'Access-Control-Allow-Origin': '*'},
                    'body': json.dumps(response_body)
                }
            else:
                return {
                    'statusCode': 500,
                    'headers': {'Access-Control-Allow-Origin': '*'},
                    'body': json.dumps({'error': 'Failed to generate cultural content.'})
                }

        elif http_method == 'DELETE':
            event_body = json.loads(event.get('body', '{}'))
            cultural_topic = event_body.get('cultural_topic')

            if not cultural_topic:
                return {
                    'statusCode': 400,
                    'headers': {'Access-Control-Allow-Origin': '*'},
                    'body': json.dumps({'error': 'Cultural topic is required for deletion.'})
                }

            # Delete the content from S3
            if delete_content_from_s3(cultural_topic):
                return {
                    'statusCode': 200,
                    'headers': {'Access-Control-Allow-Origin': '*'},
                    'body': json.dumps({
                        'status': 'success',
                        'message': f'Content for topic "{cultural_topic}" has been deleted.'
                    })
                }
            else:
                return {
                    'statusCode': 500,
                    'headers': {'Access-Control-Allow-Origin': '*'},
                    'body': json.dumps({'error': 'Failed to delete the content.'})
                }

        else:
            return {
                'statusCode': 405,
                'headers': {'Access-Control-Allow-Origin': '*'},
                'body': json.dumps({'error': 'Method Not Allowed'})
            }

    except Exception as e:
        print(f"Error processing request: {e}")
        return {
            'statusCode': 500,
            'headers': {'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({'error': 'Internal server error.'})
        }
