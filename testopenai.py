
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
# Initialize the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def test_chat_completion():
    """Test 1: Basic Chat Completion"""
    print("Testing Chat Completion...")
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello and tell me one fun fact about space."}
            ],
            max_tokens=150
        )
        
        print(f"Response: {response.choices[0].message.content}")
        print(f"Tokens used: {response.usage}")
    except Exception as e:
        print(f"Error: {e}")


def test_streaming_completion():
    """Test 2: Streaming Response"""
    print("\nTesting Streaming Completion...")
    try:
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Count from 1 to 5 slowly."}],
            stream=True
        )
        
        print("Streaming response: ", end="")
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="")
        print("\n")
    except Exception as e:
        print(f"Error: {e}")


def test_image_generation():
    """Test 3: Image Generation (DALL-E)"""
    print("Testing Image Generation...")
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt="A serene mountain landscape at sunset",
            n=1,
            size="1024x1024"
        )
        
        print(f"Image URL: {response.data[0].url}")
    except Exception as e:
        print(f"Error: {e}")


def test_embeddings():
    """Test 4: Text Embeddings"""
    print("Testing Embeddings...")
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input="The quick brown fox jumps over the lazy dog"
        )
        
        embedding = response.data[0].embedding
        print(f"Embedding dimensions: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
    except Exception as e:
        print(f"Error: {e}")


def test_function_calling():
    """Test 5: Function Calling"""
    print("Testing Function Calling...")
    try:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather in a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA"
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"]
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
            tools=tools,
            tool_choice="auto"
        )
        
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            print(f"Function called: {tool_call.function.name}")
            print(f"Arguments: {tool_call.function.arguments}")
    except Exception as e:
        print(f"Error: {e}")


def test_vision():
    """Test 6: Vision (Image Understanding)"""
    print("\nTesting Vision API...")
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        
        print(f"Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Error: {e}")


def test_json_mode():
    """Test 7: JSON Mode"""
    print("\nTesting JSON Mode...")
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You output JSON only."},
                {"role": "user", "content": "Generate a person with name, age, and city."}
            ],
            response_format={"type": "json_object"}
        )
        
        print(f"JSON Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Run all tests"""
    print("=" * 50)
    print("OpenAI API Testing Suite")
    print("=" * 50)
    
    # Check if API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-api-key'")
        return
    
    test_chat_completion()
    # test_streaming_completion()
    # test_embeddings()
    # test_function_calling()
    # test_json_mode()
    # test_vision()
    # test_image_generation()  # Uncomment if you want to test image generation
    
    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()