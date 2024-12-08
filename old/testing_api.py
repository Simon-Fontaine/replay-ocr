import requests
import json


def test_analyze_replay(
    image_path: str, url: str = "http://localhost:8000/analyze_replay"
):
    """
    Send an image file to the /analyze_replay endpoint and print the response.
    """
    # Prepare the files dictionary for the request
    with open(image_path, "rb") as f:
        files = {"file": (image_path, f, "image/png")}

        try:
            # Send POST request
            response = requests.post(url, files=files)

            # Check if request was successful
            if response.status_code == 200:
                # Attempt to parse JSON
                try:
                    data = response.json()
                    print("Success! Here is the parsed response:")
                    print(json.dumps(data, indent=2))
                except json.JSONDecodeError:
                    print("Received non-JSON response:")
                    print(response.text)
            else:
                print(
                    f"Request failed with status code {response.status_code}: {response.reason}"
                )
                print("Response content:")
                print(response.text)

        except requests.exceptions.RequestException as e:
            print(f"An error occurred while making the request: {e}")


if __name__ == "__main__":
    # Example usage
    test_analyze_replay("overwatch_replay.png")
