import requests
import socket
import time

# Discord webhook URL
discord_webhook_url = 'https://discord.com/api/webhooks/1157127756124008490/zg-IhdWgMR7j-sXB4cGbYXlz0Fx0mvyXuUeIr9_-bybzi_XA5iLoPpNoNd1JjVvBe_cv'


# Function to get the current public IP address
def get_public_ip():
    try:
        # Use a service like 'ipinfo.io' to get the public IP address
        response = requests.get('https://ipinfo.io')
        ip_data = response.json()
        return ip_data['ip']
    except Exception as e:
        print(f"Error: {e}")
        return None


# Function to send a message to Discord webhook
def send_discord_message(old_ip, new_ip):
    message = f"Public IP Address has changed!\n\nOld IP: {old_ip}\nNew IP: {new_ip}"
    data = {
        "content": message
    }
    response = requests.post(discord_webhook_url, json=data)
    if response.status_code == 204:
        print("Message sent successfully to Discord!")
    else:
        print(f"Failed to send message. Status code: {response.status_code}")


# Initialize the old IP address
old_ip = get_public_ip()

while True:
    new_ip = get_public_ip()
    print(f"Old IP: {old_ip}")
    print(f"New IP: {new_ip}")

    if new_ip and old_ip != new_ip:
        send_discord_message(old_ip, new_ip)
        old_ip = new_ip

    time.sleep(300)  # Sleep for 5 minutes (300 seconds)
