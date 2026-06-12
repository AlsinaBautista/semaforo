import asyncio
import websockets

async def test_ws():
    try:
        async with websockets.connect("ws://localhost:9001/mqtt", subprotocols=["mqtt"]) as websocket:
            print("Connected to /mqtt")
    except Exception as e:
        print(f"/mqtt failed: {e}")

    try:
        async with websockets.connect("ws://localhost:9001", subprotocols=["mqtt"]) as websocket:
            print("Connected to /")
    except Exception as e:
        print(f"/ failed: {e}")

asyncio.run(test_ws())
