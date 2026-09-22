import os

import requests
from dotenv import load_dotenv
from smolagents import Tool

load_dotenv()

API_URL = "https://api.openweathermap.org/data/2.5/weather"
TIMEOUT_SECONDS = 10


class WeatherInfoTool(Tool):
    name = "weather_info"
    description = (
        "Fetches the current weather for a location using the OpenWeatherMap API. "
        "Useful for deciding whether outdoor plans, such as fireworks, are sensible."
    )
    inputs = {
        "location": {
            "type": "string",
            "description": "City name, optionally with a country code, e.g. 'Paris' or 'Paris,FR'."
        }
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("OPENWEATHERMAP_API_KEY")

    def forward(self, location: str):
        if not self.api_key:
            return "Weather unavailable: OPENWEATHERMAP_API_KEY is not set in the environment."

        params = {"q": location, "appid": self.api_key, "units": "metric"}
        try:
            response = requests.get(API_URL, params=params, timeout=TIMEOUT_SECONDS)
        except requests.RequestException as exc:
            return f"Weather unavailable for {location}: network error ({exc})."

        # Return the failure to the agent as text rather than raising: a tool that
        # throws kills the run, whereas a message lets the agent try another city.
        if response.status_code == 401:
            return "Weather unavailable: the OpenWeatherMap API key was rejected (new keys take up to 2 hours to activate)."
        if response.status_code == 404:
            return f"Weather unavailable: OpenWeatherMap does not recognise the location '{location}'."
        if response.status_code == 429:
            return "Weather unavailable: OpenWeatherMap rate limit reached (free tier allows 60 calls/minute)."
        if not response.ok:
            return f"Weather unavailable for {location}: OpenWeatherMap returned HTTP {response.status_code}."

        data = response.json()
        condition = data["weather"][0]["description"].capitalize()
        main = data["main"]
        wind_speed = data.get("wind", {}).get("speed")

        report = (
            f"Weather in {data['name']}, {data['sys']['country']}: {condition}, "
            f"{main['temp']:.1f}°C (feels like {main['feels_like']:.1f}°C), "
            f"humidity {main['humidity']}%"
        )
        if wind_speed is not None:
            report += f", wind {wind_speed} m/s"
        return report


# Initialize the tool
weather_info_tool = WeatherInfoTool()

if __name__ == "__main__":
    print(weather_info_tool.forward("Khandwa"))
