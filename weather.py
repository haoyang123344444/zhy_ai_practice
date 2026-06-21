import requests
import yfinance as yf


WEATHER_CODES = {
    0: "晴天",
    1: "大致晴朗",
    2: "局部多云",
    3: "阴天",
    45: "有雾",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    95: "雷暴",
    99: "强雷暴伴冰雹",
}

def geocode_city(city: str):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {
        "name": city,
        "count": 1,
        "language": "zh",
        "format": "json",
    }

    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    if "results" not in data or not data["results"]:
        return None

    result = data["results"][0]

    return {
        "name": result["name"],
        "country": result.get("country"),
        "latitude": result["latitude"],
        "longitude": result["longitude"],
        "timezone": result.get("timezone", "auto"),
    }


def get_weather(city: str):
    location = geocode_city(city)

    if location is None:
        return f"找不到城市：{city}"

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "current": "temperature_2m,weather_code,wind_speed_10m",
        "timezone": location["timezone"],
    }

    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()

    current = data["current"]

    return {
        "tool": "get_weather",
        "city": location["name"],
        "country": location["country"],
        "temperature": current["temperature_2m"],
        "temperature_unit": "C",
        "wind_speed": current["wind_speed_10m"],
        "wind_speed_unit": "km/h",
        "weather_code": current["weather_code"],
        "weather_description": WEATHER_CODES.get(
            current["weather_code"],
            "未知天气"
        ),
    }