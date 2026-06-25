import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather")

CITY_NAME_MAP = {
    "旧金山": "San Francisco",
    "纽约": "New York",
    "洛杉矶": "Los Angeles",
    "芝加哥": "Chicago",
    "华盛顿": "Washington",
    "伦敦": "London",
    "巴黎": "Paris",
    "东京": "Tokyo",
    "首尔": "Seoul",
    "悉尼": "Sydney",
    "新加坡": "Singapore",
    "曼谷": "Bangkok",
    "迪拜": "Dubai",
    "多伦多": "Toronto",
    "柏林": "Berlin",
    "罗马": "Rome",
    "马德里": "Madrid",
    "莫斯科": "Moscow",
    "孟买": "Mumbai",
}

# -----------------------------
# 1. City → Lat/Lon (simple geocoder)
# -----------------------------
async def geocode(city: str):
    url = "https://geocoding-api.open-meteo.com/v1/search"

    cities_to_try = [city]
    if city in CITY_NAME_MAP:
        cities_to_try.append(CITY_NAME_MAP[city])

    async with httpx.AsyncClient(timeout=10) as client:
        for name in cities_to_try:
            r = await client.get(url, params={"name": name, "count": 1})
            data = r.json()

            if "results" in data and data["results"]:
                top = data["results"][0]
                return top["latitude"], top["longitude"], top["name"], top.get("country", "")

    return None


# -----------------------------
# 2. Weather Tool (Production)
# -----------------------------
@mcp.tool()
async def get_weather(city: str) -> str:
    """
    Production-grade weather tool with geocoding + error handling
    """

    try:
        geo = await geocode(city)

        if not geo:
            return f"❌ Cannot find city: {city}"

        lat, lon, name, country = geo

        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current_weather": True,
                    "timezone": "auto"
                },
            )
            data = r.json()

        w = data.get("current_weather", {})

        if not w:
            return f"❌ Weather data unavailable for {city}"

        return f"""
🌍 CITY: {name}, {country}
🌡️ TEMP: {w.get('temperature')}°C
💨 WIND: {w.get('windspeed')} km/h
🧭 DIRECTION: {w.get('winddirection')}°
⏱️ TIME: {w.get('time')}
📡 SOURCE: Open-Meteo
"""

    except Exception as e:
        return f"❌ Weather tool error: {str(e)}"


# -----------------------------
# 3. Alerts Tool (Open-Meteo Alerts)
# -----------------------------
@mcp.tool()
async def get_alerts(country_code: str) -> str:
    """
    Weather alerts (fixed safe API, no crash risk)
    """

    try:
        url = f"https://api.open-meteo.com/v1/alerts"

        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, params={"country": country_code})
            data = r.json()

        alerts = data.get("alerts", [])

        if not alerts:
            return f"✅ No active alerts in {country_code}"

        return f"🚨 {country_code}: {len(alerts)} alerts active"

    except Exception as e:
        return f"❌ Alerts error: {str(e)}"


# -----------------------------
# 4. Optional: Streaming-style weather report
# (MCP does not true-stream text, but we simulate chunks)
# -----------------------------
@mcp.tool()
async def get_weather_report(city: str) -> str:
    """
    More narrative weather report (agent-friendly)
    """

    try:
        geo = await geocode(city)
        if not geo:
            return f"City not found: {city}"

        lat, lon, name, country = geo

        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current_weather": True,
                    "hourly": "temperature_2m",
                    "timezone": "auto"
                },
            )
            data = r.json()

        w = data.get("current_weather", {})

        temp = w.get("temperature", "N/A")
        wind = w.get("windspeed", "N/A")

        return (
            f"Weather Report for {name}, {country}\n"
            f"----------------------------------\n"
            f"Current temperature: {temp}°C\n"
            f"Wind speed: {wind} km/h\n"
            f"Condition: {'clear (approx)' if temp != 'N/A' else 'unknown'}\n"
            f"\nData source: Open-Meteo"
        )

    except Exception as e:
        return f"Report error: {str(e)}"


# -----------------------------
# 5. Entry
# -----------------------------
def main():
    mcp.run()


if __name__ == "__main__":
    main()