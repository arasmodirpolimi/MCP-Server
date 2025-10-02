from mcp.server.fastmcp import FastMCP
import os, json, requests
from typing import Dict, Any
from dotenv import load_dotenv
load_dotenv()
mcp = FastMCP("weather")
def ensure_env(name: str, default: str | None = None) -> str:
    val = os.getenv(name)
    if val:
        return val
    if default is not None:
        os.environ[name] = default
        return default
    raise RuntimeError(f"{name} is not set")
@mcp.tool()
def get_current_weather(location: str, unit: str = "celsius") -> Dict[str, Any]:
    key = ensure_env("OPENWEATHER_API_KEY")
    base_url = os.getenv("OPENWEATHER_BASE_URL") or "https://api.openweathermap.org/data/2.5/weather"
    unit_map = {"celsius": "metric", "fahrenheit": "imperial"}
    owm_unit = unit_map.get(unit.lower(), "metric")
    params = {"q": location, "units": owm_unit, "appid": key}
    resp = requests.get(base_url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    resolved_name = data.get("name") or location
    temp = (data.get("main") or {}).get("temp")
    weather_list = data.get("weather")
    forecast = [w.get("description") for w in weather_list if isinstance(w, dict) and w.get("description")] if isinstance(weather_list, list) else []
    return {
        "location": resolved_name,
        "temperature": temp,
        "unit": "celsius" if owm_unit == "metric" else "fahrenheit",
        "forecast": forecast
    }
if __name__ == "__main__":
    mcp.run(transport='stdio')
