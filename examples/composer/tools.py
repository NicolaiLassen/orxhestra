"""Example tools referenced by compose.yaml."""


async def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"


async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny, 22C."


async def lookup_order(order_id: str) -> str:
    """Look up an order by ID."""
    return f"Order {order_id}: shipped, arriving tomorrow."


async def check_billing(account_id: str) -> str:
    """Check billing status for an account."""
    return f"Account {account_id}: paid, next invoice in 30 days."
