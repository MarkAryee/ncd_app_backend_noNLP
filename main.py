from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# Create FastAPI app
app = FastAPI(title="My FastAPI Demo", version="1.0.0")


# Root route → show a simple HTML page
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>FastAPI Demo</title>
        </head>
        <body style="font-family: Arial; text-align: center; margin-top: 50px;">
            <h1>🚀 FastAPI is running!</h1>
            <p>Welcome to my demo app.</p>
            <p>Try these routes:</p>
            <ul style="list-style:none;">
                <li><a href="/">🏠 Home</a></li>
                <li><a href="/hello?name=Mark">👋 /hello</a></li>
                <li><a href="/routes">📜 /routes</a></li>
                <li><a href="/docs">📖 /docs (Swagger UI)</a></li>
            </ul>
        </body>
    </html>
    """


# Example route with query parameter
@app.get("/hello")
def say_hello(name: str = "World"):
    return {"message": f"Hello, {name}!"}


# Route to show all registered endpoints
@app.get("/routes")
def list_routes():
    routes_info = []
    for route in app.routes:
        routes_info.append({"path": route.path, "name": route.name, "methods": list(route.methods)})
    return {"available_routes": routes_info}
