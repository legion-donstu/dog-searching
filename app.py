from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI

app = FastAPI()

app.mount("/", StaticFiles(directory="site", html=True))
