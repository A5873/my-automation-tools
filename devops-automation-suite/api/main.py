"""
DevOps Automation Suite - FastAPI Entry Point
Main API application setup
"""

from fastapi import FastAPI, APIRouter
from loguru import logger
from core.config import settings


# Create FastAPI app instance
app = FastAPI(title=settings.app_name, version=settings.app_version, debug=settings.debug)

# Include API routes
@app.get("/")
async def root():
    return {"message": "Welcome to the DevOps Automation Suite"}


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Perform health checks for database and other dependencies
    db_status = "healthy"  # Replace with actual check
    redis_status = "healthy"  # Replace with actual check
    return {
        "database": db_status,
        "redis": redis_status,
        "version": settings.app_version,
        "status": "ok" if db_status == "healthy" and redis_status == "healthy" else "degraded"
    }

# Include Log Analyzer, Infrastructure Monitor, and any other modules here
# E.g., router = APIRouter()
# app.include_router(router, prefix="/logs")


# Startup event
