#!/usr/bin/env python3
"""
DevOps Automation Suite - Main CLI Entry Point
Enterprise-grade DevOps automation toolkit with AI-powered insights
"""

import typer
from typing import Optional
from rich.console import Console
from rich.table import Table
from loguru import logger

from core.config import settings
from core.database import init_database, test_database_connection, test_redis_connection
from log_analyzer.log_analyzer import analyze_logs
from infrastructure_monitor.infrastructure_monitor import start_infrastructure_monitor

# Initialize CLI app
app = typer.Typer(
    name="DevOps Automation Suite",
    help="🚀 Enterprise-grade DevOps automation toolkit with AI-powered insights",
    add_completion=False
)

console = Console()


@app.command()
def setup():
    """🔧 Initialize the DevOps Automation Suite"""
    console.print("🚀 [bold blue]Setting up DevOps Automation Suite...[/bold blue]")
    
    try:
        # Initialize database
        init_database()
        console.print("✅ Database initialized successfully")
        
        # Test connections
        if test_database_connection():
            console.print("✅ Database connection verified")
        else:
            console.print("❌ Database connection failed")
            
        if test_redis_connection():
            console.print("✅ Redis connection verified")
        else:
            console.print("❌ Redis connection failed")
            
        console.print("🎉 [bold green]Setup completed successfully![/bold green]")
        
    except Exception as e:
        console.print(f"❌ [bold red]Setup failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def analyze_logs_cmd():
    """📊 Run log analysis and anomaly detection"""
    console.print("📊 [bold blue]Starting log analysis...[/bold blue]")
    
    try:
        analyze_logs()
        console.print("✅ [bold green]Log analysis completed[/bold green]")
    except Exception as e:
        console.print(f"❌ [bold red]Log analysis failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def monitor():
    """🖥️ Start infrastructure monitoring"""
    console.print("🖥️ [bold blue]Starting infrastructure monitoring...[/bold blue]")
    
    try:
        start_infrastructure_monitor()
    except KeyboardInterrupt:
        console.print("🛑 [bold yellow]Monitoring stopped by user[/bold yellow]")
    except Exception as e:
        console.print(f"❌ [bold red]Infrastructure monitoring failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def status():
    """📈 Show system status and health"""
    console.print("📈 [bold blue]DevOps Automation Suite Status[/bold blue]")
    
    # Create status table
    table = Table()
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Details", style="white")
    
    # Test database
    db_status = "✅ Healthy" if test_database_connection() else "❌ Unhealthy"
    table.add_row("Database", db_status, settings.database_url)
    
    # Test Redis
    redis_status = "✅ Healthy" if test_redis_connection() else "❌ Unhealthy"
    table.add_row("Redis", redis_status, settings.redis_url)
    
    # Show application info
    table.add_row("Version", settings.app_version, "Current application version")
    table.add_row("Environment", settings.environment, "Runtime environment")
    
    console.print(table)


@app.command()
def web():
    """🌐 Start web API server"""
    console.print("🌐 [bold blue]Starting web API server...[/bold blue]")
    
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )


if __name__ == "__main__":
    app()
