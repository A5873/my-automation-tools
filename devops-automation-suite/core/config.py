"""
DevOps Automation Suite - Configuration Management
Enterprise-grade configuration with environment variable support
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    app_name: str = "DevOps Automation Suite"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # Database Configuration
    database_url: str = Field(
        default="postgresql://user:password@localhost:5432/devops_automation",
        env="DATABASE_URL"
    )
    
    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # Security
    secret_key: str = Field(default="dev-secret-key", env="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    
    # Log Analysis Settings
    log_retention_days: int = Field(default=30, env="LOG_RETENTION_DAYS")
    anomaly_threshold: float = Field(default=0.8, env="ANOMALY_THRESHOLD")
    
    # Infrastructure Monitoring
    monitoring_interval: int = Field(default=60, env="MONITORING_INTERVAL")  # seconds
    alert_cpu_threshold: float = Field(default=80.0, env="ALERT_CPU_THRESHOLD")
    alert_memory_threshold: float = Field(default=85.0, env="ALERT_MEMORY_THRESHOLD")
    alert_disk_threshold: float = Field(default=90.0, env="ALERT_DISK_THRESHOLD")
    
    # Security Scanner
    scan_interval: int = Field(default=3600, env="SCAN_INTERVAL")  # seconds
    vulnerability_db_update_interval: int = Field(default=86400, env="VULN_DB_UPDATE_INTERVAL")
    
    # Pipeline Optimizer
    pipeline_analysis_depth: int = Field(default=100, env="PIPELINE_ANALYSIS_DEPTH")
    optimization_threshold: float = Field(default=0.15, env="OPTIMIZATION_THRESHOLD")
    
    # Database Performance
    query_timeout: int = Field(default=30, env="QUERY_TIMEOUT")
    performance_baseline_days: int = Field(default=7, env="PERFORMANCE_BASELINE_DAYS")
    
    # External Integrations
    slack_webhook_url: Optional[str] = Field(default=None, env="SLACK_WEBHOOK_URL")
    email_smtp_server: Optional[str] = Field(default=None, env="EMAIL_SMTP_SERVER")
    email_smtp_port: Optional[int] = Field(default=587, env="EMAIL_SMTP_PORT")
    email_username: Optional[str] = Field(default=None, env="EMAIL_USERNAME")
    email_password: Optional[str] = Field(default=None, env="EMAIL_PASSWORD")
    
    # AWS Configuration (for cloud integrations)
    aws_access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(default="us-east-1", env="AWS_REGION")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings"""
    return Settings()


# Global settings instance
settings = get_settings()
