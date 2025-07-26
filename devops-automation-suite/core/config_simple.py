"""
DevOps Automation Suite - Simple Configuration
"""

class Settings:
    """Simple application settings"""
    
    app_name = "DevOps Automation Suite"
    app_version = "1.0.0"
    debug = True
    environment = "development"
    
    # Monitoring Thresholds
    alert_cpu_threshold = 80.0
    alert_memory_threshold = 85.0
    alert_disk_threshold = 90.0
    monitoring_interval = 60

# Global settings instance
settings = Settings()
